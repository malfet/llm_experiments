# Taken from llama code and lightly modified
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# Self contained script that can be used to benchmark PyTorch inference speed

import array
import os
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, Tuple
from urllib.request import urlopen

import torch
import torch.nn as nn
from sentencepiece import SentencePieceProcessor


@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape))
        self.register_buffer("v_cache", torch.zeros(cache_shape))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        if input_pos.numel() == 1 and input_pos.item() >= k_out.shape[2]:
            self.k_cache = k_out = torch.roll(k_out, 1, 2)
            self.v_cache = v_out = torch.roll(v_out, 1, 2)
            minus_one = torch.tensor([k_out.shape[2] - 1], device=k_out.device)
            k_out[:, :minus_one] = k_val
            v_out[:, :minus_one] = v_val
        else:
            k_out[:, :, input_pos] = k_val
            v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.kv_cache = None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        mask: torch.Tensor,
        input_pos: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Update cache
        if self.kv_cache is not None and input_pos is not None:
            xk, xv = self.kv_cache.update(input_pos, xk, xv)

        # flash implementation
        output = torch.nn.functional.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=mask,
            dropout_p=0,
        )
        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin, mask, input_pos):
        h = x + self.attention.forward(
            self.attention_norm(x), freqs_cos, freqs_sin, mask, input_pos
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # Initialize caches
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len
        )
        causal_mask = torch.tril(
            torch.ones(
                self.params.max_seq_len, self.params.max_seq_len, dtype=torch.bool
            )
        )
        for block in self.layers:
            block.attention.kv_cache = KVCache(
                1,
                self.params.max_seq_len,
                self.params.n_heads,
                self.params.dim // self.params.n_heads,
            )
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self, tokens: torch.Tensor, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        if input_pos is not None and input_pos.numel() > 1:
            freqs_cos = self.freqs_cos[input_pos]
            freqs_sin = self.freqs_sin[input_pos]
            mask = self.causal_mask[None, None, input_pos]
        elif input_pos is not None:
            freqs_cos = self.freqs_cos[input_pos % self.params.max_seq_len]
            freqs_sin = self.freqs_sin[input_pos % self.params.max_seq_len]
            mask = self.causal_mask[
                None,
                None,
                input_pos
                if input_pos.item() < self.params.max_seq_len
                else torch.tensor([-1]),
            ]
        else:
            freqs_cos = self.freqs_cos[:seqlen]
            freqs_sin = self.freqs_sin[:seqlen]
            mask = self.causal_mask[None, None, :seqlen, :seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, mask, input_pos)
        h = self.norm(h)

        # inference-time mini-optimization: only forward the output on the very last position
        logits = self.output(
            h[:, [-1], :]
        )  # note: using list [-1] to preserve the time dim

        return logits

    @torch.inference_mode()
    def generate(self, idx, temperature=1.0, top_k=None):
        def logits_to_idx(logits):
            logits = logits[:, -1, :]  # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
                return idx_next
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = nn.functional.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        # Initialize cache state
        logits = self(idx, input_pos=torch.arange(0, idx.size(1)))
        idx_next = logits_to_idx(logits)
        yield idx_next.item()
        input_pos = torch.tensor(
            [
                idx.size(1),
            ]
        )
        while True:
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_next, input_pos=input_pos)
            idx_next = logits_to_idx(logits)
            yield idx_next.item()
            input_pos += 1


class Tokenizer:
    def __init__(self, model_path=None):
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert isinstance(s, str)
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def decode_id(self, t: int) -> str:
        if t == self.bos_id:
            return "\n\n"
        rc = self.sp_model.IdToPiece(t)
        # Sentencepiece uses Lower One Eighth Block (U+2581) as whitespace
        return rc.replace("\u2581", " ") if rc != "<0x0A>" else "\n"


def download_url(url: str) -> None:
    fname = os.path.basename(url)
    if os.path.exists(fname):
        return
    with urlopen(url) as s, open(fname, "wb") as f:
        f.write(s.read())


def get_file_size(name: str) -> int:
    with open(name, "rb") as f:
        return f.seek(0, 2)


def untyped_storage_from_file(name: str) -> torch.storage.UntypedStorage:
    return torch.storage.UntypedStorage.from_file(name, False, get_file_size(name))


def tensor_from_storage(storage, dtype, offs, shape, strides=None) -> torch.Tensor:
    rc = torch.tensor([], dtype=dtype)
    if strides is None:
        strides = []
        prev = 1
        for s in reversed(shape):
            strides.insert(0, prev)
            prev *= s
    rc.set_(storage, offs, shape, strides)
    return rc


class SerializedReader:
    def __init__(self, storage) -> None:
        self.storage = storage
        self.offs = 0

    def _read_bytes(self, byte_length: int) -> bytes:
        data = bytes(
            array.array("B", self.storage[self.offs : self.offs + byte_length])
        )
        self.offs += byte_length
        return data

    def _read_int(self, byte_length: int, unsigned: bool) -> int:
        fmt = "i" if byte_length == 4 else "l"
        if unsigned:
            fmt = fmt.upper()
        return struct.unpack(fmt, self._read_bytes(byte_length))[0]

    def read_int32(self) -> int:
        return self._read_int(4, False)

    def read_int64(self) -> int:
        return self._read_int(8, False)

    def read_float32(self) -> float:
        return struct.unpack("f", self._read_bytes(4))[0]

    def read_pascal_string(self) -> str:
        strlen = self.read_int64()
        rc = self._read_bytes(strlen)
        return rc.decode("utf-8")

    def read_typed_value(self, val_type: Optional[int] = None) -> Any:
        if val_type is None:
            val_type = self.read_int32()

        if val_type == 9:  # GGUFValueType.ARRAY:
            elem_type = self.read_int32()
            array_len = self.read_int64()
            rc = []
            for _ in range(array_len):
                rc.append(self.read_typed_value(elem_type))
            return rc
        elif val_type == 8:  # GGUFValueType.STRING:
            return self.read_pascal_string()
        elif val_type == 5:  # GGUFValueType.INT32:
            return self.read_int32()
        elif val_type == 4:  # GGUFValueType.UINT32:
            return self._read_int(4, True)
        elif val_type == 6:  # GGUFValueType.FLOAT32:
            return self.read_float32()
        raise RuntimeError(f"Unknown type {val_type}")


class GGUFReader:
    def __init__(self, filename) -> None:
        self.storage = storage = untyped_storage_from_file(filename)
        self.stream = stream = SerializedReader(storage)
        self.props = {}
        self.tensors = {}
        magic = stream.read_int32()
        if magic != 0x46554747:
            raise RuntimeError(f"Unexpected magic number {magic}")
        version = stream.read_int32()
        if version not in [3]:
            raise RuntimeError(f"Unsupported version {version}")
        self.tensor_count = stream.read_int64()
        self.kv_count = stream.read_int64()

        for i in range(self.kv_count):
            key = stream.read_pascal_string()
            value = stream.read_typed_value()
            self.props[key] = value

        tensor_meta = []
        for i in range(self.tensor_count):
            key = stream.read_pascal_string()
            ndim = stream.read_int32()
            dims = list(reversed([stream.read_int64() for _ in range(ndim)]))
            tensor_dtype = stream.read_int32()
            tensor_offs = stream.read_int64()
            tensor_meta.append((key, dims, tensor_dtype, tensor_offs))

        self.alignment = self.props.get("general.alignment", 32)
        if (padding := stream.offs % self.alignment) != 0:
            stream.offs += self.alignment - padding

        for key, shape, raw_dtype, offs in tensor_meta:
            offs = offs + stream.offs
            if raw_dtype == 0:
                self.tensors[key] = tensor_from_storage(
                    storage, torch.float32, offs // 4, shape
                )
            elif raw_dtype == 1:
                self.tensors[key] = tensor_from_storage(
                    storage, torch.float16, offs // 2, shape
                )
            else:
                raise RuntimeError(f"Unknown tensor type {raw_dtype}")


@contextmanager
def default_dtype(dtype=None):
    orig_dtype = torch.get_default_dtype()
    try:
        if dtype is not None:
            torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(orig_dtype)


def model_from_gguf(model_path: str) -> nn.Module:
    gguf = GGUFReader(model_path)
    model_arch = gguf.props["general.architecture"]
    if model_arch != "llama":
        raise RuntimeError(f"Unsupported model architecutre {model_arch}")
    llama_args = ModelArgs(
        dim=gguf.props["llama.embedding_length"],
        n_heads=gguf.props["llama.attention.head_count"],
        n_layers=gguf.props["llama.block_count"],
        max_seq_len=gguf.props["llama.context_length"],
    )

    def gguf_to_model_name(name: str) -> str:
        if name == "token_embd.weight":
            return "tok_embeddings.weight"
        if name == "output_norm.weight":
            return "norm.weight"
        if name.startswith("blk."):
            blk, idx, suffix, weight = name.split(".")
            suffix = {
                "attn_q": "attention.wq",
                "attn_k": "attention.wk",
                "attn_v": "attention.wv",
                "attn_output": "attention.wo",
                "attn_norm": "attention_norm",
                "ffn_up": "feed_forward.w1",
                "ffn_down": "feed_forward.w2",
                "ffn_gate": "feed_forward.w3",
                "ffn_norm": "ffn_norm",
            }[suffix]
            return f"layers.{idx}.{suffix}.weight"
        return name

    state_dict = {gguf_to_model_name(k): v for (k, v) in gguf.tensors.items()}
    return llama_args, state_dict


def model_from_pth(model_path: str) -> nn.Module:
    checkpoint_dict = torch.load(
        model_path, map_location="cpu", weights_only=True, mmap=True
    )
    if "model_args" in checkpoint_dict:
        model_args = checkpoint_dict["model_args"]
        if "n_kv_heads" in model_args:
            assert model_args["n_heads"] == model_args["n_kv_heads"]
            del model_args["n_kv_heads"]
        gptconf = ModelArgs(**model_args)
        state_dict = checkpoint_dict["model"]
    else:
        gptconf = ModelArgs()
        state_dict = checkpoint_dict
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    return gptconf, state_dict


def load_model(
    model_path: str, device: str, dtype: Optional[torch.dtype] = None
) -> nn.Module:
    start_time = datetime.now()
    with default_dtype(dtype), torch.device(device):
        if model_path.endswith(".gguf"):
            conf, state_dict = model_from_gguf(model_path)
        else:
            conf, state_dict = model_from_pth(model_path)
        model = Transformer(conf)
        model.load_state_dict(state_dict, strict=False)
    duration = (datetime.now() - start_time).total_seconds()
    print(f"Loaded {model_path} in {duration:.2f} seconds")
    return model


def run_inference(
    model_path: str = "stories15M.pt",
    tokenizer_path: str = "tokenizer.model",
    prompt: str = "Once upon a time",
    device: str = "cpu",
    dtype: Optional[str] = None,
    seqlen: int = 512,
) -> None:
    model = load_model(
        model_path, device, getattr(torch, dtype) if dtype is not None else None
    )
    tokenizer = Tokenizer(tokenizer_path)
    tokens = tokenizer.encode(prompt, bos=False, eos=False)
    x = torch.tensor(tokens, device=device).reshape(1, -1)
    print(prompt, end="", flush=True)
    start_time = datetime.now()
    for idx, tok in enumerate(model.generate(x)):
        if idx > seqlen:
            print("", flush=True)
            break
        print(tokenizer.decode_id(tok), end="", flush=True)
    duration = (datetime.now() - start_time).total_seconds()
    print(f"Speed is {seqlen/duration:.2f} tokens per second")


def benchmark(
    model_path: str = "stories15M.pt",
    device: str = "cpu",
    dtype: Optional[str] = None,
) -> None:
    model = load_model(model_path, device)
    if dtype is not None:
        model.to(dtype=getattr(torch, dtype))
    x = torch.randint(3, 512, (1, model.params.max_seq_len), device=device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
    ) as prof:
        for idx, tok in enumerate(model.generate(x)):
            break
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total"))


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser("Simple LLM text generator")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-path", type=str, default="stories15M.pt")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--benchmark", action="store_true")
    # Do not attempt to parse CLI arguments if running inside notebook
    return parser.parse_args([] if hasattr(__builtins__, "__IPYTHON__") else None)


if __name__ == "__main__":
    args = parse_args()

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    download_url("https://github.com/karpathy/llama2.c/raw/master/tokenizer.model")
    download_url(
        "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt"
    )

    # Attempt to use reduced precision computations when possible
    if hasattr(torch._C, "_set_cpu_allow_fp16_reduced_precision_reduction"):
        try:
            torch._C._set_cpu_allow_fp16_reduced_precision_reduction(True)
        except RuntimeError:
            print("Failed to set reduced precision computations")

    if args.benchmark:
        benchmark(device=args.device, dtype=args.dtype, model_path=args.model_path)
    else:
        run_inference(
            device=args.device,
            dtype=args.dtype,
            model_path=args.model_path,
            prompt=args.prompt,
            seqlen=args.seq_len,
        )

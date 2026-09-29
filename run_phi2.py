import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer


class PhiAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg["num_attention_heads"]
        self.head_dim = cfg["hidden_size"] // self.n_heads
        self.rotary_dim = int(cfg["partial_rotary_factor"] * self.head_dim)
        dim = cfg["hidden_size"]
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.dense = nn.Linear(dim, dim)

    def forward(self, x, cos, sin, kv_cache):
        bsz, seqlen, _ = x.shape
        q, k, v = (proj(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
                   for proj in (self.q_proj, self.k_proj, self.v_proj))
        q, k = apply_partial_rope(q, cos, sin, self.rotary_dim), apply_partial_rope(k, cos, sin, self.rotary_dim)
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v
        # Causal mask is only needed when processing more than one new token (prefill)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=seqlen > 1)
        return self.dense(out.transpose(1, 2).reshape(bsz, seqlen, -1))


class PhiMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"])
        self.fc2 = nn.Linear(cfg["intermediate_size"], cfg["hidden_size"])

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class PhiDecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(cfg["hidden_size"], eps=cfg["layer_norm_eps"])
        self.self_attn = PhiAttention(cfg)
        self.mlp = PhiMLP(cfg)

    def forward(self, x, cos, sin, kv_cache):
        # Phi uses a parallel block: attention and MLP share one LayerNorm and are summed
        h = self.input_layernorm(x)
        return x + self.self_attn(h, cos, sin, kv_cache) + self.mlp(h)


class PhiModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(PhiDecoderLayer(cfg) for _ in range(cfg["num_hidden_layers"]))
        self.final_layernorm = nn.LayerNorm(cfg["hidden_size"], eps=cfg["layer_norm_eps"])


class PhiForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = PhiModel(cfg)
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"])
        self.register_buffer("inv_freq", self.rope_inv_freq(), persistent=False)

    def rope_inv_freq(self):
        rotary_dim = self.model.layers[0].self_attn.rotary_dim
        return 1.0 / (self.model.cfg["rope_theta"] ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))

    @classmethod
    def from_pretrained(cls, repo_id, dtype=None, device="cpu"):
        path = Path(snapshot_download(repo_id, allow_patterns=["*.json", "*.safetensors"]))
        cfg = json.loads((path / "config.json").read_text())
        # Like transformers' dtype="auto": default to the checkpoint's dtype
        dtype = dtype or getattr(torch, cfg["torch_dtype"])
        with torch.device("meta"):
            model = cls(cfg)
        state_dict = {}
        for f in sorted(path.glob("*.safetensors")):
            state_dict.update(load_file(f, device=str(device)))
        model.load_state_dict({k: v.to(dtype) for k, v in state_dict.items()}, assign=True)
        # Non-persistent buffers are not in the checkpoint, so recompute inv_freq off the meta device
        with torch.device(device):
            model.inv_freq = model.rope_inv_freq()
        return model.eval()

    def forward(self, input_ids, start_pos=0, kv_caches=None, exit_layer=None):
        """Returns last-position logits of the full network and, if `exit_layer` is set,
        of final LayerNorm + LM head applied to the hidden state after `exit_layer` layers."""
        seqlen = input_ids.shape[1]
        pos = torch.arange(start_pos, start_pos + seqlen, device=input_ids.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        dtype = self.lm_head.weight.dtype
        cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)

        x = self.model.embed_tokens(input_ids)
        exit_logits = None
        for i, layer in enumerate(self.model.layers):
            x = layer(x, cos, sin, kv_caches[i] if kv_caches is not None else None)
            if i + 1 == exit_layer:
                exit_logits = self.logits(x[:, -1:])
        return self.logits(x[:, -1:]), exit_logits

    def logits(self, hidden):
        return self.lm_head(self.model.final_layernorm(hidden)).float()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_partial_rope(x, cos, sin, rotary_dim):
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    return torch.cat([x_rot * cos + rotate_half(x_rot) * sin, x_pass], dim=-1)


@torch.inference_mode()
def generate(model, input_ids, max_new_tokens, exit_layer):
    """Greedy decode with the full network; also record what the early exit would predict at each step."""
    kv_caches = [{} for _ in model.model.layers]
    logits, exit_logits = model(input_ids, 0, kv_caches, exit_layer)
    new_tokens, exit_tokens = [], []
    for _ in range(max_new_tokens):
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        new_tokens.append(next_token.item())
        exit_tokens.append(exit_logits[:, -1].argmax(dim=-1).item())
        logits, exit_logits = model(next_token, input_ids.shape[1] + len(new_tokens) - 1, kv_caches, exit_layer)
    return new_tokens, exit_tokens


def side_by_side(tokenizer, tokens, exit_tokens, left_title, right_title, width=60):
    """Aligns step-by-step: line breaks follow the full model's tokens, so each row pairs predictions from the same steps."""
    rows, left, right = [], "", ""
    for tok, exit_tok in zip(tokens, exit_tokens):
        text, exit_text = tokenizer.decode([tok]), tokenizer.decode([exit_tok]).replace("\n", "⏎")
        *done, left_rest = (left + text).split("\n")
        right += exit_text
        for line in done:
            rows.append((line, right))
            right = ""
        left = left_rest
    rows.append((left, right))
    print(f"{left_title:<{width}} | {right_title}")
    print("-" * width + "-+-" + "-" * width)
    for l, r in rows:
        print(f"{l.expandtabs()[:width]:<{width}} | {r.expandtabs()[:width]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--exit-layer", type=int, default=None,
                        help="Also attach final LayerNorm + LM head after this many layers (random if omitted)")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None,
                        help="Defaults to the checkpoint dtype (float16 for phi-2)")
    args = parser.parse_args()

    device = torch.accelerator.current_accelerator()
    model = PhiForCausalLM.from_pretrained("microsoft/phi-2", dtype=getattr(torch, args.dtype) if args.dtype else None,
                                           device=device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", clean_up_tokenization_spaces=False)
    num_layers = len(model.model.layers)
    exit_layer = args.exit_layer or random.randint(1, num_layers - 1)

    prompt = '''def print_prime(n):
   """
   Print all primes between 1 and n
   """'''
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    start = time.perf_counter()
    new_tokens, exit_tokens = generate(model, input_ids, args.max_new_tokens, exit_layer)
    elapsed = time.perf_counter() - start
    print(f"{len(new_tokens)} tokens in {elapsed:.2f}s ({len(new_tokens) / elapsed:.1f} tok/s) on {device}")
    agree = sum(a == b for a, b in zip(new_tokens, exit_tokens))
    print(f"Layer {exit_layer} agrees with layer {num_layers} on {agree}/{len(new_tokens)} tokens "
          f"({100 * agree / len(new_tokens):.0f}%)\n")
    side_by_side(tokenizer, new_tokens, exit_tokens, f"All {num_layers} layers",
                 f"Layer {exit_layer} exit (same context)")


if __name__ == "__main__":
    main()

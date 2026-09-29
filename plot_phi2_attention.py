"""Plot head-averaged attention heatmaps for Phi-2 layers over prompt + generated tokens.

Also checks that incremental (KV-cached) decoding produces exactly the same rows as a single
full-sequence pass, i.e. that earlier rows of the attention matrix never change.
"""
import argparse
import math

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from transformers import AutoTokenizer

from run_phi2 import DEFAULT_PROMPT, PhiForCausalLM, generate

# Single-hue sequential ramp (blue 100 -> 700), light = near zero
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def token_label(tokenizer, tok):
    return tokenizer.decode([tok]).replace("\n", "⏎").replace(" ", "·") or "∅"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 8, 16, 32], help="1-based layer indices")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--output", default="phi2_attention.png")
    # Last-layer attention logits reach ~1e5, so float16 rounding flips near-tied keys between
    # the incremental and full-pass kernels; float32 keeps the rows bit-for-bit comparable
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    args = parser.parse_args()

    device = torch.accelerator.current_accelerator()
    model = PhiForCausalLM.from_pretrained("microsoft/phi-2", dtype=getattr(torch, args.dtype), device=device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", clean_up_tokenization_spaces=False)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    attns = [model.model.layers[i - 1].self_attn for i in args.layers]

    # Incremental decode: prefill yields the prompt rows, each decode step yields one new row
    for attn in attns:
        attn.recorded_probs = []
    new_tokens, _ = generate(model, input_ids, args.max_new_tokens, exit_layer=None)
    tokens = input_ids[0].tolist() + new_tokens
    seq_len = len(tokens)
    incremental = []
    for attn in attns:
        mat = torch.zeros(attn.n_heads, seq_len, seq_len, device=device)
        row = 0
        for probs in attn.recorded_probs:
            mat[:, row:row + probs.shape[1], :probs.shape[2]] = probs
            row += probs.shape[1]
        incremental.append(mat)

    # Single full-sequence pass over the same tokens
    for attn in attns:
        attn.recorded_probs = []
    with torch.inference_mode():
        model(torch.tensor([tokens], device=device))
    full = [attn.recorded_probs[0] for attn in attns]
    for attn in attns:
        attn.recorded_probs = None

    for layer, inc, ful in zip(args.layers, incremental, full):
        print(f"layer {layer}: max |incremental - full pass| = {(inc - ful).abs().max().item():.2e}")

    labels = [token_label(tokenizer, t) for t in tokens]
    n_prompt = input_ids.shape[1]
    ncols = min(len(args.layers), 2)
    nrows = math.ceil(len(args.layers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 9.5 * nrows), squeeze=False)
    for ax in axes.flat[len(args.layers):]:
        ax.set_visible(False)
    norm = LogNorm(vmin=1e-3, vmax=1.0)
    for ax, layer, mat in zip(axes.flat, args.layers, full):
        avg = mat.mean(dim=0).cpu()
        avg = avg.masked_fill(torch.ones_like(avg, dtype=torch.bool).triu(1), float("nan"))
        im = ax.imshow(avg.clamp(min=1e-3), cmap=BLUES, norm=norm, interpolation="nearest")
        ax.set_title(f"Layer {layer} (mean over {mat.shape[0]} heads)", fontsize=12, loc="left")
        ax.set_xticks(range(seq_len), labels, rotation=90, fontsize=5, fontfamily="monospace")
        ax.set_yticks(range(seq_len), labels, fontsize=5, fontfamily="monospace")
        ax.set_xlabel("key (attended-to token)", color="#555")
        ax.set_ylabel("query (current token)", color="#555")
        # Boundary between prompt and generated tokens
        for pos in (ax.axhline, ax.axvline):
            pos(n_prompt - 0.5, color="#888", linewidth=0.6, linestyle="--")
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.4, pad=0.02)
    cbar.set_label("attention probability (log scale)", color="#555")
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.output} ({seq_len} tokens, {n_prompt} prompt)")


if __name__ == "__main__":
    main()

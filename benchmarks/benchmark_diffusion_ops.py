"""Per-operator forward/backward benchmark for the ops used in Stable Diffusion v1.5.

Each operator is exercised in isolation at the same shapes the real pipeline
uses for 512x512 inference (CFG batch of 2 = cond + uncond), with forward and
backward measured separately. Backward is timed on a freshly-built autograd
graph each iteration so we measure the backward kernel, not graph-reuse
artifacts.

Coverage:
  * UNet (sd-legacy/stable-diffusion-v1-5):
      - timestep sinusoidal embedding + time MLP
      - Conv2d / GroupNorm / SiLU at multiple resolutions
      - ResNet block (full, with time-embedding injection)
      - Self-attention and cross-attention (via fused SDPA) at 64^2/32^2/16^2
      - Transformer-block FFN with GeGLU
      - Down/Up sample blocks
  * VAE decoder (high-res ops dominate decode cost):
      - ResNet block at 64x64x512 (mid)
      - ResNet block at 512x512x128 (top)
      - Upsample (nearest + Conv3x3)
      - Final conv 128 -> 3

Stable Diffusion is typically inference-only (torch.no_grad), but backward is
still timed for parity with the transformer benchmark and to characterize
LoRA/Dreambooth fine-tuning costs.
"""

import math
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Config (matches Stable Diffusion v1.5 inference at 512x512) ----
BATCH = 2                # CFG: cond + uncond
LATENT_CH = 4
LATENT_HW = 64           # 512 / 8
BASE_CH = 320            # UNet block_out_channels[0]
TEXT_LEN = 77            # CLIP tokens
CROSS_DIM = 768          # CLIP hidden size
NUM_HEADS = 8            # SD v1.5 attention heads (constant across resolutions)
TIME_EMB_DIM = 1280      # 4 * BASE_CH
GN_GROUPS = 32
DROPOUT = 0.0            # SD inference uses 0; transformer-block dropout is 0 too

VAE_LATENT_CH = 4
VAE_OUT_CH = 3
VAE_BLOCK_OUT = (128, 256, 512, 512)  # decoder: rev = 512,512,256,128

WARMUP = 5
BENCH = 25


# ---------- device / timing helpers ----------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _stats(times):
    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg, std


def time_fwd(fn: Callable, device) -> tuple[float, float]:
    for _ in range(WARMUP):
        fn()
    sync(device)
    times = []
    for _ in range(BENCH):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)
    return _stats(times)


def time_bwd(make_graph: Callable, device) -> tuple[float, float]:
    """make_graph() returns (out, grad_out); only `out.backward(grad_out)` is timed."""
    for _ in range(WARMUP):
        out, grad_out = make_graph()
        out.backward(grad_out)
    sync(device)
    times = []
    for _ in range(BENCH):
        out, grad_out = make_graph()
        sync(device)
        t0 = time.perf_counter()
        out.backward(grad_out)
        sync(device)
        times.append(time.perf_counter() - t0)
    return _stats(times)


def _shape(t: torch.Tensor) -> str:
    return "x".join(str(d) for d in t.shape)


def _dtype(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


def _desc(t: torch.Tensor) -> str:
    return f"{_shape(t)} {_dtype(t)}"


def _zero_grads(*tensors):
    for t in tensors:
        if t.grad is not None:
            t.grad = None


def _zero_module_grads(*mods):
    for m in mods:
        m.zero_grad(set_to_none=True)


def _print_header():
    print(
        f"  {'op':38s}  {'fwd (ms)':>17s}  {'bwd (ms)':>17s}  "
        f"{'input':<32s}  {'output':<24s}"
    )
    print("  " + "-" * 134)


def _print_row(name, fwd, bwd, in_desc, out_desc):
    fa, fs = fwd
    f_str = f"{fa*1000:8.3f} ± {fs*1000:6.3f}"
    if bwd is not None:
        ba, bs = bwd
        b_str = f"{ba*1000:8.3f} ± {bs*1000:6.3f}"
    else:
        b_str = f"{'-':>17s}"
    print(f"  {name:38s}  {f_str}  {b_str}  {in_desc:<32s}  {out_desc:<24s}")


# ---------- minimal building blocks (functionally close to diffusers) ----------
class ResnetBlock(nn.Module):
    """SD-style ResNet block: GN -> SiLU -> Conv -> (+time_emb) -> GN -> SiLU -> Drop -> Conv -> +skip."""
    def __init__(self, in_ch, out_ch, time_emb_dim=TIME_EMB_DIM, groups=GN_GROUPS):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(DROPOUT)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_emb_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class GeGLU(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.proj = nn.Linear(dim, dim * mult * 2)
        self.out = nn.Linear(dim * mult, dim)

    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)
        return self.out(a * F.gelu(b))


# ---------- timestep sinusoidal embedding ----------
def timestep_embedding(t: torch.Tensor, dim: int, out_dtype=torch.float32) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(out_dtype)


# ---------- benchmark body ----------
def run_benchmarks(device, dtype):
    print(f"\n========== dtype={str(dtype).replace('torch.', '')} ==========")
    _print_header()

    B = BATCH
    H = LATENT_HW                # 64
    C0 = BASE_CH                 # 320
    C1 = BASE_CH * 2             # 640
    C2 = BASE_CH * 4             # 1280
    HD = NUM_HEADS               # 8

    # ============================================================
    # ===========          UNet ops                    ===========
    # ============================================================

    # ---------- timestep sinusoidal embedding ----------
    t_idx = torch.randint(0, 1000, (B,), device=device)
    fwd = time_fwd(lambda: timestep_embedding(t_idx, C0, dtype), device)
    out = timestep_embedding(t_idx, C0, dtype)
    _print_row("timestep_embedding (sin/cos)", fwd, None, f"({B},) int64", _desc(out))

    # ---------- Time MLP: Linear(320,1280) -> SiLU -> Linear(1280,1280) ----------
    time_mlp = nn.Sequential(
        nn.Linear(C0, TIME_EMB_DIM),
        nn.SiLU(),
        nn.Linear(TIME_EMB_DIM, TIME_EMB_DIM),
    ).to(device=device, dtype=dtype)
    t_in = torch.randn(B, C0, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(B, TIME_EMB_DIM, device=device, dtype=dtype)
    fwd = time_fwd(lambda: time_mlp(t_in), device)
    def mg():
        _zero_grads(t_in); _zero_module_grads(time_mlp)
        return time_mlp(t_in), g
    bwd = time_bwd(mg, device)
    _print_row("time_mlp (320->1280->1280)", fwd, bwd, _desc(t_in), f"{B}x{TIME_EMB_DIM} {_dtype(g)}")

    # ---------- conv_in: 4 -> 320 @ 64x64 ----------
    conv_in = nn.Conv2d(LATENT_CH, C0, 3, padding=1).to(device=device, dtype=dtype)
    x = torch.randn(B, LATENT_CH, H, H, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(B, C0, H, H, device=device, dtype=dtype)
    fwd = time_fwd(lambda: conv_in(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(conv_in)
        return conv_in(x), g
    bwd = time_bwd(mg, device)
    _print_row("conv_in (4->320, 3x3, 64^2)", fwd, bwd, _desc(x), _desc(g))

    # ---------- Conv2d 3x3 320->320 @ 64x64 (resnet conv body) ----------
    conv33 = nn.Conv2d(C0, C0, 3, padding=1).to(device=device, dtype=dtype)
    x = torch.randn(B, C0, H, H, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn_like(x)
    fwd = time_fwd(lambda: conv33(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(conv33)
        return conv33(x), g
    bwd = time_bwd(mg, device)
    _print_row("Conv2d 3x3 320->320 @ 64^2", fwd, bwd, _desc(x), _desc(g))

    # ---------- GroupNorm(32, 320) @ 64x64 ----------
    gn = nn.GroupNorm(GN_GROUPS, C0).to(device=device, dtype=dtype)
    x = torch.randn(B, C0, H, H, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn_like(x)
    fwd = time_fwd(lambda: gn(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(gn)
        return gn(x), g
    bwd = time_bwd(mg, device)
    _print_row("GroupNorm(32,320) @ 64^2", fwd, bwd, _desc(x), _desc(g))

    # ---------- SiLU @ 64x64x320 ----------
    x = torch.randn(B, C0, H, H, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn_like(x)
    fwd = time_fwd(lambda: F.silu(x), device)
    def mg():
        _zero_grads(x)
        return F.silu(x), g
    bwd = time_bwd(mg, device)
    _print_row("SiLU @ 64^2x320", fwd, bwd, _desc(x), _desc(g))

    # ---------- ResNet block @ 64^2x320 (full, with time emb) ----------
    rn0 = ResnetBlock(C0, C0).to(device=device, dtype=dtype)
    x = torch.randn(B, C0, H, H, device=device, dtype=dtype, requires_grad=True)
    t_emb = torch.randn(B, TIME_EMB_DIM, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn_like(x)
    fwd = time_fwd(lambda: rn0(x, t_emb), device)
    def mg():
        _zero_grads(x, t_emb); _zero_module_grads(rn0)
        return rn0(x, t_emb), g
    bwd = time_bwd(mg, device)
    _print_row("ResnetBlock 320@64^2 (+t_emb)", fwd, bwd, _desc(x), _desc(g))

    # ---------- Self-attention via SDPA at three UNet resolutions ----------
    for label, ch, hw in [("64^2 320", C0, 64), ("32^2 640", C1, 32), ("16^2 1280", C2, 16)]:
        T = hw * hw
        head_dim = ch // HD
        # x is the spatial-flattened sequence (B, T, ch); reshape into heads for SDPA
        q = torch.randn(B, HD, T, head_dim, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        g_o = torch.randn_like(q)
        fwd = time_fwd(
            lambda: F.scaled_dot_product_attention(q, k, v), device
        )
        def mg():
            _zero_grads(q, k, v)
            return F.scaled_dot_product_attention(q, k, v), g_o
        bwd = time_bwd(mg, device)
        _print_row(
            f"SDPA self-attn {label}", fwd, bwd,
            f"{B}x{HD}x{T}x{head_dim}", _desc(g_o),
        )

    # ---------- Cross-attention via SDPA at three UNet resolutions ----------
    # Q from spatial latents (B, HD, T, head_dim), K/V from text (B, HD, 77, head_dim).
    for label, ch, hw in [("64^2 320", C0, 64), ("32^2 640", C1, 32), ("16^2 1280", C2, 16)]:
        T = hw * hw
        head_dim = ch // HD
        q = torch.randn(B, HD, T, head_dim, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, HD, TEXT_LEN, head_dim, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, HD, TEXT_LEN, head_dim, device=device, dtype=dtype, requires_grad=True)
        g_o = torch.randn_like(q)
        fwd = time_fwd(
            lambda: F.scaled_dot_product_attention(q, k, v), device
        )
        def mg():
            _zero_grads(q, k, v)
            return F.scaled_dot_product_attention(q, k, v), g_o
        bwd = time_bwd(mg, device)
        _print_row(
            f"SDPA cross-attn {label}", fwd, bwd,
            f"q={B}x{HD}x{T}x{head_dim} kv={B}x{HD}x{TEXT_LEN}x{head_dim}",
            _desc(g_o),
        )

    # ---------- KV projection from CLIP hidden (768) -> ch, used by cross-attn ----------
    kv_proj = nn.Linear(CROSS_DIM, 2 * C0, bias=False).to(device=device, dtype=dtype)
    ctx = torch.randn(B, TEXT_LEN, CROSS_DIM, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(B, TEXT_LEN, 2 * C0, device=device, dtype=dtype)
    fwd = time_fwd(lambda: kv_proj(ctx), device)
    def mg():
        _zero_grads(ctx); _zero_module_grads(kv_proj)
        return kv_proj(ctx), g
    bwd = time_bwd(mg, device)
    _print_row("Linear ctx->kv (768->2*320)", fwd, bwd, _desc(ctx), _desc(g))

    # ---------- LayerNorm @ (B, 64^2, 320) (transformer-block norm) ----------
    ln = nn.LayerNorm(C0).to(device=device, dtype=dtype)
    x = torch.randn(B, H * H, C0, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn_like(x)
    fwd = time_fwd(lambda: ln(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(ln)
        return ln(x), g
    bwd = time_bwd(mg, device)
    _print_row("LayerNorm 320 @ (B,4096,320)", fwd, bwd, _desc(x), _desc(g))

    # ---------- FFN GeGLU (320 -> 320*4 -> 320) at 64^2 tokens ----------
    ffn = GeGLU(C0, mult=4).to(device=device, dtype=dtype)
    x = torch.randn(B, H * H, C0, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn_like(x)
    fwd = time_fwd(lambda: ffn(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(ffn)
        return ffn(x), g
    bwd = time_bwd(mg, device)
    _print_row("FFN GeGLU 320 @ (B,4096,320)", fwd, bwd, _desc(x), _desc(g))

    # ---------- Downsample: Conv 3x3 stride=2 (320->320, 64^2 -> 32^2) ----------
    down = nn.Conv2d(C0, C0, 3, stride=2, padding=1).to(device=device, dtype=dtype)
    x = torch.randn(B, C0, H, H, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(B, C0, H // 2, H // 2, device=device, dtype=dtype)
    fwd = time_fwd(lambda: down(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(down)
        return down(x), g
    bwd = time_bwd(mg, device)
    _print_row("Downsample Conv3x3 s=2 320 64->32", fwd, bwd, _desc(x), _desc(g))

    # ---------- Upsample: nearest x2 + Conv3x3 (320 @ 32^2 -> 64^2) ----------
    up_conv = nn.Conv2d(C0, C0, 3, padding=1).to(device=device, dtype=dtype)
    x = torch.randn(B, C0, H // 2, H // 2, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(B, C0, H, H, device=device, dtype=dtype)
    def fwd_up():
        return up_conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))
    fwd = time_fwd(fwd_up, device)
    def mg():
        _zero_grads(x); _zero_module_grads(up_conv)
        return fwd_up(), g
    bwd = time_bwd(mg, device)
    _print_row("Upsample nearest+Conv3x3 320 32->64", fwd, bwd, _desc(x), _desc(g))

    # ---------- ResNet at deeper resolutions ----------
    for label, ch, hw in [("640@32^2", C1, 32), ("1280@16^2", C2, 16)]:
        rn = ResnetBlock(ch, ch).to(device=device, dtype=dtype)
        x = torch.randn(B, ch, hw, hw, device=device, dtype=dtype, requires_grad=True)
        t_emb = torch.randn(B, TIME_EMB_DIM, device=device, dtype=dtype, requires_grad=True)
        g = torch.randn_like(x)
        fwd = time_fwd(lambda: rn(x, t_emb), device)
        def mg(rn=rn, x=x, t_emb=t_emb, g=g):
            _zero_grads(x, t_emb); _zero_module_grads(rn)
            return rn(x, t_emb), g
        bwd = time_bwd(mg, device)
        _print_row(f"ResnetBlock {label} (+t_emb)", fwd, bwd, _desc(x), _desc(g))

    # ---------- conv_out: 320 -> 4 @ 64^2 ----------
    conv_out = nn.Conv2d(C0, LATENT_CH, 3, padding=1).to(device=device, dtype=dtype)
    x = torch.randn(B, C0, H, H, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(B, LATENT_CH, H, H, device=device, dtype=dtype)
    fwd = time_fwd(lambda: conv_out(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(conv_out)
        return conv_out(x), g
    bwd = time_bwd(mg, device)
    _print_row("conv_out (320->4, 3x3, 64^2)", fwd, bwd, _desc(x), _desc(g))

    # ============================================================
    # ===========          VAE decoder ops             ===========
    # (decoder block_out_channels reversed: 512, 512, 256, 128)
    # spatial: 64 -> 128 -> 256 -> 512
    # ============================================================

    # ---------- VAE post_quant_conv (1x1, 4->4) at 64^2 ----------
    pqc = nn.Conv2d(VAE_LATENT_CH, VAE_LATENT_CH, 1).to(device=device, dtype=dtype)
    x = torch.randn(B, VAE_LATENT_CH, 64, 64, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn_like(x)
    fwd = time_fwd(lambda: pqc(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(pqc)
        return pqc(x), g
    bwd = time_bwd(mg, device)
    _print_row("VAE post_quant_conv 1x1 4->4 64^2", fwd, bwd, _desc(x), _desc(g))

    # VAE ResNet block has no time embedding; reuse ResnetBlock with a dummy zero t_emb.
    class VaeResnetBlock(nn.Module):
        def __init__(self, in_ch, out_ch, groups=GN_GROUPS):
            super().__init__()
            self.norm1 = nn.GroupNorm(groups, in_ch)
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.norm2 = nn.GroupNorm(groups, out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
            self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x):
            h = self.conv1(F.silu(self.norm1(x)))
            h = self.conv2(F.silu(self.norm2(h)))
            return h + self.skip(x)

    # ---------- VAE ResNet @ 64^2x512 (mid block) ----------
    rn_v0 = VaeResnetBlock(512, 512).to(device=device, dtype=dtype)
    x = torch.randn(B, 512, 64, 64, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn_like(x)
    fwd = time_fwd(lambda: rn_v0(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(rn_v0)
        return rn_v0(x), g
    bwd = time_bwd(mg, device)
    _print_row("VAE ResnetBlock 512@64^2", fwd, bwd, _desc(x), _desc(g))

    # ---------- VAE ResNet @ 256^2x256 (mid up block) ----------
    rn_v1 = VaeResnetBlock(256, 256).to(device=device, dtype=dtype)
    x = torch.randn(B, 256, 256, 256, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn_like(x)
    fwd = time_fwd(lambda: rn_v1(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(rn_v1)
        return rn_v1(x), g
    bwd = time_bwd(mg, device)
    _print_row("VAE ResnetBlock 256@256^2", fwd, bwd, _desc(x), _desc(g))

    # ---------- VAE ResNet @ 512^2x128 (top up block, dominates wall-clock) ----------
    rn_v2 = VaeResnetBlock(128, 128).to(device=device, dtype=dtype)
    x = torch.randn(B, 128, 512, 512, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn_like(x)
    fwd = time_fwd(lambda: rn_v2(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(rn_v2)
        return rn_v2(x), g
    bwd = time_bwd(mg, device)
    _print_row("VAE ResnetBlock 128@512^2", fwd, bwd, _desc(x), _desc(g))

    # ---------- VAE Upsample (nearest x2 + Conv3x3) 256@256^2 -> 256@512^2 ----------
    up_v = nn.Conv2d(256, 256, 3, padding=1).to(device=device, dtype=dtype)
    x = torch.randn(B, 256, 256, 256, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(B, 256, 512, 512, device=device, dtype=dtype)
    def fwd_upv():
        return up_v(F.interpolate(x, scale_factor=2.0, mode="nearest"))
    fwd = time_fwd(fwd_upv, device)
    def mg():
        _zero_grads(x); _zero_module_grads(up_v)
        return fwd_upv(), g
    bwd = time_bwd(mg, device)
    _print_row("VAE Upsample 256@256->512", fwd, bwd, _desc(x), _desc(g))

    # ---------- VAE final conv_out: 128 -> 3 @ 512^2 ----------
    conv_out_v = nn.Conv2d(128, VAE_OUT_CH, 3, padding=1).to(device=device, dtype=dtype)
    x = torch.randn(B, 128, 512, 512, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(B, VAE_OUT_CH, 512, 512, device=device, dtype=dtype)
    fwd = time_fwd(lambda: conv_out_v(x), device)
    def mg():
        _zero_grads(x); _zero_module_grads(conv_out_v)
        return conv_out_v(x), g
    bwd = time_bwd(mg, device)
    _print_row("VAE conv_out (128->3, 3x3, 512^2)", fwd, bwd, _desc(x), _desc(g))


def main():
    device = get_device()
    print(f"Device: {device}  |  torch {torch.__version__}")
    print(
        f"BATCH={BATCH}  latent={LATENT_CH}x{LATENT_HW}^2  base_ch={BASE_CH}  "
        f"heads={NUM_HEADS}  text_len={TEXT_LEN}  cross_dim={CROSS_DIM}  "
        f"time_emb={TIME_EMB_DIM}"
    )
    print(f"Warmup={WARMUP}  Bench iters={BENCH}")

    # SD inference is typically fp16 on MPS/CUDA. Keep both for comparison.
    run_benchmarks(device, torch.float32)
    if device.type in ("cuda", "mps"):
        run_benchmarks(device, torch.float16)


if __name__ == "__main__":
    main()

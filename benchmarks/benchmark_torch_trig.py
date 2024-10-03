# Benchmark torch.sin + torch.cos performance on varios platforms/dtypes
# Against torch-2.5.0 for 4096x4096

from timeit import default_timer

import torch
import torch.utils.cpp_extension
from torch.utils.benchmark import Measurement, Timer

def bench_unary(
    m,
    n,
    unary_func,
    dtype=torch.float32,
    device: str = "cpu",
) -> Measurement:
    if device == "mps":
        sync_cmd = "torch.mps.synchronize()"
    elif device == "cuda":
        sync_cmd = "torch.cuda.synchronize()"
    else:
        sync_cmd = ""
    t = Timer(
        stmt=f"f(x);{sync_cmd}",
        setup=f"x=torch.rand(({m}, {n}), dtype={dtype}, device='{device}')",
        globals = {'f': unary_func},
        language="python", timer=default_timer
    )
    return t.blocked_autorange()


if __name__ == "__main__":
    def f(x):
        return torch.sin(x) + torch.cos(x)

    f_c=torch.compile(f)

    torch.set_num_threads(1)
    m, n = 8192, 16384
    device = "cpu"
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        eager_t = bench_unary(m, n, f, dtype, device=device)
        comp_t = bench_unary(m, n, f_c, dtype, device=device)
        use_msec = eager_t.mean > 1e-4 or comp_t.mean > 1e-4
        multiplier = 1e3 if use_msec else 1e6
        uname = "msec" if use_msec else "usec"
        print(f"torch.sin+torch.cos({device}) {str(dtype):>14} {eager_t.mean*multiplier:>7.2f} {uname} {comp_t.mean*multiplier:>7.2f} {uname} {eager_t.mean/comp_t.mean:>7.2f}")

    if torch.cuda.is_available():
        device = "cuda"
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            eager_t = bench_unary(m, n, f, dtype, device=device)
            comp_t = bench_unary(m, n, f_c, dtype, device=device)
            use_msec = eager_t.mean > 1e-4 or comp_t.mean > 1e-4
            multiplier = 1e3 if use_msec else 1e6
            uname = "msec" if use_msec else "usec"
            print(f"torch.sin+torch.cos({device}) {str(dtype):>14} {eager_t.mean*multiplier:>7.2f} {uname} {comp_t.mean*multiplier:>7.2f} {uname} {eager_t.mean/comp_t.mean:>7.2f}")
    if torch.backends.mps.is_available():
        ext = torch.utils.cpp_extension.load(name="mps_ext", sources=["mps_sum_sincos.mm"])
        device = "mps"
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            eager_t = bench_unary(m, n, f, dtype, device=device)
            comp_t = bench_unary(m, n, ext.mps_sum_sincos, dtype, device=device)
            use_msec = eager_t.mean > 1e-4
            multiplier = 1e3 if use_msec else 1e6
            uname = "msec" if use_msec else "usec"
            print(f"torch.sin+torch.cos({device}) {str(dtype):>14} {eager_t.mean*multiplier:>7.2f} {uname} {comp_t.mean*multiplier:>7.2f} {uname} {eager_t.mean/comp_t.mean:>7.2f}")

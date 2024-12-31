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


mps_ext = None
mps_lib = None
def mps_optimized_func():
    global mps_ext, mps_lib
    if not hasattr(torch.mps, "_compile_shader"):
        if mps_ext is None:
            mps_ext = torch.utils.cpp_extension.load(name="mps_ext", sources=["mps_sum_sincos.mm"])
        return mps_ext.mps_sum_sincos

    if mps_lib is None:
        mps_lib = torch.mps._compile_shader("""
#include <metal_stdlib>
using namespace metal;
template<typename T>
kernel void sum_sincos(constant T* x,
                       device   T* out,
                       uint index [[thread_position_in_grid]])
{
    out[index] = static_cast<T>(sin(x[index]) + cos(x[index]));
}

template [[host_name("sum_sincos_float")]] kernel void sum_sincos(constant float*, device float*, uint);
template [[host_name("sum_sincos_half")]] kernel void sum_sincos(constant half*, device half*, uint);
template [[host_name("sum_sincos_bfloat")]] kernel void sum_sincos(constant bfloat*, device bfloat*, uint);
            """)

    def f_s(x):
        rc = torch.empty_like(x)
        if x.dtype == torch.float:
            mps_lib.sum_sincos_float(x, rc)
        elif x.dtype == torch.half:
            mps_lib.sum_sincos_half(x, rc)
        elif x.dtype == torch.bfloat16:
            mps_lib.sum_sincos_bfloat(x, rc)
        return rc
    return f_s


def run_bench_for_device(m, n, device, func, func_compiled):
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        # Validate correctness first
        inp = torch.rand(m, n, dtype=dtype, device=device)
        out = func(inp)
        out_compiled = func_compiled(inp)
        atol = 1e-2 if dtype is torch.bfloat16 else 2e-3 if dtype is torch.float16 else 1e-8
        if not torch.allclose(out, out_compiled, atol = atol):
            raise RuntimeError(f"out-out_compiled.abs().max() is {(out-out_compiled).abs().max().item()} for {dtype} and {device}")
        eager_t = bench_unary(m, n, func, dtype, device=device)
        comp_t = bench_unary(m, n, func_compiled, dtype, device=device)
        use_msec = eager_t.mean > 1e-4 or comp_t.mean > 1e-4
        multiplier = 1e3 if use_msec else 1e6
        uname = "msec" if use_msec else "usec"
        print(f"torch.sin+torch.cos({device}) {str(dtype):>14} {eager_t.mean*multiplier:>7.2f} {uname} {comp_t.mean*multiplier:>7.2f} {uname} {eager_t.mean/comp_t.mean:>7.2f}")

if __name__ == "__main__":
    def f(x):
        return torch.sin(x) + torch.cos(x)

    f_c=torch.compile(f)

    torch.set_num_threads(1)
    m, n = 8192, 16384
    run_bench_for_device(m, n, "cpu", f, f_c)

    if torch.cuda.is_available():
        run_bench_for_device(m, n, "cuda", f, f_c)

    if torch.backends.mps.is_available():
        run_bench_for_device(m, n, "mps", f, mps_optimized_func())

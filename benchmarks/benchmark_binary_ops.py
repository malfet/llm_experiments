# Benchmark elementwise binary ops
# Against torch-2.5.0 for 4096x4096

from timeit import default_timer

import torch
import torch.utils.cpp_extension
from torch.utils.benchmark import Measurement, Timer

def bench_binary(
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
        stmt=f"f(x, y);{sync_cmd}",
        setup=f"x, y=torch.rand((2, {n}), dtype={dtype}, device='{device}').unbind(0)",
        globals = {'f': unary_func},
        language="python", timer=default_timer
    )
    return t.blocked_autorange()


mps_lib = None
def mps_optimized_func():
    global mps_lib
    if mps_lib is None:
        mps_lib = torch.mps._compile_shader("""
#include <metal_stdlib>
using namespace metal;
template<typename T>
kernel void add(constant T* x,
                constant T* y,
                device   T* out,
                uint index [[thread_position_in_grid]])
{
    out[index] = static_cast<T>(x[index] + y[index]);
}

template [[host_name("add_float")]] kernel void add(constant float*, constant float*, device float*, uint);
template [[host_name("add_half")]] kernel void add(constant half*, constant half*, device half*, uint);
template [[host_name("add_bfloat")]] kernel void add(constant bfloat*, constant bfloat*, device bfloat*, uint);
template [[host_name("add_half4")]] kernel void add(constant half4*, constant half4*, device half4*, uint);
            """)

    def f_s(x, y):
        rc = torch.empty_like(x)
        if x.dtype == torch.float:
            mps_lib.add_float(x, y, rc)
        elif x.dtype == torch.half:
            mps_lib.add_half4(x, y, rc, threads=x.numel()>>2)
        elif x.dtype == torch.bfloat16:
            mps_lib.add_bfloat(x, y, rc)
        return rc
    return f_s


def run_bench_for_device(n, device, func, func_compiled):
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        # Validate correctness first
        inp = torch.rand(2, n, dtype=dtype, device=device).unbind(0)
        out = func(*inp)
        out_compiled = func_compiled(*inp)
        if not torch.allclose(out, out_compiled):
            raise RuntimeError(f"out-out_compiled.abs().max() is {(out-out_compiled).abs().max().item()} for {dtype} and {device}")
        eager_t = bench_binary(n, func, dtype, device=device)
        comp_t = bench_binary(n, func_compiled, dtype, device=device)
        use_msec = eager_t.mean > 1e-4 or comp_t.mean > 1e-4
        multiplier = 1e3 if use_msec else 1e6
        uname = "msec" if use_msec else "usec"
        print(f"torch.add({device}) {str(dtype):>14} {eager_t.mean*multiplier:>7.2f} {uname} {comp_t.mean*multiplier:>7.2f} {uname} {eager_t.mean/comp_t.mean:>7.2f}")

if __name__ == "__main__":
    def f(x, y):
        return x + y

    f_c=torch.compile(f)

    torch.set_num_threads(1)
    n = 1024**2
    run_bench_for_device(n, "cpu", f, f_c)

    if torch.cuda.is_available():
        run_bench_for_device(n, "cuda", f, f_c)

    if torch.backends.mps.is_available():
        run_bench_for_device(n, "mps", f, mps_optimized_func())

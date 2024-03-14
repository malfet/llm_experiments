# Benchmark torch.mm performance on varios platforms/dtypes
# Against torch-2.2.0 for 256x288 and 288x768 matrices
# |                  | float32 | float16 | bfloat16 |
# | Apple M1         |  85 us  |   83ms  |   24 ms  |
# | Apple M2 Pro     |  103 us |   85ms  |   28 ms  |
# | AWS Tg4          |  484 us |  187ms  |   93 ms  |
# | Xeon 8275CL@3Ghz |  631 us |   64ms  |  3.27 ms |
# | Xeon @2.2Ghz     | 1.67 ms |   93ms  | 73.52 ms |

from timeit import default_timer

import torch
from torch.utils.benchmark import Measurement, Timer


def bench_mm(
    m,
    n,
    k,
    dtype=torch.float32,
    device: str = "cpu",
    trans_a: bool = False,
    trans_b: bool = False,
) -> Measurement:
    setup = f"""
     x = torch.rand({m}, {n}, dtype={dtype}, device="{device}") if not {trans_a} else torch.rand({n}, {m}, dtype={dtype}, device="{device}").t()
     y = torch.rand({n}, {k}, dtype={dtype}, device="{device}") if not {trans_b} else torch.rand({k}, {n}, dtype={dtype}, device="{device}").t()
    """

    t = Timer(
        stmt="torch.mm(x, y)", setup=setup, language="python", timer=default_timer
    )
    return t.blocked_autorange()


def bench_mv(
    m,
    n,
    dtype=torch.float32,
    device: str = "cpu",
    trans_a: bool = False,
) -> Measurement:
    setup = f"""
     x = torch.rand({m}, {n}, dtype={dtype}, device="{device}") if not {trans_a} else torch.rand({n}, {m}, dtype={dtype}, device="{device}").t()
     y = torch.rand({n}, dtype={dtype}, device="{device}")
    """

    t = Timer(
        stmt="torch.mv(x, y)", setup=setup, language="python", timer=default_timer
    )
    return t.blocked_autorange()


def plot_mv_perf(dtype=torch.float32):
    import matplotlib.pyplot as plt
    torch.set_num_threads(1)

    sizes = [i for i in range(100, 1001, 100)]
    times_nt = [bench_mv(n, n, dtype=dtype, trans_a=False).mean * 1e6 for n in sizes]
    times_t = [bench_mv(n, n, dtype=dtype, trans_a=True).mean * 1e6 for n in sizes]
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_nt, '-o', label=f"{dtype} normal")
    plt.plot(sizes, times_t, '-o', label=f"{dtype} transposed")
    plt.title('Benchmarking Matrix-Vector Multiplication')
    plt.xlabel('Matrix Size (n x n)')
    plt.ylabel('Time Taken (microeconds)')
    plt.legend()
    plt.grid(True)
    plt.show()


def benchmark_linalg(m: int = 1, n: int = 256, k: int = 768, device="cpu") -> None:
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        rc = bench_mv(n, k, dtype, device=device, trans_a=False)
        print(f"mv_nt   {str(dtype):>14} {rc.mean*1e6:>7.2f} usec")

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        rc = bench_mv(n, k, dtype, device=device, trans_a=True)
        print(f"mv_ta   {str(dtype):>14} {rc.mean*1e6:>7.2f} usec")

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        rc = bench_mm(m, n, k, dtype, device=device)
        print(f"notrans {str(dtype):>14} {rc.mean*1e6:>7.2f} usec")

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        rc = bench_mm(m, n, k, dtype, trans_a=True, device=device)
        print(f"trans_a {str(dtype):>14} {rc.mean*1e6:>7.2f} usec")

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        rc = bench_mm(m, n, k, dtype, trans_b=True, device=device)
        print(f"trans_b {str(dtype):>14} {rc.mean*1e6:>7.2f} usec")


if __name__ == "__main__":
    torch.set_num_threads(1)
    benchmark_linalg()
    if hasattr(torch._C, "_set_cpu_allow_fp16_reduced_precision_reduction"):
        prev = torch._C._get_cpu_allow_fp16_reduced_precision_reduction()
        try:
            torch._C._set_cpu_allow_fp16_reduced_precision_reduction(True)
            print("\n\nUsing FP16 accumulation")
            benchmark_linalg()
        finally:
            torch._C._set_cpu_allow_fp16_reduced_precision_reduction(prev)

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


if __name__ == "__main__":
    torch.set_num_threads(1)
    # m, n, k = 256, 288, 768
    m, n, k = 1, 256, 768
    device = "cpu"
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        rc = bench_mv(n, k, dtype, device=device, trans_a=False)
        print("mv_nt", dtype, f"{rc.mean*1e6:.2f} usec")

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        rc = bench_mv(n, k, dtype, device=device, trans_a=True)
        print("mv_ta", dtype, f"{rc.mean*1e6:.2f} usec")

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        rc = bench_mm(m, n, k, dtype, device=device)
        print("notrans", dtype, f"{rc.mean*1e6:.2f} usec")

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        rc = bench_mm(m, n, k, dtype, trans_a=True, device=device)
        print("trans_a", dtype, f"{rc.mean*1e6:.2f} usec")

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        rc = bench_mm(m, n, k, dtype, trans_b=True, device=device)
        print("trans_b", dtype, f"{rc.mean*1e6:.2f} usec")

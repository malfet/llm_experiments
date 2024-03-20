# Benchmark torch.rand performance on varios platforms/dtypes
# Against torch-2.2.0 for 4096x4096
# |                  |  float32 | float16  | bfloat16 |
# | Apple M1         |          |          |          |
# | Apple M2 Pro     |  50 msec | 137 msec | 50 msec  |
# | AWS Tg4          |          |          |          |
# | Xeon 8275CL@3Ghz | 107 msec | 130 msec | 123 msec |
# | Xeon @2.2Ghz     |          |          |          |

from timeit import default_timer

import torch
from torch.utils.benchmark import Measurement, Timer


def bench_rand(
    m,
    n,
    dtype=torch.float32,
    device: str = "cpu",
    trans_a: bool = False,
    trans_b: bool = False,
) -> Measurement:
    t = Timer(
        stmt=f"torch.rand(({m}, {n}), dtype={dtype}, device='{device}')", setup="", language="python", timer=default_timer
    )
    return t.blocked_autorange()


if __name__ == "__main__":
    torch.set_num_threads(1)
    m, n = 4096, 4096
    device = "cpu"
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        rc = bench_rand(m, n, dtype, device=device)
        use_msec = rc.mean > 1e-4
        print(f"rand {str(dtype):>14} {rc.mean*(1e3 if use_msec else 1e6):>7.2f} {'m' if use_msec else 'u'}sec")

# Benchmark softmax performance on varios platforms/dtypes
# Against torch-2.2.0 for 4096x4096
# |                  |  float32 dim=0 | float32 dim=1 | float16 dim=0 | float16 dim=1 | bfloat16 dim=0 | bfloat16 dim=1 |
# | Apple M2 Pro     |    186 msec    |     40 msec   |     65 msec   |    48 msec    |     67 msec    |     48 msec    |

from timeit import default_timer

import torch
from torch.utils.benchmark import Measurement, Timer


def bench_softmax(
    m,
    n,
    dtype=torch.float32,
    dim: int = 0,
    device: str = "cpu",
    trans_a: bool = False,
    trans_b: bool = False,
) -> Measurement:
    setup = f"x=torch.rand(({m}, {n}), dtype={dtype}, device='{device}')"
    t = Timer(stmt=f"x.softmax({dim})", setup=setup, language="python", timer=default_timer
    )
    return t.blocked_autorange()


if __name__ == "__main__":
    torch.set_num_threads(1)
    m, n = 4096, 4096
    device = "cpu"
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        for dim in range(2):
            rc = bench_softmax(m, n, dtype=dtype, dim=dim, device=device)
            use_msec = rc.mean > 1e-4
            print(f"softmax({dim}) {str(dtype):>14} {rc.mean*(1e3 if use_msec else 1e6):>7.2f} {'m' if use_msec else 'u'}sec")

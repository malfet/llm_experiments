# Benchmark torch.mm performance on varios platforms/dtypes
# Against torch-2.2.0
# |                  | float32 | float16 | bfloat16 |
# | Apple M1         |  85 us  |   83ms  |   24 ms  |
# | Apple M2 Pro     |  103 us |   85ms  |   28 ms  |
# | AWS Tg4          |  484 us |  187ms  |   93 ms  |
# | Xeon 8275CL@3Ghz |  631 us |   64ms  |  3.27 ms |
# | Xeon @2.2Ghz     | 1.67 ms |   93ms  | 73.52 ms |

import torch
from timeit import default_timer

def bench_mm(m, n, k, dtype):
    setup = f"""
     x = torch.rand({m}, {n}, dtype={dtype})
     y = torch.rand({n}, {k}, dtype={dtype})
    """

    from torch.utils.benchmark import Timer
    t = Timer(stmt="torch.mm(x, y)", setup=setup, language="python", timer=default_timer)
    print(t.blocked_autorange())

if __name__ == "__main__":
    bench_mm(256, 288, 768, torch.float32)
    bench_mm(256, 288, 768, torch.float16)
    bench_mm(256, 288, 768, torch.bfloat16)

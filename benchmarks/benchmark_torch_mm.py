# Benchmark torch.mm performance on varios platforms/dtypes
# Against torch-2.2.0
# |        | float32 | float16 | bfloat16 |
# | M2 Pro |  103 us |   85ms  |   28 ms  |
# | Tg4    |  484 us |  187ms  |   93 ms  |
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
    bench_mm(256, 288, 768, torch.bfloat16)
    bench_mm(256, 288, 768, torch.float16)

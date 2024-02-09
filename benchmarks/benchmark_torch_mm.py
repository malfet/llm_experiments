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

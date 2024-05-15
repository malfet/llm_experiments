Performance running on varios Apple Silicon devices


| Device | naive_bf16 | vectorized_bf16| group_mat4_bf16 |
|--------|----|----| ---- |
| Apple M1 Pro | 9 GFLOP/s | 35 GFLOP/s |
| Apple M2 Pro | 16.4 GFLOP/s | 64.5 GFLOP/s | 322 GFLOP/s |
| Apple M3 Max | 237 GFLOP/s | 735 GFLOP/s |

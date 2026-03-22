import time
import torch

naive_gemm_src = """// Naive
// One thread per output element
kernel void gemm(constant float *A [[buffer(0)]],
                 constant float *B [[buffer(1)]],
                 device float *outputData [[buffer(2)]],
                 constant uint3 &sizes [[buffer(3)]],
                 uint2 thread_index [[thread_position_in_grid]]) {
  const uint lda = sizes.y;
  const uint ldc = sizes.z;
  const uint m = thread_index.y; // 0..sizes.x-1
  const uint n = thread_index.x; // 0..sizes.z-1
  constant auto *A_ptr = A + m * lda;
  constant auto *B_ptr = B + n * lda;

  float rc = 0.0;
  for (uint k = 0; k < sizes.y; k++) {
    const auto a_val = A_ptr[k];
    const auto b_val = B_ptr[k];
    rc += a_val * b_val;
  }
  outputData[m * ldc + n] = rc;
}
"""

vec4_gemm_src = """// SIMD(vec4)
// One thread per output element
using namespace metal;

kernel void gemm(constant float *A [[buffer(0)]],
                 constant float *B [[buffer(1)]],
                 device float *outputData [[buffer(2)]],
                 constant uint3 &sizes [[buffer(3)]],
                 uint2 thread_index [[thread_position_in_grid]]) {
  const uint lda = sizes.y;
  const uint ldc = sizes.z;
  const uint m = thread_index.y; // 0..sizes.x-1
  const uint n = thread_index.x; // 0..sizes.z-1
  constant auto *A_ptr = reinterpret_cast<constant float4 *>(A + m * lda);
  constant auto *B_ptr = reinterpret_cast<constant float4 *>(B + n * lda);

  float rc = 0.0;
  for (uint k = 0; k < sizes.y / 4; k++) {
    rc += dot(A_ptr[k], B_ptr[k]);
  }
  outputData[m * ldc + n] = rc;
}
"""

mat4_gemm_src = """// SIMD(mat4xvec4)
// One thread per group of 4 output elements, 8x8 blocks
using namespace metal;

kernel void gemm(constant float *A [[buffer(0)]],
                 constant float *B [[buffer(1)]],
                 device float *outputData [[buffer(2)]],
                 constant uint3 &sizes [[buffer(3)]],
                 uint2 thread_index [[thread_position_in_grid]]) {
  const uint lda = sizes.y;
  const uint ldc = sizes.z;
  const uint m = thread_index.y; // 0..sizes.x-1
  const uint n = thread_index.x; // 0..sizes.z/4-1
  constant auto *A_ptr = reinterpret_cast<constant float4 *>(A + m * lda);
  constant auto *B_ptr = reinterpret_cast<constant float4 *>(B + n * 4 * lda);

  float4 rc = 0.0;
  for (uint k = 0; k < sizes.y / 4; k++) {
    float4x4 b_mat;
    for(int j = 0; j < 4; ++j) {
      b_mat[j] = B_ptr[k + j * lda /4];
    }
    rc += transpose(b_mat) * A_ptr[k];
  }
  reinterpret_cast<device float4*>(outputData + m * ldc)[n] = rc;
}
"""


class CompiledShader:
    def __init__(self, source, col_div=1):
        self._lib = torch.mps.compile_shader(source)
        # Extract name from first line comment prefix "// "
        self.name = source.strip().split("\n")[0][3:]
        self.col_div = col_div

    def __getattr__(self, name):
        return getattr(self._lib, name)


def validate_gemm(lib, M, N, K):
    A = torch.randn(M, K, device="mps")
    B = torch.randn(N, K, device="mps")
    C = torch.zeros(M, N, device="mps")
    sizes = torch.tensor([M, K, N], device="mps", dtype=torch.int32)

    if lib.col_div == 1:
        lib.gemm(A, B, C, sizes, threads=(N, M))
    else:
        lib.gemm(A, B, C, sizes, threads=(N // lib.col_div, M), group_size=(8, 8))

    torch.testing.assert_close(A @ B.T, C, atol=1e-3, rtol=1e-3)
    print(f"  {lib.name} dim {M}x{N}x{K} validation passed")


def benchmark_gemm(lib, M, N, K, repeat_cnt=200, batch_size=10):
    A = torch.randn(M, K, device="mps")
    B = torch.randn(N, K, device="mps")
    C = torch.zeros(M, N, device="mps")
    sizes = torch.tensor([M, K, N], device="mps", dtype=torch.int32)

    if lib.col_div == 1:
        threads = (N, M)
    else:
        threads = (N // lib.col_div, M)
        group_size = (8, 8)

    def dispatch():
        if lib.col_div == 1:
            lib.gemm(A, B, C, sizes, threads=threads)
        else:
            lib.gemm(A, B, C, sizes, threads=threads, group_size=group_size)

    # Warmup
    dispatch()
    torch.mps.synchronize()

    # Batch multiple dispatches between syncs to amortize sync overhead
    num_batches = repeat_cnt // batch_size
    start = time.perf_counter()
    for _ in range(num_batches):
        for _ in range(batch_size):
            dispatch()
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    total_iters = num_batches * batch_size
    avg_time = elapsed / total_iters
    gflops = M * N * K * 1e-9 / avg_time
    print(f"  Perf of {lib.name} dim {M}x{N}x{K} is {gflops:.2f} GFLOPs")


def main():
    M, N, K = 32, 4128, 4096
    print(f"Using device {torch.backends.mps.get_name()}")

    shaders = [
        CompiledShader(naive_gemm_src),
        CompiledShader(vec4_gemm_src),
        CompiledShader(mat4_gemm_src, col_div=4),
    ]

    print("\nValidation:")
    for lib in shaders:
        validate_gemm(lib, M, N, K)

    print("\nBenchmark:")
    for lib in shaders:
        benchmark_gemm(lib, M, N, K)


if __name__ == "__main__":
    main()

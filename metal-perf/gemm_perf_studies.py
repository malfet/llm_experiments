import time
import torch

# Kernel function name encodes the layout: gemm_nn, gemm_nt, gemm_tn, gemm_tt
# N = row-major (not transposed), T = column-major (transposed storage)


def make_naive_gemm_src(trans_a, trans_b):
    """Generate a naive GEMM shader for the given layout combination."""
    a_index = "m * K + k" if trans_a == "N" else "k * M + m"
    b_index = "k * N + n" if trans_b == "N" else "n * K + k"
    label = f"{trans_a}{trans_b}".lower()
    return f"""// Naive({label})
kernel void gemm_{label}(constant float *A [[buffer(0)]],
                 constant float *B [[buffer(1)]],
                 device float *C [[buffer(2)]],
                 constant uint3 &sizes [[buffer(3)]],
                 uint2 thread_index [[thread_position_in_grid]]) {{
  const uint M = sizes.x;
  const uint K = sizes.y;
  const uint N = sizes.z;
  const uint m = thread_index.y;
  const uint n = thread_index.x;

  float rc = 0.0;
  for (uint k = 0; k < K; k++) {{
    rc += A[{a_index}] * B[{b_index}];
  }}
  C[m * N + n] = rc;
}}
"""


naive_gemm_src = make_naive_gemm_src("N", "T")


def make_vec4_gemm_src(trans_a, trans_b):
    """Generate a vec4 GEMM shader for the given layout combination.

    If K is the contiguous dimension, read as float4* pointer.
    Otherwise, gather 4 scalar elements into a float4.
    """
    label = f"{trans_a}{trans_b}".lower()

    # A: K contiguous when trans_a == "N" (row-major [M,K])
    if trans_a == "N":
        a_setup = "  constant auto *A_ptr = reinterpret_cast<constant float4 *>(A + m * K);"
        a_vec = "A_ptr[k]"
    else:
        a_setup = ""
        a_vec = "float4(A[(k*4+0)*M+m], A[(k*4+1)*M+m], A[(k*4+2)*M+m], A[(k*4+3)*M+m])"

    # B: K contiguous when trans_b == "T" (row-major [N,K])
    if trans_b == "T":
        b_setup = "  constant auto *B_ptr = reinterpret_cast<constant float4 *>(B + n * K);"
        b_vec = "B_ptr[k]"
    else:
        b_setup = ""
        b_vec = "float4(B[(k*4+0)*N+n], B[(k*4+1)*N+n], B[(k*4+2)*N+n], B[(k*4+3)*N+n])"

    return f"""// SIMD(vec4)({label})
using namespace metal;

kernel void gemm_{label}(constant float *A [[buffer(0)]],
                    constant float *B [[buffer(1)]],
                    device float *C [[buffer(2)]],
                    constant uint3 &sizes [[buffer(3)]],
                    uint2 thread_index [[thread_position_in_grid]]) {{
  const uint M = sizes.x;
  const uint K = sizes.y;
  const uint N = sizes.z;
  const uint m = thread_index.y;
  const uint n = thread_index.x;
{a_setup}
{b_setup}

  float rc = 0.0;
  for (uint k = 0; k < K / 4; k++) {{
    rc += dot({a_vec}, {b_vec});
  }}
  C[m * N + n] = rc;
}}
"""


vec4_gemm_src = make_vec4_gemm_src("N", "T")


def make_mat4_gemm_src(trans_a, trans_b):
    """Generate a mat4xvec4 GEMM shader for the given layout combination.

    Processes 4 output N elements at once. For each K chunk of 4:
    - a_vec: float4 of A[m][k*4..k*4+3] (pointer or gather)
    - b_mat: float4x4 from 4 K values x 4 N values
    - If B has N contiguous (trans_b=N): b_mat[j] is a contiguous float4 row,
      use b_mat * a_vec
    - If B has K contiguous (trans_b=T): b_mat[j] is a contiguous float4 column,
      use transpose(b_mat) * a_vec
    """
    label = f"{trans_a}{trans_b}".lower()

    # A vector: 4 consecutive K values for row m
    if trans_a == "N":
        a_setup = "  constant auto *A_ptr = reinterpret_cast<constant float4 *>(A + m * K);"
        a_vec = "A_ptr[k]"
    else:
        a_setup = ""
        a_vec = "float4(A[(k*4+0)*M+m], A[(k*4+1)*M+m], A[(k*4+2)*M+m], A[(k*4+3)*M+m])"

    # B matrix: 4x4 block and multiply
    if trans_b == "T":
        # B[N,K] row-major, K contiguous. Read float4 along K for 4 consecutive N rows.
        b_setup = "  constant auto *B_ptr = reinterpret_cast<constant float4 *>(B + n * 4 * K);"
        b_mat_fill = "b_mat[j] = B_ptr[k + j * K / 4];"
        multiply = f"transpose(b_mat) * {a_vec}"
    else:
        # B[K,N] row-major, N contiguous. Read float4 along N for 4 consecutive K rows.
        b_setup = ""
        b_mat_fill = "b_mat[j] = *reinterpret_cast<constant float4*>(B + (k*4+j)*N + n*4);"
        multiply = f"b_mat * {a_vec}"

    return f"""// SIMD(mat4xvec4)({label})
using namespace metal;

kernel void gemm_{label}(constant float *A [[buffer(0)]],
                    constant float *B [[buffer(1)]],
                    device float *C [[buffer(2)]],
                    constant uint3 &sizes [[buffer(3)]],
                    uint2 thread_index [[thread_position_in_grid]]) {{
  const uint M = sizes.x;
  const uint K = sizes.y;
  const uint N = sizes.z;
  const uint m = thread_index.y;
  const uint n = thread_index.x;
{a_setup}
{b_setup}

  float4 rc = 0.0;
  for (uint k = 0; k < K / 4; k++) {{
    float4x4 b_mat;
    for(int j = 0; j < 4; ++j) {{
      {b_mat_fill}
    }}
    rc += {multiply};
  }}
  reinterpret_cast<device float4*>(C + m * N)[n] = rc;
}}
"""


mat4_gemm_src = make_mat4_gemm_src("N", "T")


def _get_layout(tensor):
    """Determine if a 2D tensor is row-major ('N') or column-major ('T')."""
    if tensor.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {tensor.dim()}D")
    s0, s1 = tensor.stride()
    d0, d1 = tensor.shape
    if s1 == 1 and s0 == d1:
        return "N"
    elif s0 == 1 and s1 == d0:
        return "T"
    else:
        raise ValueError(f"Non-contiguous tensor with strides {(s0, s1)}")


class GemmDispatcher:
    """Dispatches GEMM to the best available kernel based on input layouts.

    Kernels are registered by layout key (e.g. 'nt') with a dispatch function.
    At call time, input layouts are inferred from strides. If no kernel matches,
    inputs are transposed (.T.contiguous()) to match an available kernel.
    """

    def __init__(self):
        self._kernels = {}  # layout_key -> (lib, dispatch_fn, name)

    def register(self, source, dispatch_fn):
        lib = torch.mps.compile_shader(source)
        name = source.strip().split("\n")[0][3:]
        # Extract layout from kernel function name: gemm_xx
        for func_name in dir(lib):
            if func_name.startswith("gemm_"):
                layout_key = func_name[5:]  # e.g. "nt"
                self._kernels[layout_key] = (lib, dispatch_fn, name)
                return
        raise ValueError("No gemm_XX function found in shader")

    @property
    def available_layouts(self):
        return list(self._kernels.keys())

    def _adapt(self, A, B, a_layout, b_layout, target_key):
        """Adapt A and B to match the target kernel layout.

        For the target kernel gemm_XY:
          - X='n' means kernel wants A as row-major [M,K]
          - X='t' means kernel wants A as column-major [M,K] (= row-major [K,M])
          - Y='n' means kernel wants B as row-major [K,N]
          - Y='t' means kernel wants B as column-major [K,N] (= row-major [N,K])

        A column-major tensor is the .T of a row-major one in memory.
        .contiguous() converts column-major to row-major (same logical data).
        .T.contiguous() transposes then makes row-major (flips the layout).
        """
        a_need = target_key[0].upper()
        b_need = target_key[1].upper()
        A_out = A if a_layout == a_need else (A.T.contiguous().T if a_need == "T" else A.contiguous())
        B_out = B if b_layout == b_need else (B.T.contiguous().T if b_need == "T" else B.contiguous())
        return A_out, B_out

    def __call__(self, A, B, C, M, N, K):
        """Dispatch C[M,N] = A[M,K] @ B[K,N] using the best available kernel.

        A and B can be in any contiguous layout (row-major or column-major).
        """
        a_layout = _get_layout(A)
        b_layout = _get_layout(B)
        layout_key = (a_layout + b_layout).lower()

        if layout_key in self._kernels:
            lib, dispatch_fn, _ = self._kernels[layout_key]
            dispatch_fn(lib, A, B, C, M, N, K)
            return

        # Adapt inputs to match the first available kernel
        for avail_key, (lib, dispatch_fn, _) in self._kernels.items():
            A_adapted, B_adapted = self._adapt(A, B, a_layout, b_layout, avail_key)
            dispatch_fn(lib, A_adapted, B_adapted, C, M, N, K)
            return

        raise RuntimeError("No kernels registered")

    def name_for(self, a_layout, b_layout):
        key = (a_layout + b_layout).lower()
        if key in self._kernels:
            _, _, name = self._kernels[key]
            return name
        for avail_key, (_, _, name) in self._kernels.items():
            return f"{name} (adapted {key}->{avail_key})"
        return "???"


def _make_naive_dispatch(layout_key):
    def dispatch(lib, A, B, C, M, N, K):
        sizes = torch.tensor([M, K, N], device="mps", dtype=torch.int32)
        getattr(lib, f"gemm_{layout_key}")(A, B, C, sizes, threads=(N, M))
    return dispatch


def _make_mat4_dispatch(layout_key):
    def dispatch(lib, A, B, C, M, N, K):
        sizes = torch.tensor([M, K, N], device="mps", dtype=torch.int32)
        getattr(lib, f"gemm_{layout_key}")(A, B, C, sizes, threads=(N // 4, M), group_size=(8, 8))
    return dispatch


def validate_gemm(dispatcher, M, N, K, a_layout="N", b_layout="N"):
    """Validate C = A @ B where A[M,K] and B[K,N] may be in any layout."""
    A = torch.randn(M, K, device="mps")
    B = torch.randn(K, N, device="mps")
    C = torch.zeros(M, N, device="mps")

    # Optionally transpose storage
    A_input = A.T.contiguous().T if a_layout == "T" else A
    B_input = B.T.contiguous().T if b_layout == "T" else B

    dispatcher(A_input, B_input, C, M, N, K)
    torch.mps.synchronize()
    torch.testing.assert_close(A @ B, C, atol=1e-3, rtol=1e-3)
    name = dispatcher.name_for(a_layout, b_layout)
    print(f"  {name} A={a_layout} B={b_layout} dim {M}x{N}x{K} passed")


def benchmark_gemm(dispatcher, M, N, K, a_layout="N", b_layout="N",
                   repeat_cnt=200, batch_size=10):
    A = torch.randn(M, K, device="mps")
    B = torch.randn(K, N, device="mps")
    C = torch.zeros(M, N, device="mps")

    A_input = A.T.contiguous().T if a_layout == "T" else A
    B_input = B.T.contiguous().T if b_layout == "T" else B

    def dispatch():
        dispatcher(A_input, B_input, C, M, N, K)

    dispatch()
    torch.mps.synchronize()

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
    name = dispatcher.name_for(a_layout, b_layout)
    print(f"  {name} A={a_layout} B={b_layout}: {gflops:.2f} GFLOPs")


def main():
    M, N, K = 64, 4128, 4096
    print(f"Using device {torch.backends.mps.get_name()}")

    dispatchers = {}

    # Naive: register all 4 layout variants
    naive = GemmDispatcher()
    for ta, tb in [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")]:
        key = f"{ta}{tb}".lower()
        naive.register(make_naive_gemm_src(ta, tb), _make_naive_dispatch(key))
    dispatchers["Naive"] = naive

    # SIMD(vec4): register all 4 layout variants
    vec4 = GemmDispatcher()
    for ta, tb in [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")]:
        key = f"{ta}{tb}".lower()
        vec4.register(make_vec4_gemm_src(ta, tb), _make_naive_dispatch(key))
    dispatchers["SIMD(vec4)"] = vec4

    # SIMD(mat4xvec4): register all 4 layout variants
    mat4 = GemmDispatcher()
    for ta, tb in [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")]:
        key = f"{ta}{tb}".lower()
        mat4.register(make_mat4_gemm_src(ta, tb), _make_mat4_dispatch(key))
    dispatchers["SIMD(mat4xvec4)"] = mat4

    layouts = [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")]

    print("\nValidation:")
    for label, d in dispatchers.items():
        for al, bl in layouts:
            validate_gemm(d, M, N, K, al, bl)

    print("\nBenchmark:")
    for label, d in dispatchers.items():
        for al, bl in layouts:
            benchmark_gemm(d, M, N, K, al, bl)


if __name__ == "__main__":
    main()

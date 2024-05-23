#include <metal_stdlib>
using namespace metal;

template <typename T> struct Vec4Type {};

template <> struct Vec4Type<float> {
  using type = float4;
};

template <> struct Vec4Type<half> {
  using type = half4;
};

#if __METAL_VERSION__ >= 310
template <> struct Vec4Type<bfloat> {
  using type = bfloat4;
};
#endif

template <typename T, unsigned blockSize=8>
kernel void
int8pack_mm(constant T *A [[buffer(0)]], constant char *B [[buffer(1)]],
            constant T *scales [[buffer(2)]],
            device T *outputData [[buffer(3)]],
            constant int3 &sizes [[buffer(4)]], // M, K, N
            uint2 group_index [[threadgroup_position_in_grid]],
            uint2 threadgroup_index [[thread_position_in_threadgroup]]) {
  using vecT = typename Vec4Type<T>::type;
  const uint K = sizes.y;
  const uint N = sizes.z;
  int out_idx = (group_index.x * blockSize + threadgroup_index.x);
  int n = 4 * (out_idx % (N/4));
  int m = 4 * (out_idx / (N/4));
  // Offset pointers
  constant vecT *A_ptr = reinterpret_cast<constant vecT *>(A + m * K);
  constant char4 *B_ptr = reinterpret_cast<constant char4 *>(B + n * K);

  outputData += m * N;

  float4x4 rc;
  for(int j = 0; j < 4; ++j) {
    rc[j] = float4(0.0);
  }
  for (unsigned k = threadgroup_index.y * 4; k < K; k += 4 * blockSize) {
    threadgroup_barrier(mem_flags::mem_none);

    float4x4 b_mat;
    for(int j = 0; j < 4; ++j) {
      b_mat[j] = float4(B_ptr[k / 4 + j * K / 4]);
    }

    float4x4 a_mat;
    for(int j = 0; j < 4; ++j) {
      a_mat[j] = float4(A_ptr[k / 4 + j * K / 4]);
    }

    rc += transpose(b_mat) * a_mat;
  }

  // Accumulate results acorss SIMD group? (8 threads using vec4)
  threadgroup float4x4 tgp_memory[blockSize][blockSize];
  tgp_memory[threadgroup_index.x][threadgroup_index.y] = rc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (threadgroup_index.y == 0) {
    for (int i = 1; i < blockSize; i++) {
      rc += tgp_memory[threadgroup_index.x][i];
    }
    *reinterpret_cast<device vecT *>(outputData + n) =
        vecT(rc[0] * float4(*reinterpret_cast<constant vecT *>(scales + n)));
    *reinterpret_cast<device vecT *>(outputData + n + N) =
        vecT(rc[1] * float4(*reinterpret_cast<constant vecT *>(scales + n)));
    *reinterpret_cast<device vecT *>(outputData + n + 2 * N) =
        vecT(rc[2] * float4(*reinterpret_cast<constant vecT *>(scales + n)));
    *reinterpret_cast<device vecT *>(outputData + n + 3 * N) =
        vecT(rc[3] * float4(*reinterpret_cast<constant vecT *>(scales + n)));
  }
}

#define INSTANTIATE_INT8MM(DTYPE)                                              \
  template [[host_name("int8pack_mm_" #DTYPE)]] kernel void                    \
  int8pack_mm<DTYPE>(                                                          \
      constant DTYPE * A [[buffer(0)]], constant char *B [[buffer(1)]],        \
      constant DTYPE *scales [[buffer(2)]],                                    \
      device DTYPE *outputData [[buffer(3)]],                                  \
      constant int3 &sizes [[buffer(4)]],                                      \
      uint2 group_index [[threadgroup_position_in_grid]],                      \
      uint2 threadgroup_index [[thread_position_in_threadgroup]]);

INSTANTIATE_INT8MM(half);
INSTANTIATE_INT8MM(float);
#if __METAL_VERSION__ >= 310
INSTANTIATE_INT8MM(bfloat);
#endif

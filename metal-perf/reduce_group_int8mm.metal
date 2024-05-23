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
            constant uint3 &sizes [[buffer(4)]],
            uint2 group_index [[threadgroup_position_in_grid]],
            uint2 threadgroup_index [[thread_position_in_threadgroup]]) {
  using vecT = typename Vec4Type<T>::type;
  const uint lda = sizes.y;
  const uint ldc = sizes.z;
  int out_idx = (group_index.x * blockSize + threadgroup_index.x) * 4;
  int n = out_idx % sizes.z;
  int m = out_idx / sizes.z;
  // Offset pointers
  A += m * lda;
  B += n * lda;
  outputData += m *ldc;

  float4 rc = 0;
  for (unsigned k = threadgroup_index.y * 4; k < sizes.y; k += 4 * blockSize) {
    threadgroup_barrier(mem_flags::mem_none);
    auto a_val = float4(*reinterpret_cast<constant vecT *>(A  + k));
    float4x4 b_val;
    for (int i = 0; i < 4; ++i) {
      b_val[i] = float4(*reinterpret_cast<constant char4 *>(B + i * lda + k));
    }
    rc += transpose(b_val) * a_val;
  }

  // Accumulate results acorss SIMD group? (8 threads using vec4)
  threadgroup float4 tgp_memory[blockSize][blockSize];
  tgp_memory[threadgroup_index.x][threadgroup_index.y] = rc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (threadgroup_index.y == 0) {
    for (unsigned i = 1; i < blockSize; i++) {
      rc += tgp_memory[threadgroup_index.x][i];
    }
    *reinterpret_cast<device vecT *>(outputData + n) =
        vecT(rc * float4(*reinterpret_cast<constant vecT *>(scales + n)));
  }
}

#define INSTANTIATE_INT8MM(DTYPE)                                              \
  template [[host_name("int8pack_mm_" #DTYPE)]] kernel void                    \
  int8pack_mm<DTYPE>(                                                          \
      constant DTYPE * A [[buffer(0)]], constant char *B [[buffer(1)]],        \
      constant DTYPE *scales [[buffer(2)]],                                    \
      device DTYPE *outputData [[buffer(3)]],                                  \
      constant uint3 &sizes [[buffer(4)]],                                     \
      uint2 group_index [[threadgroup_position_in_grid]],                      \
      uint2 threadgroup_index [[thread_position_in_threadgroup]]);

INSTANTIATE_INT8MM(half);
INSTANTIATE_INT8MM(float);
#if __METAL_VERSION__ >= 310
INSTANTIATE_INT8MM(bfloat);
#endif

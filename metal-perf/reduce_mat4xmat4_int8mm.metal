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

template <typename T>
kernel void int8pack_mm(constant T *A [[buffer(0)]],
                        constant char *B [[buffer(1)]],
                        constant T *scales [[buffer(2)]],
                        device T *outputData [[buffer(3)]],
                        constant uint3 &sizes [[buffer(4)]], // M, K, N
                        uint2 thread_index [[thread_position_in_grid]]) {
  const uint K = sizes.y;
  const uint N = sizes.z;
  const uint m = thread_index.y; // 0..M/4-1
  const uint n = thread_index.x; // 0..N/4-1
  using vecT = typename Vec4Type<T>::type;
  constant vecT *A_ptr = reinterpret_cast<constant vecT *>(A + m * 4 * K);
  constant char4 *B_ptr = reinterpret_cast<constant char4 *>(B + n * 4 * K);

  float4x4 rc;
  for(int j = 0; j < 4; ++j) {
    rc[j] = float4(0.0);
  }
  for (uint k = 0; k < K / 4; k++) {
    float4x4 b_mat;
    for(int j = 0; j < 4; ++j) {
      b_mat[j] = float4(B_ptr[k + j * K / 4]);
    }
    float4x4 a_mat;
    for(int j = 0; j < 4; ++j) {
      a_mat[j] = float4(A_ptr[k + j * K / 4]);
    }
    rc += transpose(b_mat) * a_mat;
  }
  reinterpret_cast<device vecT*>(outputData + 4 * m * N)[n] = vecT(rc[0] * float4(reinterpret_cast<constant vecT *>(scales)[n]));
  reinterpret_cast<device vecT*>(outputData + (4 * m + 1) * N)[n] = vecT(rc[1] * float4(reinterpret_cast<constant vecT *>(scales)[n]));
  reinterpret_cast<device vecT*>(outputData + (4 * m + 2) * N)[n] = vecT(rc[2] * float4(reinterpret_cast<constant vecT *>(scales)[n]));
  reinterpret_cast<device vecT*>(outputData + (4 * m + 3) * N)[n] = vecT(rc[3] * float4(reinterpret_cast<constant vecT *>(scales)[n]));
}

#define INSTANTIATE_INT8MM(DTYPE)                                              \
  template [[host_name("int8pack_mm_" #DTYPE)]] kernel void                    \
  int8pack_mm<DTYPE>(constant DTYPE * A [[buffer(0)]],                         \
                     constant char *B [[buffer(1)]],                           \
                     constant DTYPE *scales [[buffer(2)]],                     \
                     device DTYPE *outputData [[buffer(3)]],                   \
                     constant uint3 &sizes [[buffer(4)]],                      \
                     uint2 thread_index [[thread_position_in_grid]])

INSTANTIATE_INT8MM(half);
INSTANTIATE_INT8MM(float);
#if __METAL_VERSION__ >= 310
INSTANTIATE_INT8MM(bfloat);
#endif

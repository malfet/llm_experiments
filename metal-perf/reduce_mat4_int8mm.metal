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
                        constant uint3 &sizes [[buffer(4)]],
                        uint2 thread_index [[thread_position_in_grid]]) {
  const uint lda = sizes.y;
  const uint ldc = sizes.z;
  const uint m = thread_index.y; // 0..sizes.x-1
  const uint n = thread_index.x; // 0..sizes.z/4-1
  using vecT = typename Vec4Type<T>::type;
  constant vecT *A_ptr = reinterpret_cast<constant vecT *>(A + m * lda);
  constant char4 *B_ptr = reinterpret_cast<constant char4 *>(B + n * 4 * lda);

  float4 rc = 0.0;
  for (uint k = 0; k < sizes.y / 4; k++) {
    float4x4 b_mat;
    for(int j = 0; j < 4; ++j) {
      b_mat[j] = float4(B_ptr[k + j * lda /4]);
    }
    const auto a_vec = float4(A_ptr[k]);
    rc += transpose(b_mat) * a_vec;
  }
  reinterpret_cast<device vecT*>(outputData + m * sizes.z)[n] = vecT(rc * float4(reinterpret_cast<constant vecT *>(scales)[n]));
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

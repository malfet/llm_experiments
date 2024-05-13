#include <metal_stdlib>
using namespace metal;

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
  const uint n = thread_index.x; // 0..sizes.z-1
  constant T *A_ptr = A + m * lda;
  constant char *B_ptr = B + n * lda;

  float rc = 0.0;
  for (uint k = 0; k < sizes.y; k++) {
    const auto a_val = float(A_ptr[k]);
    const auto b_val = float(B_ptr[k]);
    rc += a_val * b_val;
  }
  outputData[m * sizes.z + n] = T(rc * float(scales[n]));
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

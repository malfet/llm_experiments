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

template <typename T> struct Vec2Type {};

template <> struct Vec2Type<float> {
  using type = float2;
};

template <> struct Vec2Type<half> {
  using type = half2;
};

#if __METAL_VERSION__ >= 310
template <> struct Vec2Type<bfloat> {
  using type = bfloat2;
};
#endif

//    [encoder dispatchThreads:MTLSizeMake(N / 4, 1, 1)
//        threadsPerThreadgroup:MTLSizeMake(8, 1, 1)];

template<typename T, unsigned groupSize>
kernel void int4pack_mm(
    constant T                 * A              [[buffer(0)]],
    constant uchar             * B              [[buffer(1)]],
    constant T                 * scalesAndZeros [[buffer(2)]],
    device   T                 * outputData     [[buffer(3)]],
    constant uint3             & sizes          [[buffer(4)]], // M, K, N
    uint2 group_index [[threadgroup_position_in_grid]],
    uint2 threadgroup_index [[thread_position_in_threadgroup]]) {

    const uint K = sizes.y;
    const uint N = sizes.z;
    const uint nb = group_index.x; // 0..N/32-1
    const uint n4 = 8 * nb + threadgroup_index.x; // 0..N/4-1
    const uint ldb = min(32U,  N - nb * 32);
    const uint32_t k_block = (K + groupSize - 1) / groupSize;

    using vecT = typename Vec4Type<T>::type;

    constant T *A_ptr = A;
    constant uchar *B_ptr = B + (nb * 16 * K);

    float4 rc = 0.0;
    uint k = 0;
    for (uint32_t kb = 0; kb < k_block ; kb ++) {
      float4 scales, zeros;
      for (int i = 0; i < 4; ++i) {
        scales[i] = scalesAndZeros[(kb * N + 4*n4 + i) * 2 + 0];
        zeros[i] = scalesAndZeros[(kb * N + 4*n4 + i) * 2 + 1] - scales[i] * T(8);
      }

      for(uint idx = 0; idx < groupSize && k < K; idx++, k++) {
        const auto a_val = float(A_ptr[k]);
        uchar b_byte0 = B_ptr[(k * ldb + (4*n4 % 32))/2];
        uchar b_byte1 = B_ptr[(k * ldb + (4*n4 % 32))/2 + 1];

        float4 b_val = float4(
          float(b_byte0 & 0x0f),
          float(b_byte0 >> 4),
          float(b_byte1 & 0x0f),
          float(b_byte1 >> 4));

        float4 b_vec = scales * b_val + zeros;

        rc += a_val * b_vec;
      }
    }
    reinterpret_cast<device vecT*>(outputData)[n4] = vecT(rc);
}

#define INSTANTIATE_INT4MM(DTYPE, GSIZE)                                 \
template                                                                 \
[[host_name("int4pack_mm_" #GSIZE "_" #DTYPE)]]                          \
kernel void int4pack_mm<DTYPE, GSIZE>(                                   \
    constant DTYPE             * A              [[buffer(0)]],           \
    constant uchar             * B              [[buffer(1)]],           \
    constant DTYPE             * scalesAndZeros [[buffer(2)]],           \
    device   DTYPE             * outputData     [[buffer(3)]],           \
    constant uint3             & sizes          [[buffer(4)]],           \
    uint2 group_index [[threadgroup_position_in_grid]], \
    uint2 threadgroup_index [[thread_position_in_threadgroup]])

INSTANTIATE_INT4MM(float, 32);
INSTANTIATE_INT4MM(half, 32);
INSTANTIATE_INT4MM(float, 64);
INSTANTIATE_INT4MM(half, 64);
INSTANTIATE_INT4MM(float, 128);
INSTANTIATE_INT4MM(half, 128);
INSTANTIATE_INT4MM(float, 256);
INSTANTIATE_INT4MM(half, 256);
#if __METAL_VERSION__ >= 310
INSTANTIATE_INT4MM(bfloat, 32);
INSTANTIATE_INT4MM(bfloat, 64);
INSTANTIATE_INT4MM(bfloat, 128);
INSTANTIATE_INT4MM(bfloat, 256);
#endif

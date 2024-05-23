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

// dispatchThreads:MTLSizeMake(N/4, M/4, 1)

template<typename T, unsigned groupSize>
kernel void int4pack_mm(
    constant T                 * A              [[buffer(0)]],
    constant uchar             * B              [[buffer(1)]],
    constant T                 * scalesAndZeros [[buffer(2)]],
    device   T                 * outputData     [[buffer(3)]],
    constant uint3             & sizes          [[buffer(4)]], // M, K, N
    uint2                        thread_index   [[thread_position_in_grid]]) {
    const uint K = sizes.y;
    const uint N = sizes.z;
    const uint m = thread_index.y; // 0..M/4-1
    const uint n = thread_index.x; // 0..N/4-1
    const uint nb = n / 8;
    const uint ldb = min(32U,  N - nb * 32);
    const uint32_t k_block = (K + groupSize - 1) / groupSize;

    using vecT = typename Vec4Type<T>::type;
    constant vecT *A_ptr = reinterpret_cast<constant vecT *>(A + m * 4 * K);
    //constant uchar2 *B_ptr = reinterpret_cast<constant uchar2 *>(B + (nb * 16 * K));
    constant uchar *B_ptr = B + (nb * 16 * K);

    float4x4 rc;
    for(int j = 0; j < 4; ++j) {
      rc[j] = float4(0.0);
    }
    uint k = 0;
    for (uint32_t kb = 0; kb < k_block ; kb ++) {
      const T scale0 = scalesAndZeros[(kb * N + 4 * n) * 2 + 0];
      const T zero0 = scalesAndZeros[(kb * N + 4 * n) * 2 + 1] - scale0 * T(8);

      const T scale1 = scalesAndZeros[(kb * N + 4 * n + 1) * 2 + 0];
      const T zero1 = scalesAndZeros[(kb * N + 4 * n + 1) * 2 + 1] - scale1 * T(8);

      const T scale2 = scalesAndZeros[(kb * N + 4 * n + 2) * 2 + 0];
      const T zero2 = scalesAndZeros[(kb * N + 4 * n + 2) * 2 + 1] - scale2 * T(8);

      const T scale3 = scalesAndZeros[(kb * N + 4 * n + 3) * 2 + 0];
      const T zero3 = scalesAndZeros[(kb * N + 4 * n + 3) * 2 + 1] - scale3 * T(8);

      for(uint idx = 0; idx < groupSize && k < K; idx += 4, k += 4) {
        float4x4 a_mat;
        for(int j = 0; j < 4; ++j) {
          a_mat[j] = float4(A_ptr[k/4 + j * K / 4]);
        }

        float4x4 t_b_mat;
        for(int j = 0; j < 4; ++j) {
          uchar b_val0 = B_ptr[((k + j) * ldb + ((4 * n) % 32))/2];
          uchar b_val1 = B_ptr[((k + j) * ldb + ((4 * n) % 32))/2 + 1];

          t_b_mat[j] = float4(
            scale0 * float(b_val0 & 0x0f) + zero0,
            scale1 * float(b_val0 >> 4) + zero1,
            scale2 * float(b_val1 & 0x0f) + zero2,
            scale3 * float(b_val1 >> 4) + zero3);
        }

        rc += t_b_mat * a_mat;
      }
    }
    reinterpret_cast<device vecT*>(outputData + 4 * m * N)[n] = vecT(rc[0]);
    reinterpret_cast<device vecT*>(outputData + (4 * m + 1) * N)[n] = vecT(rc[1]);
    reinterpret_cast<device vecT*>(outputData + (4 * m + 2) * N)[n] = vecT(rc[2]);
    reinterpret_cast<device vecT*>(outputData + (4 * m + 3) * N)[n] = vecT(rc[3]);
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
    uint2                        thread_index [[thread_position_in_grid]])

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

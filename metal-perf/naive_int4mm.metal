#include <metal_stdlib>
using namespace metal;

// dispatchThreads:MTLSizeMake(N, M, 1)

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
    const uint m = thread_index.y; // 0..M-1
    const uint n = thread_index.x; // 0..N-1
    const uint nb = n / 32;
    const uint ldb = min(32U,  N - nb * 32);
    const uint32_t k_block = (K + groupSize - 1) / groupSize;
    constant T *A_ptr = A + m * K;
    constant uchar *B_ptr = B + (nb * 16 * K);

    float rc = 0.0;
    uint k = 0;
    for (uint32_t kb = 0; kb < k_block ; kb ++) {
      const T scale = scalesAndZeros[(kb * N + n) * 2 + 0];
      const T zero = scalesAndZeros[(kb * N + n) * 2 + 1] - scale * T(8);
      for(uint idx = 0; idx < groupSize && k < K; idx++, k++) {
        const auto a_val = float(A_ptr[k]);
        uchar b_val = B_ptr[(k * ldb + (n % 32))/2];
        b_val = (n & 1) == 0 ? b_val & 0x0f : (b_val >> 4);
        rc += a_val * float(scale * T(b_val) + zero);
      }
    }
    outputData[m * N + n] = T(rc);
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

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

//    [encoder dispatchThreads:MTLSizeMake(N / 2, 4, 1)
//        threadsPerThreadgroup:MTLSizeMake(16, 4, 1)];

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
    const uint n2 = 16 * nb + threadgroup_index.x; // 0..N/2-1
    const uint ldb = min(32U,  N - nb * 32);
    const uint32_t k_block = (K + groupSize - 1) / groupSize;

    using vec2T = typename Vec2Type<T>::type;
    using vec4T = typename Vec4Type<T>::type;

    constant vec4T *A_ptr = reinterpret_cast<constant vec4T *>(A);
    constant uchar *B_ptr = B + (nb * 16 * K);

    float2 rc = 0.0;
    for (uint k = threadgroup_index.y * 4; k < K; k += 16) {
      threadgroup_barrier(mem_flags::mem_none);

      const auto a_vec = float4(A_ptr[k/4]);
      uchar4 b_byte;
      for (int i = 0; i < 4; i++) {
        b_byte[i] = B_ptr[((k + i) * ldb + (2*n2 % 32))/2];
      }

      uint kb = k / groupSize;
      float2 scales, zeros;
      for (int i = 0; i < 2; ++i) {
        scales[i] = scalesAndZeros[(kb * N + 2*n2 + i) * 2 + 0];
        zeros[i] = scalesAndZeros[(kb * N + 2*n2 + i) * 2 + 1] - scales[i] * T(8);
      }

      float4x2 b_mat;

      for (int i = 0; i < 4; i++) {
        b_mat[i] = scales * float2(
          float(b_byte[i] & 0x0f),
          float(b_byte[i] >> 4)) + zeros;
      }

      rc += b_mat * a_vec;
    }

    threadgroup float2 tgp_memory[16][4];
    tgp_memory[threadgroup_index.x][threadgroup_index.y] = rc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (threadgroup_index.y == 0) {
      for (unsigned i = 1; i < 4; i++) {
        rc += tgp_memory[threadgroup_index.x][i];
      }
      reinterpret_cast<device vec2T*>(outputData)[n2] = vec2T(rc);
    }
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

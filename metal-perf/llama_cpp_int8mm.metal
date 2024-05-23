//
//  LlamaCppInt8Linear.metal
//  CustomLinear
//
//  Created by Mengwei Liu on 5/13/24.
//


#include <metal_stdlib>
using namespace metal;
template <typename T> struct BlockType {};

template <> struct BlockType<float> {
  using type4x4 = float4x4;
  using type2x4 = float2x4;
  using simdgroup_type8x8 = simdgroup_float8x8;
};

template <> struct BlockType<half> {
  using type4x4 = half4x4;
  using type2x4 = half2x4;
  using simdgroup_type8x8 = simdgroup_half8x8;
};


template<typename T>
void dequantize_f32(constant float * src, constant T * scales, uint index, thread float4x4 & reg) {
    for (int i = 0; i < 16; i++){
        reg[i/4][i%4] = src[i];
    }
}

template<typename T>
void dequantize_f16(constant half * src, constant T * scales, uint index, thread float4x4 & reg) {
    for (int i = 0; i < 16; i++){
        reg[i/4][i%4] = src[i];
    }
}

template<typename T>
void dequantize_i8(constant char * src, constant T * scales, uint index, thread float4x4 & reg) {
    T scale = scales[index];
    for (int i = 0; i < 16; i++){
        reg[i/4][i%4] = src[i] * scale;
    }
}

#define BLOCK_SIZE_M 64 // 8 simdgroup matrices from matrix A
#define BLOCK_SIZE_N 32 // 4 simdgroup matrices from matrix B
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 4 // each thread take 4 simdgroup matrices from matrix A
#define THREAD_MAT_N 2 // each thread take 2 simdgroup matrices from matrix B
#define THREAD_PER_ROW 2 // 2 thread for each row in matrix A to load numbers
#define THREAD_PER_COL 4 // 4 thread for each row in matrix B to load numbers
#define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8
#define SG_MAT_ROW 8

// T: input type, W: weight type
template<typename T, typename W, void (*dequantize_func)(constant W *, constant T *, uint, thread float4x4 &)>
kernel void kernel_mul_mm(
    constant T                 * A              [[buffer(0)]],  // 2 x 4096
    constant char              * B              [[buffer(1)]],  // 1024 x 4096
    constant T                 * scales         [[buffer(2)]],
    device T                   * outputData     [[buffer(3)]],  // 2 x 1024
    constant uint3             & sizes          [[buffer(4)]],
    threadgroup uchar          * shared_memory  [[threadgroup(0)]], // threadgroup buffer at index 0
    uint3                        tgpig          [[threadgroup_position_in_grid]], // 3d coordinates
    uint                         tiitg          [[thread_index_in_threadgroup]], // 128 per threadgroup
    uint                         sgitg          [[simdgroup_index_in_threadgroup]]) {

    using T4x4 = typename BlockType<T>::type4x4;
    using T2x4 = typename BlockType<T>::type2x4;
    using Tsimd8x8 = typename BlockType<T>::simdgroup_type8x8;
    // sizes: x = M, y = K, z = N
    // pytorch: M x K @ N x K -> M x N
    // ggml: K x N @ K x M -> N x M
    uint32_t ne00 = sizes.y; // K
    uint32_t ne01 = sizes.z; // N
    uint32_t nb00 = sizeof(W);
    uint32_t nb01 = nb00 * ne00;
    uint32_t nb02 = nb01 * ne01;
    uint32_t ne10 = sizes.y; // K
    uint32_t ne11 = sizes.x; // M
    uint32_t nb10 = sizeof(T);
    uint32_t nb11 = nb10 * ne10;
    uint32_t nb12 = nb11 * ne11;
    uint32_t ne0 = sizes.z; // N
    uint32_t ne1 = sizes.x; // M
    constant char * src0 = (constant char *)B;
    constant char * src1 = (constant char *)A;

    // 8192 for sa, 4096 for sb
    threadgroup float * sa = (threadgroup float *)(shared_memory);
    threadgroup T     * sb = (threadgroup T     *)(shared_memory + 8192);

    const uint r0 = tgpig.y;
    const uint r1 = tgpig.x;

    // if this block is of 64x32 shape or smaller
    short n_rows = (ne0 - r0 * BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0 * BLOCK_SIZE_M) : BLOCK_SIZE_M;
    short n_cols = (ne1 - r1 * BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1 * BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_float8x8  ma[4]; // dequantized weight
    Tsimd8x8 mb[2]; // input
    Tsimd8x8 c_res[8]; // outer product result
    for (int i = 0; i < 8; i++){
        c_res[i] = make_filled_simdgroup_matrix<T, 8>(0.f);
    }

    constant W * x = (constant W *)(src0
        + nb01 * (r0 * BLOCK_SIZE_M + thread_row)
        + nb00 * (BLOCK_SIZE_K / THREAD_PER_ROW * (tiitg % THREAD_PER_ROW)));
    constant T * y = (constant T *)(src1
        + nb11 * (r1 * BLOCK_SIZE_N + thread_col)
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));
    // DEBUG:
    constant T4x4 * temp_y = (constant T4x4 *)y;
    constant W * temp_x = (constant W *)x;

    // scales index
    uint scale_index = r0 * BLOCK_SIZE_M + thread_row;

    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        float4x4 temp_a;
        dequantize_func(x, scales, scale_index, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (int i = 0; i < 16; i++) {
            // for example, tiitg 32, i 12 -> 0 + 1 = 1, it needs to work on sg mat grid row 1
            int sg_mat_grid_row_index = (tiitg % THREAD_PER_ROW) * THREAD_PER_ROW + i / 8;
            // same example, sg mat grid col index: 32 / 2 / 8 = 2, so currently need to work with sg mat at (1, 2)
            int sg_mat_grid_col_index = tiitg / THREAD_PER_ROW / 8;
            // now inside sg mat, which index to write to? starting point is SG_MAT_SIZE * sg_mat_offset
            int row_offset = i & 7;
            int col_offset = (tiitg / THREAD_PER_ROW) % 8;
            // now calculates the overall offset for sa
            int sa_offset = (sg_mat_grid_row_index * 8 + sg_mat_grid_col_index) * 64 + (row_offset * 8 + col_offset);
            float temp_a_val = temp_a[i/4][i%4];
            *(sa + sa_offset) = temp_a[i/4][i%4];
        }
        // read 8 values for input matrix
        *(threadgroup T2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) = *((constant T2x4 *)y);

        x += BLOCK_SIZE_K;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup float * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
        threadgroup T     * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));
        // DEBUG:
        threadgroup float4x4 * temp_lsma = (threadgroup float4x4 *)lsma;
        threadgroup T4x4     * temp_lsmb = (threadgroup T4x4     *)lsmb;
        #pragma unroll(4)
        for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            #pragma unroll(4)
            for (int i = 0; i < 4; i++) {
                simdgroup_load(ma[i],lsma + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll(2)
            for (int i = 0; i < 2; i++) {
                simdgroup_load(mb[i],lsmb + SG_MAT_SIZE * i);
            }

            lsma += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
            lsmb += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;

            #pragma unroll(8)
            for (int i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(c_res[i], mb[i/4], ma[i%4], c_res[i]);
            }
        }
    }

    if ((r0 + 1) * BLOCK_SIZE_M <= ne0 && (r1 + 1) * BLOCK_SIZE_N <= ne1) {
        device T * C = outputData + (BLOCK_SIZE_M * r0 + 32 * (sgitg &  1)) \
                               + (BLOCK_SIZE_N * r1 + 16 * (sgitg >> 1)) * ne0;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], C + 8 * (i%4) + 8 * ne0 * (i/4), ne0);
            // DEBUG:
            device T4x4 * temp_C = (device T4x4 *) (C + 8 * (i%4) + 8 * ne0 * (i/4));
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup T * temp_str = ((threadgroup T *)shared_memory) \
                                      + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_M;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_M * (i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        device T * C = outputData + (BLOCK_SIZE_M * r0) + (BLOCK_SIZE_N * r1) * ne0;
        if (sgitg == 0) {
            for (int i = 0; i < n_rows; i++) {
                for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                    *(C + i + j * ne0) = *(temp_str + i + j * BLOCK_SIZE_M);
                    // DEBUG:
                    device T4x4 * temp_C = (device T4x4 *) (C + i + j * ne0);
                }
            }
        }
    }
}



typedef decltype(kernel_mul_mm<float, float, dequantize_f32>) mat_mm_f32_f32;

template [[host_name("kernel_mul_mm_f32_f32")]]
kernel mat_mm_f32_f32 kernel_mul_mm<float, float, dequantize_f32>;

typedef decltype(kernel_mul_mm<half, half, dequantize_f16>) mat_mm_f16_f16;

template [[host_name("kernel_mul_mm_f16_f16")]]
kernel mat_mm_f16_f16 kernel_mul_mm<half, half, dequantize_f16>;

typedef decltype(kernel_mul_mm<float, char, dequantize_i8>) mat_mm_f32_i8;

template [[host_name("int8pack_mm_float")]]
kernel mat_mm_f32_i8 kernel_mul_mm<float, char, dequantize_i8>;

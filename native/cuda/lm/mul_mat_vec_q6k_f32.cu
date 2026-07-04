// Fused Q6_K × F32 matrix-vector product for LLM decode (M small).
//
// Same idea as mul_mat_vec_q4k_f32: read the Q6_K weight bytes ONCE and dequant inline
// (no F16 materialization), F32 accumulate, one warp per output row. Q6_K is the common
// dtype for the lm_head / output projection (the single largest weight, run every token).
//
// Q6_K super-block: 256 elements / 210 bytes:
//   [128 B ql (low 4 bits)] [64 B qh (high 2 bits, 4/byte)] [16 B int8 scales] [2 B fp16 d]
// Reconstruction mirrors canonical ggml dequantize_row_q6_K (see dequant_q6_k_to_f16.cu):
// 2 halves of 128, each (half,l) work-item emits 4 elements at strides {0,+32,+64,+96}.
//
// The 64 (half,l) work-items are split across the 32 warp lanes (2 items each → 8 elems/lane).
// Layout / launch identical to the Q4_K kernel.

#include <cuda_fp16.h>

#define SUPER_ELEMS 256
#define SUPER_BYTES 210
#define WARP_SIZE 32

extern "C" __global__ void mul_mat_vec_q6k_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ bias,     // may be nullptr
    int N, int K, int M)
{
    const int lane = threadIdx.x;                          // 0..31
    const int n = blockIdx.x * blockDim.y + threadIdx.y;   // output row
    const int m = blockIdx.y;                              // batch row
    if (n >= N || m >= M) return;

    const int nsb = K / SUPER_ELEMS;
    const unsigned char* wrow = weight + (size_t)n * nsb * SUPER_BYTES;
    const float* xrow = input + (size_t)m * K;

    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const unsigned char* block = wrow + (size_t)sb * SUPER_BYTES;
        const unsigned char* ql = block;
        const unsigned char* qh = block + 128;
        const signed char* scales = (const signed char*)(block + 192);
        const float d = __half2float(*(const __half*)(block + 208));
        const float* xb = xrow + sb * SUPER_ELEMS;

        // Each lane owns work-items hl = lane and hl = lane + 32 (of the 64 per super-block).
        #pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            const int hl = lane + rep * WARP_SIZE;
            const int half = hl >> 5;          // 0 or 1
            const int l = hl & 31;             // 0..31
            const int isOffset = l >> 4;       // 0 for l<16, 1 for l>=16
            const unsigned char* qlH = ql + half * 64;
            const unsigned char* qhH = qh + half * 32;
            const signed char* scH = scales + half * 8;
            const int b = half * 128;          // element base within the super-block

            const int q1 = ((qlH[l]      & 0x0F) | (((qhH[l] >> 0) & 0x03) << 4)) - 32;
            const int q2 = ((qlH[l + 32] & 0x0F) | (((qhH[l] >> 2) & 0x03) << 4)) - 32;
            const int q3 = ((qlH[l]      >>   4) | (((qhH[l] >> 4) & 0x03) << 4)) - 32;
            const int q4 = ((qlH[l + 32] >>   4) | (((qhH[l] >> 6) & 0x03) << 4)) - 32;

            acc += d * (float)scH[isOffset + 0] * (float)q1 * xb[b + l];
            acc += d * (float)scH[isOffset + 2] * (float)q2 * xb[b + l + 32];
            acc += d * (float)scH[isOffset + 4] * (float)q3 * xb[b + l + 64];
            acc += d * (float)scH[isOffset + 6] * (float)q4 * xb[b + l + 96];
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        if (bias != nullptr) acc += bias[n];
        output[(size_t)m * N + n] = acc;
    }
}

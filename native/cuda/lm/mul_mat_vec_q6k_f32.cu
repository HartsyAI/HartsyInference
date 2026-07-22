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
    const int isOffset = lane >> 4;    // 0 for lane<16, 1 for lane>=16 -- same for both halves (hl&31 == lane)

    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const unsigned char* block = wrow + (size_t)sb * SUPER_BYTES;
        const unsigned char* ql = block;
        const unsigned char* qh = block + 128;
        const signed char* scales = (const signed char*)(block + 192);
        const float d = __half2float(*(const __half*)(block + 208));
        const float* xb = xrow + sb * SUPER_ELEMS;

        // Issue every load for BOTH half-super-blocks up front, before any bit-unpacking or FMA.
        // ncu on the original interleaved load-then-immediately-use version showed low IPC (1.39 vs
        // 2.70 for the Q4_K kernel) and only ~34% issue-slot occupancy with "low compute throughput
        // AND memory bandwidth -- indicates latency issues": each q-value's load was on the critical
        // path of the very next FMA, serializing on round-trip latency instead of overlapping. Grouping
        // the loads gives the scheduler many independent in-flight requests per warp to hide that
        // latency behind. Every access below is still lane-contiguous (coalesced) exactly as before;
        // only the ORDER of loads vs. arithmetic changed, so the computed value is bit-for-bit identical.
        const unsigned char ql0a = ql[lane],      ql0b = ql[lane + 32];
        const unsigned char qh0  = qh[lane];
        const unsigned char ql1a = ql[64 + lane], ql1b = ql[96 + lane];
        const unsigned char qh1  = qh[32 + lane];
        const signed char sc0_0 = scales[isOffset + 0], sc0_2 = scales[isOffset + 2];
        const signed char sc0_4 = scales[isOffset + 4], sc0_6 = scales[isOffset + 6];
        const signed char sc1_0 = scales[8 + isOffset + 0], sc1_2 = scales[8 + isOffset + 2];
        const signed char sc1_4 = scales[8 + isOffset + 4], sc1_6 = scales[8 + isOffset + 6];
        const float x0a = xb[lane],       x0b = xb[lane + 32];
        const float x0c = xb[lane + 64],  x0d = xb[lane + 96];
        const float x1a = xb[128 + lane], x1b = xb[128 + lane + 32];
        const float x1c = xb[128 + lane + 64], x1d = xb[128 + lane + 96];

        const int q1_0 = ((ql0a & 0x0F) | (((qh0 >> 0) & 0x03) << 4)) - 32;
        const int q2_0 = ((ql0b & 0x0F) | (((qh0 >> 2) & 0x03) << 4)) - 32;
        const int q3_0 = ((ql0a >>   4) | (((qh0 >> 4) & 0x03) << 4)) - 32;
        const int q4_0 = ((ql0b >>   4) | (((qh0 >> 6) & 0x03) << 4)) - 32;
        const int q1_1 = ((ql1a & 0x0F) | (((qh1 >> 0) & 0x03) << 4)) - 32;
        const int q2_1 = ((ql1b & 0x0F) | (((qh1 >> 2) & 0x03) << 4)) - 32;
        const int q3_1 = ((ql1a >>   4) | (((qh1 >> 4) & 0x03) << 4)) - 32;
        const int q4_1 = ((ql1b >>   4) | (((qh1 >> 6) & 0x03) << 4)) - 32;

        acc += d * (float)sc0_0 * (float)q1_0 * x0a;
        acc += d * (float)sc0_2 * (float)q2_0 * x0b;
        acc += d * (float)sc0_4 * (float)q3_0 * x0c;
        acc += d * (float)sc0_6 * (float)q4_0 * x0d;
        acc += d * (float)sc1_0 * (float)q1_1 * x1a;
        acc += d * (float)sc1_2 * (float)q2_1 * x1b;
        acc += d * (float)sc1_4 * (float)q3_1 * x1c;
        acc += d * (float)sc1_6 * (float)q4_1 * x1d;
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

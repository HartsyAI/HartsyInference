// Fused Q8_0 × Q8_1 matrix-vector product for LLM decode, using __dp4a int8 dot products.
//
// Same approach as mul_mat_vec_q4k_q8_1: the activation is pre-quantized to int8 (Q8_1, per-32-block
// scale), the weight is already int8 (Q8_0), so each 32-elem block reduces to 8 __dp4a instructions
// across the warp instead of 32 scalar float dequant-multiply chains. Q8_0 is symmetric (no min
// term), so only the activation scale is needed — no int-sum correction:
//
//   dot_b = d_w[b] * xd[b] * Σ q[i]·xq[i]
//
// Layout: one WARP per output row. Each iteration the warp covers 8 consecutive 32-elem blocks:
// lanes are split into 8 groups of 4 (g = lane>>2 → block, li = lane&3 → the block's li-th 8-elem
// chunk, 2 dp4a per lane per block — the 2-dp4a shape amortizes the per-block scale broadcast and
// doubles the independent int work per lane vs a 1-dp4a-per-lane split). A Q8_0 block is 34 bytes
// (2 B fp16 d + 32 B int8), so quant bytes are only 2-byte aligned — 4-byte operands are assembled
// from aligned u16 loads (llama.cpp's get_int_b2 pattern).
//
// weight [N,K] Q8_0 row-major; xq [M,K] int8; xd [M,K/32] F32.
// Launch: blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N/WARPS_PER_BLOCK), M). Requires K % 32 == 0.

#include <cuda_fp16.h>

#define BLK_ELEMS 32
#define BLK_BYTES 34
#define WARP_SIZE 32

extern "C" __global__ void mul_mat_vec_q8_0_q8_1(
    float* __restrict__ output,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ bias,     // may be nullptr
    int N, int K, int M)
{
    const int lane = threadIdx.x;                          // 0..31
    const int n = blockIdx.x * blockDim.y + threadIdx.y;   // output row
    const int m = blockIdx.y;                              // batch row
    if (n >= N || m >= M) return;

    const int nblk = K / BLK_ELEMS;
    const unsigned char* wrow = weight + (size_t)n * nblk * BLK_BYTES;
    const signed char* xqrow = xq + (size_t)m * K;
    const float* xdrow = xd + (size_t)m * nblk;

    const int g = lane >> 2;       // which of 8 blocks this iteration
    const int li = lane & 3;       // which 8-elem chunk within the block

    float acc = 0.0f;
    for (int b0 = 0; b0 < nblk; b0 += 8) {
        const int b = b0 + g;
        if (b < nblk) {
            const unsigned char* block = wrow + (size_t)b * BLK_BYTES;
            const float dw = __half2float(*(const __half*)block);
            const unsigned short* q16 = (const unsigned short*)(block + 2);
            const unsigned int w16a = q16[li * 4 + 0];
            const unsigned int w16b = q16[li * 4 + 1];
            const unsigned int w16c = q16[li * 4 + 2];
            const unsigned int w16d = q16[li * 4 + 3];
            const int2 xv = *reinterpret_cast<const int2*>(xqrow + b * BLK_ELEMS + li * 8);
            const int wq0 = (int)(w16a | (w16b << 16));
            const int wq1 = (int)(w16c | (w16d << 16));
            int idot = __dp4a(wq0, xv.x, 0);
            idot = __dp4a(wq1, xv.y, idot);
            acc += dw * xdrow[b] * (float)idot;
        }
    }

    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        if (bias != nullptr) acc += bias[n];
        output[(size_t)m * N + n] = acc;
    }
}

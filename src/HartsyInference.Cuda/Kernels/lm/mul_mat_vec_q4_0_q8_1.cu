// Fused Q4_0 × Q8_1 matrix-vector product for LLM decode, using __dp4a int8 dot products.
//
// Q4_0 is the legacy/baseline GGUF quant (older Ollama default tags, e.g. llama3 Q4_0) and the
// int8-activation approach mirrors mul_mat_vec_q8_0_q8_1: activation pre-quantized to Q8_1
// (per-32-block scale), weight nibbles unpacked whole-word and shifted to signed with __vsub4
// (w = d·(q−8), q ∈ [0,15] ⇒ q−8 ∈ [−8,7] fits int8), so no separate min/int-sum term:
//
//   dot_b = d_w[b] * xd[b] * Σ (q[i]−8)·xq[i]
//
// Q4_0 block: 32 elements / 18 bytes: [2 B fp16 d][16 B nibbles]; element i<16 = low nibble of
// qs[i], element i≥16 = high nibble of qs[i−16] (matches Codec_Q4_0 / mul_mat_vec_q4_0_f32).
//
// Layout: one WARP per output row; each iteration covers 8 consecutive blocks (g = lane>>2 →
// block, li = lane&3 → 4 nibble bytes = elements li*4..+3 of BOTH planes, 2 dp4a/lane/block).
// 18-byte blocks are only 2-byte aligned → u16 assembly loads (llama.cpp get_int_b2 pattern).
//
// weight [N,K] Q4_0 row-major; xq [M,K] int8; xd [M,K/32] F32.
// Launch: blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N/WARPS_PER_BLOCK), M). Requires K % 32 == 0.

#include <cuda_fp16.h>

#define BLK_ELEMS 32
#define BLK_BYTES 18
#define WARP_SIZE 32

extern "C" __global__ void mul_mat_vec_q4_0_q8_1(
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
    const int li = lane & 3;       // which 4-byte nibble chunk within the block

    float acc = 0.0f;
    for (int b0 = 0; b0 < nblk; b0 += 8) {
        const int b = b0 + g;
        if (b < nblk) {
            const unsigned char* block = wrow + (size_t)b * BLK_BYTES;
            const float dw = __half2float(*(const __half*)block);
            const unsigned short* q16 = (const unsigned short*)(block + 2);
            const unsigned int qsw = (unsigned int)q16[li * 2 + 0] | ((unsigned int)q16[li * 2 + 1] << 16);
            const int wlo = __vsub4((int)(qsw & 0x0F0F0F0Fu), 0x08080808);
            const int whi = __vsub4((int)((qsw >> 4) & 0x0F0F0F0Fu), 0x08080808);
            const int xvlo = *reinterpret_cast<const int*>(xqrow + b * BLK_ELEMS + li * 4);
            const int xvhi = *reinterpret_cast<const int*>(xqrow + b * BLK_ELEMS + 16 + li * 4);
            int idot = __dp4a(wlo, xvlo, 0);
            idot = __dp4a(whi, xvhi, idot);
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

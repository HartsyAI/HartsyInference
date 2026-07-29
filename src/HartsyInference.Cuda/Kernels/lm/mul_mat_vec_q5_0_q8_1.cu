// Fused Q5_0 × Q8_1 matrix-vector product for LLM decode, using __dp4a int8 dot products.
//
// Q5_0 is llama.cpp's fallback quant in K-quant mixed schemes (Q4_K_M etc.) for tensors whose K
// isn't a multiple of 256 — very common on odd hidden sizes (e.g. qwen2.5-0.5b's 896), so a
// Q4_K_M checkpoint of such a model runs MOSTLY Q5_0 projections. Same int8-activation approach
// as mul_mat_vec_q4_0_q8_1, extended with Q5_0's high-bit plane (w = d·(q−16), q ∈ [0,31] ⇒
// q−16 ∈ [−16,15] fits int8, no min/int-sum term):
//
//   dot_b = d_w[b] * xd[b] * Σ (q[i]−16)·xq[i]
//
// Q5_0 block: 32 elements / 22 bytes: [2 B fp16 d][4 B packed high bits][16 B nibbles];
// element i<16 = low nibble of qs[i] | (bit i of the high-bit word) << 4, element i≥16 = high
// nibble of qs[i−16] | (bit i) << 4 (matches Codec_Q5_0 / mul_mat_vec_q5_0_f32). The high-bit
// word's byte offset alternates 4-byte alignment block-to-block (22-byte stride) → byte loads;
// nibbles are 2-aligned → u16 assembly. High bits are scattered into int8 byte-lanes with the
// llama.cpp shift-mask pattern (source bit k → byte k's bit 4 via << (4 + 7k)).
//
// weight [N,K] Q5_0 row-major; xq [M,K] int8; xd [M,K/32] F32.
// Launch: blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N/WARPS_PER_BLOCK), M). Requires K % 32 == 0.

#include <cuda_fp16.h>

#define BLK_ELEMS 32
#define BLK_BYTES 22
#define WARP_SIZE 32

// Per-lane partial dot over block-groups b0Start, b0Start+b0Stride, … (each group = 8 blocks).
__device__ __forceinline__ float q5_0_q8_1_row_partial(
    const unsigned char* __restrict__ wrow,
    const signed char* __restrict__ xqrow,
    const float* __restrict__ xdrow,
    int nblk, int b0Start, int b0Stride, int lane)
{
    const int g = lane >> 2;       // which of 8 blocks this iteration
    const int li = lane & 3;       // which 4-byte nibble chunk within the block
    const int e = li * 4;          // first element of this lane's low-plane chunk

    float acc = 0.0f;
    for (int b0 = b0Start; b0 < nblk; b0 += b0Stride) {
        const int b = b0 + g;
        if (b < nblk) {
            const unsigned char* block = wrow + (size_t)b * BLK_BYTES;
            const float dw = __half2float(*(const __half*)block);
            const unsigned char* hb = block + 2;
            const unsigned int qh = (unsigned int)hb[0] | ((unsigned int)hb[1] << 8)
                | ((unsigned int)hb[2] << 16) | ((unsigned int)hb[3] << 24);
            const unsigned short* q16 = (const unsigned short*)(block + 6);
            const unsigned int qsw = (unsigned int)q16[li * 2 + 0] | ((unsigned int)q16[li * 2 + 1] << 16);

            const unsigned int qhLo = qh >> e;          // bits e..e+3 → elements e..e+3
            const unsigned int qhHi = qh >> (e + 16);   // bits e+16..e+19 → elements e+16..e+19
            const unsigned int hbLo = ((qhLo << 4) & 0x00000010u) | ((qhLo << 11) & 0x00001000u)
                                    | ((qhLo << 18) & 0x00100000u) | ((qhLo << 25) & 0x10000000u);
            const unsigned int hbHi = ((qhHi << 4) & 0x00000010u) | ((qhHi << 11) & 0x00001000u)
                                    | ((qhHi << 18) & 0x00100000u) | ((qhHi << 25) & 0x10000000u);

            const int wlo = __vsub4((int)((qsw & 0x0F0F0F0Fu) | hbLo), 0x10101010);
            const int whi = __vsub4((int)(((qsw >> 4) & 0x0F0F0F0Fu) | hbHi), 0x10101010);
            const int xvlo = *reinterpret_cast<const int*>(xqrow + b * BLK_ELEMS + e);
            const int xvhi = *reinterpret_cast<const int*>(xqrow + b * BLK_ELEMS + 16 + e);
            int idot = __dp4a(wlo, xvlo, 0);
            idot = __dp4a(whi, xvhi, idot);
            acc += dw * xdrow[b] * (float)idot;
        }
    }
    return acc;
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1(
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
    float acc = q5_0_q8_1_row_partial(
        weight + (size_t)n * nblk * BLK_BYTES,
        xq + (size_t)m * K, xd + (size_t)m * nblk,
        nblk, 0, 8, lane);

    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        if (bias != nullptr) acc += bias[n];
        output[(size_t)m * N + n] = acc;
    }
}

// Block-per-row K-split (long-K/small-N shapes; see mul_mat_vec_q4k_q8_1.cu's ksplit notes).
extern "C" __global__ void mul_mat_vec_q5_0_q8_1_ksplit(
    float* __restrict__ output,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ bias,     // may be nullptr
    int N, int K, int M)
{
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int nblk = K / BLK_ELEMS;
    float acc = q5_0_q8_1_row_partial(
        weight + (size_t)n * nblk * BLK_BYTES,
        xq + (size_t)m * K, xd + (size_t)m * nblk,
        nblk, warp * 8, (int)blockDim.y * 8, lane);

    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);

    __shared__ float partial[16];
    if (lane == 0) partial[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float sum = 0.0f;
        for (int w = 0; w < (int)blockDim.y; ++w) sum += partial[w];
        if (bias != nullptr) sum += bias[n];
        output[(size_t)m * N + n] = sum;
    }
}

// Fused Q5_0 × F32 matrix-vector product for LLM decode (M small).
//
// Same design as mul_mat_vec_q8_0_f32: read the Q5_0 weight bytes ONCE, dequant inline,
// F32 accumulate, one warp per output row. Q5_0 is llama.cpp's fallback quant for K-quant
// mixed schemes (Q4_K_M etc.) on tensors whose K isn't a multiple of 256 — without this
// kernel those tensors silently miss every fused GEMV path and fall back to the ~10-20×
// slower dequant-to-F16-then-cuBLAS route (observed: qwen2.5-0.5b, hidden=896, ~2.7× slower
// decode than its own Q8_0 quant of the identical model, because Q4_K_M substitutes Q5_0 for
// every K=896 projection and none of them had a fast path).
//
// Q5_0 block: 32 elements / 22 bytes: [2B fp16 scale][4B packed high-bits][16B packed
// low-nibbles]. x[i] = scale * (q[i] - 16), q[i] = lowNibble(i) | (highBit(i) << 4) ∈ [0,31].
// For i in [0,16): low nibble = low half of qs[i], high bit = bit i of the packed word.
// For i in [16,32): low nibble = high half of qs[i-16], high bit = bit i of the packed word
// (matches Codec_Q5_0.DequantizeToF32 exactly — ground truth for the bit layout).
// One warp (32 lanes) per output row; lane l owns element l.
//
// Launch: blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N / WARPS_PER_BLOCK), M).

#include <cuda_fp16.h>

#define BLK_ELEMS 32
#define BLK_BYTES 22
#define HALF_BLOCK 16
#define WARP_SIZE 32

extern "C" __global__ void mul_mat_vec_q5_0_f32(
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

    const int nblk = K / BLK_ELEMS;
    const unsigned char* wrow = weight + (size_t)n * nblk * BLK_BYTES;
    const float* xrow = input + (size_t)m * K;

    const int subIdx = lane < HALF_BLOCK ? lane : lane - HALF_BLOCK;   // byte index within the 16-byte nibble pack
    const bool highHalf = lane >= HALF_BLOCK;

    float acc = 0.0f;
    for (int b = 0; b < nblk; ++b) {
        const unsigned char* block = wrow + (size_t)b * BLK_BYTES;
        const float scale = __half2float(*(const __half*)block);
        // BLK_BYTES=22 is not a multiple of 4, so successive blocks' "+2" offset alternates between 4-byte
        // aligned and misaligned — a vector uint load here faults (CUDA_ERROR_MISALIGNED_ADDRESS) on roughly
        // half the blocks. Byte reads have no alignment requirement, so assemble the word from 4 of them.
        const unsigned char* hb = block + 2;
        const unsigned int highBits = (unsigned int)hb[0] | ((unsigned int)hb[1] << 8)
            | ((unsigned int)hb[2] << 16) | ((unsigned int)hb[3] << 24);
        const unsigned char qByte = block[6 + subIdx];
        const int nibble = highHalf ? ((qByte >> 4) & 0xF) : (qByte & 0xF);
        const int highBit = (int)((highBits >> lane) & 0x1) << 4;
        const int qv = nibble | highBit;
        acc += scale * (float)(qv - 16) * xrow[b * BLK_ELEMS + lane];
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

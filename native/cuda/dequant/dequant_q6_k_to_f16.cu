// Dequantize Q6_K → F16 on GPU.
//
// Q6_K layout (256 elements per super-block, 210 bytes):
//   [128 bytes ql (low 4 bits per element)]
//   [64 bytes qh (high 2 bits per element, packed 4-per-byte)]
//   [16 bytes int8 scales (one per 16-element sub-block)]
//   [2 bytes FP16 d (super-block scale)]
//
// Reconstruction (canonical ggml `dequantize_row_q6_K`): 256 elements processed in 2 halves of 128.
// Each half consumes 64 bytes ql, 32 bytes qh, 8 scales. Inner pattern over l ∈ [0..31] writes
// 4 elements at strides {+0, +32, +64, +96} with sub-block scale index `scH[isOffset+{0,2,4,6}]`
// where `isOffset = l / 16` (alternating between two scale rows per 16-element half).
//
// Launch: gridDim.x = numSuperBlocks, blockDim.x = 64.
// 2 halves × 32 (l-values per half) = 64 threads cover all 256 elements (each thread emits 4).

#include <cuda_fp16.h>

#define SUPER_ELEMS 256
#define SUPER_BYTES 210

extern "C" __global__ void dequant_q6_k_to_f16(
    __half* __restrict__ output,
    const unsigned char* __restrict__ input,
    unsigned int superBlockCount)
{
    const unsigned int sb = blockIdx.x;
    if (sb >= superBlockCount) return;

    const unsigned char* block = input + (size_t)sb * SUPER_BYTES;
    const unsigned char* ql = block;
    const unsigned char* qh = block + 128;
    const signed char* scales = (const signed char*)(block + 192);
    const float d = __half2float(*(const __half*)(block + 208));

    const unsigned int tid = threadIdx.x;
    if (tid < 64) {
        const int half = tid / 32;          // 0 or 1
        const int l = tid % 32;             // 0..31
        const int isOffset = l / 16;        // 0 for l<16, 1 for l>=16
        const unsigned char* qlH = ql + half * 64;
        const unsigned char* qhH = qh + half * 32;
        const signed char* scH = scales + half * 8;
        const int halfBaseElem = half * 128;

        const int q1 = ((qlH[l] & 0x0F) | (((qhH[l] >> 0) & 0x03) << 4)) - 32;
        const int q2 = ((qlH[l + 32] & 0x0F) | (((qhH[l] >> 2) & 0x03) << 4)) - 32;
        const int q3 = ((qlH[l] >> 4) | (((qhH[l] >> 4) & 0x03) << 4)) - 32;
        const int q4 = ((qlH[l + 32] >> 4) | (((qhH[l] >> 6) & 0x03) << 4)) - 32;

        const size_t base = (size_t)sb * SUPER_ELEMS + halfBaseElem;
        output[base + l]      = __float2half(d * (float)scH[isOffset + 0] * (float)q1);
        output[base + l + 32] = __float2half(d * (float)scH[isOffset + 2] * (float)q2);
        output[base + l + 64] = __float2half(d * (float)scH[isOffset + 4] * (float)q3);
        output[base + l + 96] = __float2half(d * (float)scH[isOffset + 6] * (float)q4);
    }
}

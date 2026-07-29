// Dequantize Q5_0 → F16 on GPU.
//
// Q5_0 layout: 32 elements per block, 22 bytes/block.
//   [2 bytes FP16 scale d][4 bytes uint32 qh (the 5th bit of each element)][16 bytes qs (4-bit nibbles)]
// Reconstruction (ggml-quants.c dequantize_row_q5_0):
//   for j in 0..15:
//     y[j]    = (((qs[j] & 0x0F) | ((qh >> j      & 1) << 4)) - 16) * d
//     y[j+16] = (((qs[j] >>   4) | ((qh >> (j+16) & 1) << 4)) - 16) * d
// Each output index `tid` maps to one element: low nibbles fill y[0..15], high nibbles y[16..31], and the
// 5th bit comes from bit `tid` of qh in both halves.
//
// Launch: gridDim.x = blockCount, blockDim.x = 32. Each CUDA block handles one Q5_0 quant block.

#include <cuda_fp16.h>

#define BLOCK_ELEMS 32
#define BLOCK_BYTES 22

extern "C" __global__ void dequant_q5_0_to_f16(
    __half* __restrict__ output,
    const unsigned char* __restrict__ input,
    unsigned int blockCount)
{
    const unsigned int b = blockIdx.x;
    if (b >= blockCount) return;

    const unsigned char* block = input + (size_t)b * BLOCK_BYTES;
    const float d = __half2float(*(const __half*)block);
    unsigned int qh;
    // qh is little-endian uint32 starting at byte offset 2.
    qh = (unsigned int)block[2] | ((unsigned int)block[3] << 8)
       | ((unsigned int)block[4] << 16) | ((unsigned int)block[5] << 24);
    const unsigned char* qs = block + 6;

    const unsigned int tid = threadIdx.x;
    if (tid < BLOCK_ELEMS) {
        const unsigned int j = tid & 15;       // nibble byte index 0..15
        const unsigned int half = tid >> 4;    // 0 = low nibble, 1 = high nibble
        const unsigned int nib = half == 0 ? (qs[j] & 0x0F) : (qs[j] >> 4);
        const unsigned int bit = (qh >> tid) & 1u;
        const int q = (int)(nib | (bit << 4)) - 16;
        output[(size_t)b * BLOCK_ELEMS + tid] = __float2half((float)q * d);
    }
}

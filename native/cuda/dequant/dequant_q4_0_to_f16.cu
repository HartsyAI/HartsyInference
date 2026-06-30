// Dequantize Q4_0 → F16 on GPU.
//
// Q4_0 layout: 32 elements per block, 18 bytes/block.
//   [2 bytes FP16 scale d][16 bytes qs (4-bit nibbles)]
// Reconstruction (ggml-quants.c dequantize_row_q4_0):
//   for j in 0..15:
//     y[j]    = ((qs[j] & 0x0F) - 8) * d
//     y[j+16] = ((qs[j] >>   4) - 8) * d
//
// Launch: gridDim.x = blockCount, blockDim.x = 32. Each CUDA block handles one Q4_0 quant block.

#include <cuda_fp16.h>

#define BLOCK_ELEMS 32
#define BLOCK_BYTES 18

extern "C" __global__ void dequant_q4_0_to_f16(
    __half* __restrict__ output,
    const unsigned char* __restrict__ input,
    unsigned int blockCount)
{
    const unsigned int b = blockIdx.x;
    if (b >= blockCount) return;

    const unsigned char* block = input + (size_t)b * BLOCK_BYTES;
    const float d = __half2float(*(const __half*)block);
    const unsigned char* qs = block + 2;

    const unsigned int tid = threadIdx.x;
    if (tid < BLOCK_ELEMS) {
        const unsigned int j = tid & 15;
        const unsigned int half = tid >> 4;
        const int nib = (int)(half == 0 ? (qs[j] & 0x0F) : (qs[j] >> 4)) - 8;
        output[(size_t)b * BLOCK_ELEMS + tid] = __float2half((float)nib * d);
    }
}

// Dequantize Q8_0 → F16 on GPU.
//
// Q8_0 layout: 32 elements per block, 34 bytes/block.
//   [2 bytes FP16 scale][32 bytes int8 values]
// Reconstruction: x[i] = scale * q[i]
//
// Launch: gridDim.x = ceil(elementCount / BLOCK_ELEMS), blockDim.x = BLOCK_ELEMS = 32.
// Each CUDA block handles one Q8_0 quant block; each thread handles one element.

#include <cuda_fp16.h>

#define BLOCK_ELEMS 32
#define BLOCK_BYTES 34

extern "C" __global__ void dequant_q8_0_to_f16(
    __half* __restrict__ output,
    const unsigned char* __restrict__ input,
    unsigned int blockCount)
{
    const unsigned int b = blockIdx.x;
    if (b >= blockCount) return;

    const unsigned char* block = input + (size_t)b * BLOCK_BYTES;
    const __half scale = *(const __half*)block;
    const signed char* qs = (const signed char*)(block + 2);

    const unsigned int tid = threadIdx.x;
    if (tid < BLOCK_ELEMS) {
        float v = __half2float(scale) * (float)qs[tid];
        output[(size_t)b * BLOCK_ELEMS + tid] = __float2half(v);
    }
}

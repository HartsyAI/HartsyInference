// Dequantize IQ4_NL → F16 on GPU.
//
// IQ4_NL layout (32 elements per block, 18 bytes, canonical ggml block_iq4_nl): [2 bytes FP16 d][16 bytes nibbles],
// low nibble = element i, high nibble = element i + 16, every nibble an index into kvalues_iq4nl (IQ4_XS's table).
//
// Launch: gridDim.x = blockCount, blockDim.x = 32. Each block handles one quant block, each thread one element.

#include <cuda_fp16.h>

#define BLOCK_ELEMS 32
#define BLOCK_BYTES 18

__constant__ signed char kvalues_iq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

extern "C" __global__ void dequant_iq4_nl_to_f16(
    __half* __restrict__ output,
    const unsigned char* __restrict__ input,
    unsigned int blockCount)
{
    const unsigned int b = blockIdx.x;
    if (b >= blockCount) return;

    const unsigned char* block = input + (size_t)b * BLOCK_BYTES;
    const float d = __half2float(*(const __half*)block);
    const unsigned int tid = threadIdx.x;
    if (tid < BLOCK_ELEMS) {
        const unsigned char byte = block[2 + (tid & 15)];
        const int q = (tid < 16) ? (byte & 0xF) : (byte >> 4);
        output[(size_t)b * BLOCK_ELEMS + tid] = __float2half(d * (float)kvalues_iq4nl[q]);
    }
}

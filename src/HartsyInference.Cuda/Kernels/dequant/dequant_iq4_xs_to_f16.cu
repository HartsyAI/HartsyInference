// Dequantize IQ4_XS → F16 on GPU.
//
// IQ4_XS layout (256 elements per super-block, 136 bytes, canonical ggml block_iq4_xs):
//   [2 bytes FP16 d]
//   [2 bytes scales_h: two high bits per 32-element sub-block]
//   [4 bytes scales_l: one low nibble per sub-block]
//   [128 bytes nibbles: per sub-block, 16 bytes; low nibble = element j, high nibble = element j+16]
//
// Reconstruction: x = d · (ls − 32) · kvalues_iq4nl[q], ls = scales_l nibble | (scales_h bits << 4).
//
// Launch: gridDim.x = numSuperBlocks, blockDim.x = 256. Each block handles one super-block, each thread one element.

#include <cuda_fp16.h>

#define SUPER_ELEMS 256
#define SUPER_BYTES 136
#define SUB_ELEMS 32

// Canonical ggml kvalues_iq4nl — the same table IQ4_NL indexes.
__constant__ signed char kvalues_iq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

extern "C" __global__ void dequant_iq4_xs_to_f16(
    __half* __restrict__ output,
    const unsigned char* __restrict__ input,
    unsigned int superBlockCount)
{
    const unsigned int sb = blockIdx.x;
    if (sb >= superBlockCount) return;

    const unsigned char* block = input + (size_t)sb * SUPER_BYTES;
    const float d = __half2float(*(const __half*)block);
    const unsigned int scalesH = (unsigned int)block[2] | ((unsigned int)block[3] << 8);
    const unsigned char* scalesL = block + 4;
    const unsigned char* qs = block + 8;

    const unsigned int tid = threadIdx.x;
    if (tid < SUPER_ELEMS) {
        const int ib = tid / SUB_ELEMS;          // sub-block 0..7
        const int i = tid % SUB_ELEMS;           // element within sub-block 0..31
        const int ls = ((scalesL[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scalesH >> (2 * ib)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        const unsigned char byte = qs[ib * 16 + (i & 15)];
        const int q = (i < 16) ? (byte & 0xF) : (byte >> 4);
        output[(size_t)sb * SUPER_ELEMS + tid] = __float2half(dl * (float)kvalues_iq4nl[q]);
    }
}

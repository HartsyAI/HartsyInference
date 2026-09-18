// Dequantize Q3_K → F16 on GPU.
//
// Q3_K layout (256 elements per super-block, 110 bytes), canonical ggml `block_q3_K`:
//   [32 bytes hmask  — bit 2 of each 3-bit value, one bit per element]
//   [64 bytes qs     — low 2 bits of each value, four per byte]
//   [12 bytes scales — sixteen 6-bit signed scales (-32..31), packed]
//   [2 bytes FP16 d  — super-block scale]
//
// Reconstruction: x = d * scale * (q - (hmask_bit ? 0 : 4)), q in [0..3]. Note the INVERTED high bit: a set
// mask bit means "do not subtract 4", which is the opposite of the obvious reading and the easiest thing to
// get backwards here.
//
// The index derivation matches dequant_q2_k_to_f16 — see its header — with one difference: `hmask` is NOT
// advanced per half. Both halves read the same 32 bytes and are told apart by the mask bit, which walks
// 8 positions as `1 << (4*h + j)`.
//
// Launch: gridDim.x = numSuperBlocks, blockDim.x = 256.

#include <cuda_fp16.h>

#define SUPER_ELEMS 256
#define SUPER_BYTES 110

// Canonical ggml 6-bit signed scale unpack: four low nibbles per byte in scales[0..7], the two high bits
// per entry spread across scales[8..11], biased by -32.
__device__ __forceinline__ int unpack_q3k_scale(const unsigned char* packed, int index)
{
    const int i = index >> 1;               // which of the 8 low bytes
    const int lowByte = packed[i];
    const int high = packed[8 + (i >> 1)];
    if ((index & 1) == 0) {
        const int hi = (i % 2 == 0) ? (high & 0x03) : ((high >> 4) & 0x03);
        return ((lowByte & 0x0F) | (hi << 4)) - 32;
    }
    const int hi = (i % 2 == 0) ? ((high >> 2) & 0x03) : ((high >> 6) & 0x03);
    return ((lowByte >> 4) | (hi << 4)) - 32;
}

extern "C" __global__ void dequant_q3_k_to_f16(
    __half* __restrict__ output,
    const unsigned char* __restrict__ input,
    unsigned int superBlockCount)
{
    const unsigned int sb = blockIdx.x;
    if (sb >= superBlockCount) return;

    const unsigned char* block = input + (size_t)sb * SUPER_BYTES;
    const unsigned char* hmask = block;
    const unsigned char* qs = block + 32;
    const unsigned char* scales = block + 96;
    const float d = __half2float(*(const __half*)(block + 108));

    const unsigned int tid = threadIdx.x;
    if (tid < SUPER_ELEMS) {
        const int h = tid >> 7;              // 0..1
        const int r = tid & 127;
        const int j = r >> 5;                // 0..3, shift group
        const int o = r & 31;
        const int half16 = o >> 4;           // 0..1
        const int l = o & 15;

        const int scale = unpack_q3k_scale(scales, 8 * h + 2 * j + half16);
        const int q = (qs[32 * h + 16 * half16 + l] >> (2 * j)) & 0x03;
        const unsigned int mask = 1u << (4 * h + j);
        const int highOffset = (hmask[16 * half16 + l] & mask) != 0 ? 0 : 4;

        const float value = d * (float)scale * (float)(q - highOffset);
        output[(size_t)sb * SUPER_ELEMS + tid] = __float2half(value);
    }
}

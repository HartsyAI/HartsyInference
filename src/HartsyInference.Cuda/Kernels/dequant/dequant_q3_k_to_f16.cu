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

// Canonical ggml 6-bit signed scale unpack (dequantize_row_q3_K): entry s takes its low nibble from scales[s]
// (s < 8) or the high nibble of scales[s - 8], and its two high bits from scales[8 + s % 4] at bit 2·(s / 4).
__device__ __forceinline__ int unpack_q3k_scale(const unsigned char* packed, int index)
{
    const int low = index < 8 ? (packed[index] & 0x0F) : (packed[index - 8] >> 4);
    const int high = (packed[8 + (index & 3)] >> (2 * (index >> 2))) & 0x03;
    return (low | (high << 4)) - 32;
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

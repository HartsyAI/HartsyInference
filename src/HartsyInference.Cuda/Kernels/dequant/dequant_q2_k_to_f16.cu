// Dequantize Q2_K → F16 on GPU.
//
// Q2_K layout (256 elements per super-block, 84 bytes), canonical ggml `block_q2_K`:
//   [16 bytes scales — per 16-element sub-block, 4-bit scale in the low nibble, 4-bit min in the high]
//   [64 bytes qs — 2-bit quants, four per byte]
//   [2 bytes FP16 d    (super-block scale)]
//   [2 bytes FP16 dmin (super-block min)]
//
// Reconstruction: x = d * (sc & 0xF) * q - dmin * (sc >> 4), q in [0..3].
//
// ggml walks this serially in two halves of 128 elements, each consuming 32 bytes of qs and 8 scale
// entries across four shift positions. The closed form of that walk, for element `tid` of the super-block:
//
//   h = tid / 128            half, also which 32-byte run of qs
//   j = (tid % 128) / 32     shift group, shift = 2*j
//   o = tid % 32             position in the group
//   half16 = o / 16          which of the group's two 16-element runs
//   l = o % 16
//   scale index = 8*h + 2*j + half16
//   qs byte     = 32*h + 16*half16 + l
//
// which is what lets one thread own one element with no cooperation.
//
// Launch: gridDim.x = numSuperBlocks, blockDim.x = 256.

#include <cuda_fp16.h>

#define SUPER_ELEMS 256
#define SUPER_BYTES 84

extern "C" __global__ void dequant_q2_k_to_f16(
    __half* __restrict__ output,
    const unsigned char* __restrict__ input,
    unsigned int superBlockCount)
{
    const unsigned int sb = blockIdx.x;
    if (sb >= superBlockCount) return;

    const unsigned char* block = input + (size_t)sb * SUPER_BYTES;
    const unsigned char* scales = block;
    const unsigned char* qs = block + 16;
    const float d = __half2float(*(const __half*)(block + 80));
    const float dmin = __half2float(*(const __half*)(block + 82));

    const unsigned int tid = threadIdx.x;
    if (tid < SUPER_ELEMS) {
        const int h = tid >> 7;              // 0..1
        const int r = tid & 127;
        const int j = r >> 5;                // 0..3, shift group
        const int o = r & 31;
        const int half16 = o >> 4;           // 0..1
        const int l = o & 15;

        const unsigned char sc = scales[8 * h + 2 * j + half16];
        const int q = (qs[32 * h + 16 * half16 + l] >> (2 * j)) & 0x03;

        const float value = d * (float)(sc & 0x0F) * (float)q - dmin * (float)(sc >> 4);
        output[(size_t)sb * SUPER_ELEMS + tid] = __float2half(value);
    }
}

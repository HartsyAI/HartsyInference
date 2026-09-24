// Dequantize IQ1_S → F16 (50-byte super-block: d, 32 qs, 8 × uint16 qh; see Codec_IQ1_S).
// Launch: gridDim.x = superBlockCount, blockDim.x = 256; one thread per element, the host codec's formula per element.
#define IQ_TABLE_iq1s_grid
#include "iq_tables.cuh"
#include <cuda_fp16.h>

__device__ __forceinline__ unsigned int ld_u32(const unsigned char* p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}
#ifdef IQ_TABLE_kmask_iq2xs
__device__ __forceinline__ float sign_of(unsigned int signs, int j) { return (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f; }
#endif

extern "C" __global__ void dequant_iq1_s_to_f16(__half* __restrict__ output, const unsigned char* __restrict__ input, unsigned int superBlockCount)
{
    const unsigned int sb = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (sb >= superBlockCount || tid >= 256u) return;
    const unsigned char* block = input + (size_t)sb * 50;
    const int ib32 = tid >> 5, l = (tid >> 3) & 3, j = tid & 7;
    const float d = __half2float(*(const __half*)block);
    const unsigned char* qs = block + 2;
    const unsigned int qh = (unsigned int)block[34 + 2 * ib32] | ((unsigned int)block[35 + 2 * ib32] << 8);
    const float dl = d * (float)(2 * ((qh >> 12) & 7) + 1);
    const float delta = (qh & 0x8000u) ? -0.125f : 0.125f;
    const unsigned long long grid = iq1s_grid[qs[4 * ib32 + l] | (((qh >> (3 * l)) & 7u) << 8)];
    const float v = dl * ((float)(signed char)((grid >> (8 * j)) & 0xFFull) + delta);
    output[(size_t)sb * 256 + tid] = __float2half(v);
}

// Dequantize IQ2_S → F16 (82-byte super-block: d, 32 qs, 32 signs, 8 qh, 8 scales; see Codec_IQ2_S).
// Launch: gridDim.x = superBlockCount, blockDim.x = 256; one thread per element, the host codec's formula per element.
#define IQ_TABLE_iq2s_grid
#define IQ_TABLE_kmask_iq2xs
#include "iq_tables.cuh"
#include <cuda_fp16.h>

__device__ __forceinline__ unsigned int ld_u32(const unsigned char* p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}
#ifdef IQ_TABLE_kmask_iq2xs
__device__ __forceinline__ float sign_of(unsigned int signs, int j) { return (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f; }
#endif

extern "C" __global__ void dequant_iq2_s_to_f16(__half* __restrict__ output, const unsigned char* __restrict__ input, unsigned int superBlockCount)
{
    const unsigned int sb = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (sb >= superBlockCount || tid >= 256u) return;
    const unsigned char* block = input + (size_t)sb * 82;
    const int ib32 = tid >> 5, l = (tid >> 3) & 3, j = tid & 7;
    const float d = __half2float(*(const __half*)block);
    const unsigned char* qs = block + 2;
    const unsigned char* sg = block + 34;
    const unsigned char qh = block[66 + ib32];
    const unsigned char sc = block[74 + ib32];
    const float dl = d * (0.5f + (float)(l < 2 ? (sc & 0xF) : (sc >> 4))) * 0.25f;
    const unsigned long long grid = iq2s_grid[qs[4 * ib32 + l] | (((unsigned int)qh << (8 - 2 * l)) & 0x300u)];
    const float v = dl * (float)((grid >> (8 * j)) & 0xFFull) * sign_of(sg[4 * ib32 + l], j);
    output[(size_t)sb * 256 + tid] = __float2half(v);
}

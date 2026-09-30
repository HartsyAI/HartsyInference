// Dequantize IQ3_XXS → F16 (98-byte super-block: d, 64 qs, 32 scales-and-signs; see Codec_IQ3_XXS).
// Launch: gridDim.x = superBlockCount, blockDim.x = 256; one thread per element, the host codec's formula per element.
#define IQ_TABLE_iq3xxs_grid
#define IQ_TABLE_ksigns_iq2xs
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

extern "C" __global__ void dequant_iq3_xxs_to_f16(__half* __restrict__ output, const unsigned char* __restrict__ input, unsigned int superBlockCount)
{
    const unsigned int sb = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (sb >= superBlockCount || tid >= 256u) return;
    const unsigned char* block = input + (size_t)sb * 98;
    const int ib32 = tid >> 5, l = (tid >> 3) & 3, j = tid & 7;
    const float d = __half2float(*(const __half*)block);
    const unsigned char* q = block + 2 + 8 * ib32;
    const unsigned int aux = ld_u32(block + 66 + 4 * ib32);
    const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
    const unsigned int signs = ksigns_iq2xs[(aux >> (7 * l)) & 127u];
    const unsigned int grid = iq3xxs_grid[q[2 * l + (j >> 2)]];
    const float v = db * (float)((grid >> (8 * (j & 3))) & 0xFFu) * sign_of(signs, j);
    output[(size_t)sb * 256 + tid] = __float2half(v);
}

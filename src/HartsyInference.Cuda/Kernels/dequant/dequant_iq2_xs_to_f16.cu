// Dequantize IQ2_XS → F16 (74-byte super-block: d, 32 × uint16 qs, 8 scale bytes; see Codec_IQ2_XS).
// Launch: gridDim.x = superBlockCount, blockDim.x = 256; one thread per element, the host codec's formula per element.
#define IQ_TABLE_iq2xs_grid
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

extern "C" __global__ void dequant_iq2_xs_to_f16(__half* __restrict__ output, const unsigned char* __restrict__ input, unsigned int superBlockCount)
{
    const unsigned int sb = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (sb >= superBlockCount || tid >= 256u) return;
    const unsigned char* block = input + (size_t)sb * 74;
    const int ib32 = tid >> 5, l = (tid >> 3) & 3, j = tid & 7;
    const float d = __half2float(*(const __half*)block);
    const unsigned char sc = block[66 + ib32];
    const float dl = d * (0.5f + (float)(l < 2 ? (sc & 0xF) : (sc >> 4))) * 0.25f;
    const unsigned char* q = block + 2 + 2 * (4 * ib32 + l);
    const unsigned int word = (unsigned int)q[0] | ((unsigned int)q[1] << 8);
    const unsigned long long grid = iq2xs_grid[word & 511u];
    const unsigned int signs = ksigns_iq2xs[word >> 9];
    const float v = dl * (float)((grid >> (8 * j)) & 0xFFull) * sign_of(signs, j);
    output[(size_t)sb * 256 + tid] = __float2half(v);
}

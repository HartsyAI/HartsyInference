// Dequantize IQ3_S → F16 (110-byte super-block: d, 64 qs, 8 qh, 32 signs, 4 scales; see Codec_IQ3_S).
// Launch: gridDim.x = superBlockCount, blockDim.x = 256; one thread per element, the host codec's formula per element.
#define IQ_TABLE_iq3s_grid
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

extern "C" __global__ void dequant_iq3_s_to_f16(__half* __restrict__ output, const unsigned char* __restrict__ input, unsigned int superBlockCount)
{
    const unsigned int sb = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (sb >= superBlockCount || tid >= 256u) return;
    const unsigned char* block = input + (size_t)sb * 110;
    const int ib32 = tid >> 5, l = (tid >> 3) & 3, j = tid & 7;
    const float d = __half2float(*(const __half*)block);
    const unsigned char* q = block + 2 + 8 * ib32;
    const unsigned int qh = block[66 + ib32];
    const unsigned char sg = block[74 + 4 * ib32 + l];
    const unsigned char scb = block[106 + (ib32 >> 1)];
    const float db = d * (float)(1 + 2 * ((ib32 & 1) ? (scb >> 4) : (scb & 0xF)));
    const unsigned int idx = (j < 4) ? (q[2 * l] | ((qh << (8 - 2 * l)) & 256u)) : (q[2 * l + 1] | ((qh << (7 - 2 * l)) & 256u));
    const unsigned int grid = iq3s_grid[idx];
    const float v = db * (float)((grid >> (8 * (j & 3))) & 0xFFu) * sign_of(sg, j);
    output[(size_t)sb * 256 + tid] = __float2half(v);
}

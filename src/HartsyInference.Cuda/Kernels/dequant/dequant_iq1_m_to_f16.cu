// Dequantize IQ1_M → F16 (56-byte super-block: 32 qs, 16 qh, 8 scale bytes carrying the FP16 super-scale; see Codec_IQ1_M).
// Launch: gridDim.x = superBlockCount, blockDim.x = 256; one thread per element, the host codec's formula per element.
#define IQ_TABLE_iq1s_grid
#include "iq_tables.cuh"
#include <cuda_fp16.h>

extern "C" __global__ void dequant_iq1_m_to_f16(__half* __restrict__ output, const unsigned char* __restrict__ input, unsigned int superBlockCount)
{
    const unsigned int sb = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (sb >= superBlockCount || tid >= 256u) return;
    const unsigned char* block = input + (size_t)sb * 56;
    const int ib32 = tid >> 5, l = (tid >> 3) & 3, j = tid & 7;
    const unsigned char* qs = block;
    const unsigned char* qh = block + 32;
    const unsigned char* scb = block + 48;
    const unsigned int sc0 = scb[0] | (scb[1] << 8), sc1 = scb[2] | (scb[3] << 8), sc2 = scb[4] | (scb[5] << 8), sc3 = scb[6] | (scb[7] << 8);
    const unsigned short dBits = (unsigned short)((sc0 >> 12) | ((sc1 >> 8) & 0x00f0u) | ((sc2 >> 4) & 0x0f00u) | (sc3 & 0xf000u));
    const float d = __half2float(__ushort_as_half(dBits));
    const unsigned int sc = (ib32 >> 1) == 0 ? sc0 : (ib32 >> 1) == 1 ? sc1 : (ib32 >> 1) == 2 ? sc2 : sc3;
    const int shift = 6 * (ib32 & 1) + (l < 2 ? 0 : 3);
    const float dl = d * (float)(2 * ((sc >> shift) & 7) + 1);
    const unsigned int hb = qh[2 * ib32 + (l >> 1)];
    const unsigned int idx = qs[4 * ib32 + l] | (((l & 1) == 0 ? (hb << 8) : (hb << 4)) & 0x700u);
    const float delta = (hb & ((l & 1) == 0 ? 0x08u : 0x80u)) ? -0.125f : 0.125f;
    const unsigned long long grid = iq1s_grid[idx];
    const float v = dl * ((float)(signed char)((grid >> (8 * j)) & 0xFFull) + delta);
    output[(size_t)sb * 256 + tid] = __float2half(v);
}

// Dequantize Q4_K → F16 on GPU.
//
// Q4_K layout (256 elements per super-block, 144 bytes):
//   [2 bytes FP16 d (super-block scale)]
//   [2 bytes FP16 dmin (super-block min)]
//   [12 bytes packed 6-bit scales+mins for 8 sub-blocks of 32 elements each]
//   [128 bytes 4-bit quants]
//
// Reconstruction: x[i] = d * sc_j * q[i] - dmin * m_j
//   where (sc_j, m_j) come from canonical ggml `get_scale_min_k4`.
//
// Launch: gridDim.x = numSuperBlocks, blockDim.x = 256.
// Each CUDA block handles one super-block; each thread handles one element.

#include <cuda_fp16.h>

#define SUPER_ELEMS 256
#define SUPER_BYTES 144
#define SUB_ELEMS 32

// Canonical ggml get_scale_min_k4 — extracts a 6-bit scale and 6-bit min for sub-block j (0..7).
__device__ __forceinline__ void get_scale_min_k4(
    int j, const unsigned char* q, unsigned char* d, unsigned char* m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

extern "C" __global__ void dequant_q4_k_to_f16(
    __half* __restrict__ output,
    const unsigned char* __restrict__ input,
    unsigned int superBlockCount)
{
    const unsigned int sb = blockIdx.x;
    if (sb >= superBlockCount) return;

    const unsigned char* block = input + (size_t)sb * SUPER_BYTES;
    const __half dHalf = *(const __half*)block;
    const __half dminHalf = *(const __half*)(block + 2);
    const float d = __half2float(dHalf);
    const float dmin = __half2float(dminHalf);
    const unsigned char* scales = block + 4;
    const unsigned char* qs = block + 16;

    const unsigned int tid = threadIdx.x;
    if (tid < SUPER_ELEMS) {
        const int j = tid / SUB_ELEMS;            // sub-block index 0..7
        const int i = tid % SUB_ELEMS;            // element within sub-block 0..31

        unsigned char sc, mm;
        get_scale_min_k4(j, scales, &sc, &mm);
        const float subScale = d * (float)sc;
        const float subMin = dmin * (float)mm;

        const unsigned char* subQuants = qs + (j / 2) * SUB_ELEMS;
        const int nibbleShift = (j % 2 == 0) ? 0 : 4;
        const int q = (subQuants[i] >> nibbleShift) & 0x0F;

        const float v = subScale * (float)q - subMin;
        output[(size_t)sb * SUPER_ELEMS + tid] = __float2half(v);
    }
}

// Dequantize Q5_K → F16 on GPU.
//
// Q5_K layout (256 elements per super-block, 176 bytes):
//   [2 bytes FP16 d][2 bytes FP16 dmin]
//   [12 bytes packed 6-bit scales+mins]
//   [32 bytes high bits (1 per element)]
//   [128 bytes low nibbles (2 per byte, same packing as Q4_K)]
//
// Reconstruction: x = d * sc_j * q - dmin * m_j  where q = low | (high << 4) ∈ [0..31].
//
// Launch: gridDim.x = numSuperBlocks, blockDim.x = 256.

#include <cuda_fp16.h>

#define SUPER_ELEMS 256
#define SUPER_BYTES 176
#define SUB_ELEMS 32

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

extern "C" __global__ void dequant_q5_k_to_f16(
    __half* __restrict__ output,
    const unsigned char* __restrict__ input,
    unsigned int superBlockCount)
{
    const unsigned int sb = blockIdx.x;
    if (sb >= superBlockCount) return;

    const unsigned char* block = input + (size_t)sb * SUPER_BYTES;
    const float d = __half2float(*(const __half*)block);
    const float dmin = __half2float(*(const __half*)(block + 2));
    const unsigned char* scales = block + 4;
    const unsigned char* highBits = block + 16;
    const unsigned char* lowBits = block + 48;

    const unsigned int tid = threadIdx.x;
    if (tid < SUPER_ELEMS) {
        const int j = tid / SUB_ELEMS;
        const int i = tid % SUB_ELEMS;

        unsigned char sc, mm;
        get_scale_min_k4(j, scales, &sc, &mm);
        const float subScale = d * (float)sc;
        const float subMin = dmin * (float)mm;

        const unsigned char* subLowBits = lowBits + (j / 2) * SUB_ELEMS;
        const int lowShift = (j % 2 == 0) ? 0 : 4;
        const int low = (subLowBits[i] >> lowShift) & 0x0F;
        const int high = ((highBits[i] >> j) & 0x01) << 4;
        const int q = low | high;

        const float v = subScale * (float)q - subMin;
        output[(size_t)sb * SUPER_ELEMS + tid] = __float2half(v);
    }
}

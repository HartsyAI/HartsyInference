// Fused Q5_K × Q8_1 matrix-vector product for LLM decode, using __dp4a int8 dot products.
//
// Same structure as mul_mat_vec_q4k_q8_1 (per-sub-block 6-bit scale sc + min m, so the Q8_1
// int-sum xs carries the min term), extended with Q5_K's high-bit plane:
//
//   dot_j = xscale_j * ( d*sc * Σ q[i]·xq[i]  -  dmin*m * xsum_j ),  q = nibble | (highbit << 4)
//
// Q5_K super-block: 256 elements / 176 bytes:
//   [2B d][2B dmin][12B packed 6-bit scales+mins][32B high bits][128B low nibbles]
// Bit j of high-bit byte i = high bit of element i in sub-block j (matches mul_mat_vec_q5k_f32 /
// dequant_q5_k_to_f16 — ground truth for the layout). High bits are injected whole-word:
// ((hword >> j) & 0x01010101) << 4 lands each byte's plane-j bit at that byte's bit 4.
//
// Layout: one WARP per output row (8 rows/block); each lane owns 8 elements of sub-block
// j = lane>>2 (2 dp4a / lane / super-block). All loads 4-aligned (176-byte stride).
//
// weight [N,K] Q5_K row-major; xq [M,K] int8; xd/xs [M,K/32] F32.
// Launch: blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N/WARPS_PER_BLOCK), M). Requires K % 256 == 0.

#include <cuda_fp16.h>

#define SUPER_ELEMS 256
#define SUPER_BYTES 176
#define SUB_ELEMS 32
#define WARP_SIZE 32

// Word-based get_scale_min_k4 (see mul_mat_vec_q4k_q8_1.cu).
__device__ __forceinline__ void get_scale_min_k4_words(
    int j, unsigned int s0, unsigned int s1, unsigned int s2, unsigned int* d, unsigned int* m)
{
    if (j < 4) {
        *d = (s0 >> (8 * j)) & 63u;
        *m = (s1 >> (8 * j)) & 63u;
    } else {
        const int jj = 8 * (j - 4);
        *d = ((s2 >> jj) & 0x0Fu) | (((s0 >> (jj + 6)) & 3u) << 4);
        *m = (((s2 >> jj) >> 4) & 0x0Fu) | (((s1 >> (jj + 6)) & 3u) << 4);
    }
}

extern "C" __global__ void mul_mat_vec_q5k_q8_1(
    float* __restrict__ output,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ bias,     // may be nullptr
    int N, int K, int M)
{
    const int lane = threadIdx.x;                          // 0..31
    const int n = blockIdx.x * blockDim.y + threadIdx.y;   // output row
    const int m = blockIdx.y;                              // batch row
    if (n >= N || m >= M) return;

    const int nsb = K / SUPER_ELEMS;
    const int kblocks = K / SUB_ELEMS;
    const unsigned char* wrow = weight + (size_t)n * nsb * SUPER_BYTES;
    const signed char* xqrow = xq + (size_t)m * K;
    const float* xdrow = xd + (size_t)m * kblocks;
    const float* xsrow = xs + (size_t)m * kblocks;

    const int j = lane >> 2;               // sub-block 0..7
    const int base_i = (lane & 3) << 3;    // 0,8,16,24
    const int nibbleShift = (j & 1) ? 4 : 0;
    const int subByteBase = (j >> 1) * SUB_ELEMS + base_i;
    const int xElemBase = j * SUB_ELEMS + base_i;
    const bool minLane = (lane & 3) == 0;  // one lane per sub-block adds the min term

    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const unsigned char* block = wrow + (size_t)sb * SUPER_BYTES;
        const unsigned int ddmin = *(const unsigned int*)block;    // fp16 d | fp16 dmin, one load
        const float d = __half2float(__ushort_as_half((unsigned short)(ddmin & 0xFFFFu)));
        const float dmin = __half2float(__ushort_as_half((unsigned short)(ddmin >> 16)));
        const unsigned int* sc32 = (const unsigned int*)(block + 4);
        const unsigned char* highBits = block + 16;
        const unsigned char* qs = block + 48;

        unsigned int sc, mm;
        get_scale_min_k4_words(j, sc32[0], sc32[1], sc32[2], &sc, &mm);
        const float subScale = d * (float)sc;
        const float subMin = dmin * (float)mm;

        const int2 xqp = *reinterpret_cast<const int2*>(xqrow + sb * SUPER_ELEMS + xElemBase);
        const uint2 qpack = *reinterpret_cast<const uint2*>(qs + subByteBase);
        const uint2 hpack = *reinterpret_cast<const uint2*>(highBits + base_i);

        const int wq0 = (int)(((qpack.x >> nibbleShift) & 0x0F0F0F0Fu)
                            | (((hpack.x >> j) & 0x01010101u) << 4));
        const int wq1 = (int)(((qpack.y >> nibbleShift) & 0x0F0F0F0Fu)
                            | (((hpack.y >> j) & 0x01010101u) << 4));

        int idot = __dp4a(wq0, xqp.x, 0);
        idot = __dp4a(wq1, xqp.y, idot);

        const int subBlock = sb * 8 + j;
        const float xscale = xdrow[subBlock];
        acc += xscale * subScale * (float)idot;
        if (minLane) acc -= xscale * subMin * xsrow[subBlock];
    }

    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        if (bias != nullptr) acc += bias[n];
        output[(size_t)m * N + n] = acc;
    }
}

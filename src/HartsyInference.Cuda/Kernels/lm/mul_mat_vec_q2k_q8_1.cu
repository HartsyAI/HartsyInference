// Fused Q2_K × Q8_1 matrix-vector product for LLM decode with __dp4a int8 dot products — the int8-activation tier
// of mul_mat_vec_q2k_f32.cu, in the shape of mul_mat_vec_q4k_q8_1.cu (see there for the scheme and the k-split).
//
// Per 16-element run r (4-bit scale sc, 4-bit min mn; super-scale d, super-min dmin):
//   w[i] = d*sc*q[i] - dmin*mn ;  x[i] ≈ xd_j * xq[i]   (j = the 32-block holding the run)
//   dot  = xd_j * ( d*sc * Σ q[i]·xq[i]  -  dmin*mn * Σ xq[i] )
// The Σ xq over a lane's four elements is dp4a(0x01010101, xq), so no 16-element activation sum is stored.
//
// Lane mapping: a warp covers TWO super-blocks per iteration — lanes 0..15 the even one, 16..31 the odd one.
// Within a super-block a lane owns one 4-byte word of qs (w = lane & 7) in one half (h = lane >> 3 & 1): the
// four elements 128h + 32j + 4w .. +3 at each shift j (0..3), i.e. 16 elements across the half's four runs.
// The 84-byte super-block keeps every word 4-aligned. weight [N,K] Q2_K row-major; xq [M,K] int8; xd [M,K/32].
// Launch (standard): blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N/WARPS_PER_BLOCK), M).
// Launch (ksplit):   blockDim = (32, KSPLIT_WARPS);    grid = (N, M).

#include <cuda_fp16.h>

#define SUPER_ELEMS 256
#define SUPER_BYTES 84
#define SUB_ELEMS 32
#define WARP_SIZE 32

__device__ __forceinline__ float q2k_q8_1_row_partial(
    const unsigned char* __restrict__ wrow,
    const signed char* __restrict__ xqrow,
    const float* __restrict__ xdrow,
    int nsb, int pairStart, int pairStride, int lane)
{
    const int sbOff = lane >> 4;
    const int h = (lane >> 3) & 1;
    const int w = lane & 7;
    const int runLow = (w >> 2);                 // run within a (h, j) pair of 16-element runs

    float acc = 0.0f;
    for (int sb = 2 * pairStart + sbOff; sb < nsb; sb += 2 * pairStride) {
        const unsigned char* block = wrow + (size_t)sb * SUPER_BYTES;
        const unsigned int* scw = (const unsigned int*)block;                    // scales[16] as 4 words
        const unsigned int v = *(const unsigned int*)(block + 16 + 32 * h + 4 * w);
        const unsigned int ddmin = *(const unsigned int*)(block + 80);
        const float d = __half2float(__ushort_as_half((unsigned short)(ddmin & 0xFFFFu)));
        const float dmin = __half2float(__ushort_as_half((unsigned short)(ddmin >> 16)));
        const unsigned int s0 = scw[2 * h], s1 = scw[2 * h + 1];               // runs 8h..8h+3, 8h+4..8h+7

        const signed char* xb = xqrow + sb * SUPER_ELEMS + 128 * h + 4 * w;
        const float* xdb = xdrow + sb * 8 + 4 * h;
        float accD = 0.0f, accM = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u = *(const int*)(xb + 32 * j);
            const int vi = (int)((v >> (2 * j)) & 0x03030303u);
            const unsigned int sw = (j < 2) ? s0 : s1;
            const unsigned int sc = (sw >> (8 * ((2 * (j & 1)) + runLow))) & 0xFFu;
            const float xscale = xdb[j];
            accD += xscale * (float)(sc & 0xFu) * (float)__dp4a(vi, u, 0);
            accM += xscale * (float)(sc >> 4) * (float)__dp4a(0x01010101, u, 0);
        }
        acc += d * accD - dmin * accM;
    }
    return acc;
}

extern "C" __global__ void mul_mat_vec_q2k_q8_1(
    float* __restrict__ output,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ bias,     // may be nullptr
    int N, int K, int M)
{
    const int lane = threadIdx.x;
    const int n = blockIdx.x * blockDim.y + threadIdx.y;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int nsb = K / SUPER_ELEMS;
    const int kblocks = K / SUB_ELEMS;
    float acc = q2k_q8_1_row_partial(
        weight + (size_t)n * nsb * SUPER_BYTES, xq + (size_t)m * K, xd + (size_t)m * kblocks, nsb, 0, 1, lane);

    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        if (bias != nullptr) acc += bias[n];
        output[(size_t)m * N + n] = acc;
    }
}

extern "C" __global__ void mul_mat_vec_q2k_q8_1_ksplit(
    float* __restrict__ output,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ bias,     // may be nullptr
    int N, int K, int M)
{
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;      // all warps on the SAME row
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int nsb = K / SUPER_ELEMS;
    const int kblocks = K / SUB_ELEMS;
    float acc = q2k_q8_1_row_partial(
        weight + (size_t)n * nsb * SUPER_BYTES, xq + (size_t)m * K, xd + (size_t)m * kblocks, nsb, warp, blockDim.y, lane);

    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);

    __shared__ float partial[16];
    if (lane == 0) partial[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float sum = 0.0f;
        for (int i = 0; i < (int)blockDim.y; ++i) sum += partial[i];
        if (bias != nullptr) sum += bias[n];
        output[(size_t)m * N + n] = sum;
    }
}

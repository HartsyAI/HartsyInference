// Fused Q3_K × Q8_1 matrix-vector product for LLM decode with __dp4a int8 dot products — the int8-activation tier
// of mul_mat_vec_q3k_f32.cu, in the shape of mul_mat_vec_q4k_q8_1.cu (see there for the scheme and the k-split).
//
// Per 16-element run r (6-bit signed scale sc; super-scale d):  w[i] = d*sc*(q[i] - (hbit ? 0 : 4)), so
//   dot = xd_j * d*sc * Σ (q[i] - 4·!hbit)·xq[i]   with the signed 3-bit value formed per byte by __vsubss4.
//
// Lane mapping as in mul_mat_vec_q2k_q8_1.cu: two super-blocks per warp iteration, a lane owns qs word w of half h.
// The hmask word for a lane's four elements is bytes 4w..4w+3 for every (h, j); bit 4h + j is the INVERTED high bit.
// A 110-byte super-block is only 2-aligned, so words are assembled from 16-bit loads.
// weight [N,K] Q3_K row-major; xq [M,K] int8; xd [M,K/32].
// Launch (standard): blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N/WARPS_PER_BLOCK), M).
// Launch (ksplit):   blockDim = (32, KSPLIT_WARPS);    grid = (N, M).

#include <cuda_fp16.h>

#define SUPER_ELEMS 256
#define SUPER_BYTES 110
#define SUB_ELEMS 32
#define WARP_SIZE 32

__device__ __forceinline__ unsigned int ld_u32_2aligned(const unsigned char* p)
{
    return (unsigned int)(*(const unsigned short*)p) | ((unsigned int)(*(const unsigned short*)(p + 2)) << 16);
}

// Canonical ggml 6-bit signed scale unpack (dequantize_row_q3_K) on the 12 scale bytes held as three words.
__device__ __forceinline__ int q3k_scale(unsigned int s0, unsigned int s1, unsigned int s2, int r)
{
    const unsigned int lowByte = r < 8 ? ((r < 4 ? s0 : s1) >> (8 * (r & 3))) & 0xFFu
                                       : ((r < 12 ? s0 : s1) >> (8 * (r & 3))) & 0xFFu;
    const unsigned int low = r < 8 ? (lowByte & 0x0Fu) : (lowByte >> 4);
    const unsigned int high = ((s2 >> (8 * (r & 3))) >> (2 * (r >> 2))) & 0x03u;
    return (int)(low | (high << 4)) - 32;
}

__device__ __forceinline__ float q3k_q8_1_row_partial(
    const unsigned char* __restrict__ wrow,
    const signed char* __restrict__ xqrow,
    const float* __restrict__ xdrow,
    int nsb, int pairStart, int pairStride, int lane)
{
    const int sbOff = lane >> 4;
    const int h = (lane >> 3) & 1;
    const int w = lane & 7;
    const int runLow = (w >> 2);

    float acc = 0.0f;
    for (int sb = 2 * pairStart + sbOff; sb < nsb; sb += 2 * pairStride) {
        const unsigned char* block = wrow + (size_t)sb * SUPER_BYTES;
        const unsigned int hm = ld_u32_2aligned(block + 4 * w);
        const unsigned int v = ld_u32_2aligned(block + 32 + 32 * h + 4 * w);
        const unsigned int s0 = ld_u32_2aligned(block + 96);
        const unsigned int s1 = ld_u32_2aligned(block + 100);
        const unsigned int s2 = ld_u32_2aligned(block + 104);
        const float d = __half2float(*(const __half*)(block + 108));

        const signed char* xb = xqrow + sb * SUPER_ELEMS + 128 * h + 4 * w;
        const float* xdb = xdrow + sb * 8 + 4 * h;
        float accD = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u = *(const int*)(xb + 32 * j);
            const int vil = (int)((v >> (2 * j)) & 0x03030303u);
            const int vih = (int)(((~hm >> (4 * h + j)) & 0x01010101u) << 2);   // 4 where the high bit is clear
            const int vi = __vsubss4(vil, vih);
            const int sc = q3k_scale(s0, s1, s2, 8 * h + 2 * j + runLow);
            accD += xdb[j] * (float)sc * (float)__dp4a(vi, u, 0);
        }
        acc += d * accD;
    }
    return acc;
}

extern "C" __global__ void mul_mat_vec_q3k_q8_1(
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
    float acc = q3k_q8_1_row_partial(
        weight + (size_t)n * nsb * SUPER_BYTES, xq + (size_t)m * K, xd + (size_t)m * kblocks, nsb, 0, 1, lane);

    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        if (bias != nullptr) acc += bias[n];
        output[(size_t)m * N + n] = acc;
    }
}

extern "C" __global__ void mul_mat_vec_q3k_q8_1_ksplit(
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
    float acc = q3k_q8_1_row_partial(
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

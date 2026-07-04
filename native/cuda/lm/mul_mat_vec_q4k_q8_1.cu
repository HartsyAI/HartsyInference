// Fused Q4_K × Q8_1 matrix-vector product for LLM decode, using __dp4a int8 dot products.
//
// This is the llama.cpp mul_mat_vec_q approach: the activation is pre-quantized to int8 (Q8_1,
// per-32-block scale + int-sum), the weight stays Q4_K, and each 32-lane group dots 4-bit weights
// against int8 activations with __dp4a (4 int8 MACs / instruction) — far less ALU per byte than the
// per-element float dequant, so the GEMV approaches memory bandwidth.
//
// Per sub-block j (32 elems, own 6-bit scale sc + min m, super-scale d + super-min dmin):
//   w[i] = d*sc*q[i] - dmin*m ;  x[i] ≈ xscale_j * xq[i]
//   dot_j = xscale_j * ( d*sc * Σ q[i]·xq[i]  -  dmin*m * Σ xq[i] )
//         = xscale_j * ( subScale * dp4a(q,xq) - subMin * xsum_j )
//
// Layout: one WARP per output row (32 lanes), 8 rows/block. Each lane owns 8 elements of a sub-block
// (4 lanes cover one 32-elem sub-block); it accumulates its dp4a term, and the lane at base_i==0 adds
// the sub-block's single min term. A warp-shuffle reduction sums the row.
//
// weight [N,K] Q4_K row-major; xq [M,K] int8; xd/xs [M,K/32] F32.
// Launch: blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N/WARPS_PER_BLOCK), M).

#include <cuda_fp16.h>

#define SUPER_ELEMS 256
#define SUPER_BYTES 144
#define SUB_ELEMS 32
#define WARP_SIZE 32

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

extern "C" __global__ void mul_mat_vec_q4k_q8_1(
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
    const int kblocks = K / SUB_ELEMS;                     // 32-blocks per row
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
        const float d = __half2float(*(const __half*)block);
        const float dmin = __half2float(*(const __half*)(block + 2));
        const unsigned char* scales = block + 4;
        const unsigned char* qs = block + 16;

        unsigned char sc, mm;
        get_scale_min_k4(j, scales, &sc, &mm);
        const float subScale = d * (float)sc;
        const float subMin = dmin * (float)mm;

        // 8 int8 activations (2× int32) + 8 quant bytes (uint2).
        const int2 xqp = *reinterpret_cast<const int2*>(xqrow + sb * SUPER_ELEMS + xElemBase);
        const uint2 qpack = *reinterpret_cast<const uint2*>(qs + subByteBase);
        const unsigned int lo = qpack.x, hi = qpack.y;

        // Pack this lane's 8 weight nibbles into two int32 as four int8 lanes each.
        const int wq0 = (int)(((lo      ) >> nibbleShift) & 0xF)
                      | (int)(((lo >>  8) >> nibbleShift) & 0xF) << 8
                      | (int)(((lo >> 16) >> nibbleShift) & 0xF) << 16
                      | (int)(((lo >> 24) >> nibbleShift) & 0xF) << 24;
        const int wq1 = (int)(((hi      ) >> nibbleShift) & 0xF)
                      | (int)(((hi >>  8) >> nibbleShift) & 0xF) << 8
                      | (int)(((hi >> 16) >> nibbleShift) & 0xF) << 16
                      | (int)(((hi >> 24) >> nibbleShift) & 0xF) << 24;

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

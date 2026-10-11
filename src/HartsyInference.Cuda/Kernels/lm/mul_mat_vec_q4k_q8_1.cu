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
// Two entry points share the per-lane body:
//  - mul_mat_vec_q4k_q8_1: one WARP per output row. Each lane owns 32 elements (16 qs bytes, both
//    nibble planes) of a super-block, 8 lanes per super-block; a warp-shuffle reduction sums the row.
//    Weight and xq pointers must be 16-byte aligned (device allocations are 256-aligned).
//  - mul_mat_vec_q4k_q8_1_ksplit: one BLOCK per output row; the warps split the row's super-blocks
//    (stride blockDim.y) and combine through shared memory (deterministic order). Dispatched ONLY for
//    long-K/small-N shapes (ffn_down class, e.g. N=1536 → 1.1 waves of warps at warp-per-row, measured
//    51% of DRAM peak on DeepSeek-1.5B's 8960×1536) — at N ≥ ~2560 warp-per-row measured equal-or-
//    better (the 2026-07-22 round-1 sweep), so the host gates the split tightly.
//
// weight [N,K] Q4_K row-major; xq [M,K] int8; xd/xs [M,K/32] F32.
// Launch (standard): blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N/WARPS_PER_BLOCK), M).
// Launch (ksplit):   blockDim = (32, KSPLIT_WARPS);    grid = (N, M).

#include <cuda_fp16.h>

#define SUPER_ELEMS 256
#define SUPER_BYTES 144
#define SUB_ELEMS 32
#define WARP_SIZE 32

// Word-based reformulation of ggml's get_scale_min_k4: the 12 packed scale/min bytes are read as
// three aligned u32s (the 144-byte super-block keeps block+4 4-aligned) and the 6-bit fields are
// extracted with shifts — no per-byte global loads on the inner loop's critical path.
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

// Per-lane partial dot over one output row. Eight lanes share a super-block and a warp covers four
// super-blocks per iteration (576 contiguous bytes): lane t of its group owns qs bytes [16t, 16t+16) —
// one 16-byte load — and processes both nibble planes of them (sub-blocks 2p and 2p+1, p = t/2), so
// every weight byte is requested exactly once. The 16-byte header (d, dmin, 12 scale bytes) is one
// more 16-byte load; 144 is a multiple of 16, so a 16-aligned row keeps every block aligned.
// sbStart/sbStride count warp-iterations (the ksplit entry passes warp, warps): group g of warp w
// handles super-blocks 4*sbStart + g, stepping 4*sbStride.
// (The 2026-07-22 byte-shared variant used 8-byte loads and two super-blocks per iteration; this one
// measured +60% on the Mixtral ffn_down shape, 2026-10-10.)
__device__ __forceinline__ float q4k_q8_1_row_partial(
    const unsigned char* __restrict__ wrow,
    const signed char* __restrict__ xqrow,
    const float* __restrict__ xdrow,
    const float* __restrict__ xsrow,
    int nsb, int sbStart, int sbStride, int lane)
{
    const int t = lane & 7;                // position in the super-block group
    const int g = lane >> 3;               // super-block within the warp-iteration
    const int pr = t >> 1;                 // qs pair: sub-blocks 2pr (lo nibble) and 2pr+1 (hi nibble)
    const int hh = t & 1;                  // which 16 of the pair's 32 bytes
    const int qsOff = 16 + pr * 32 + hh * 16;
    const int xOff0 = pr * 64 + hh * 16;   // lo-nibble elements; the hi-nibble ones are 32 further
    const bool minLane = hh == 0;          // one lane per sub-block pair adds the min terms

    float acc = 0.0f;
    #pragma unroll 2
    for (int sb = sbStart * 4 + g; sb < nsb; sb += sbStride * 4) {
        const unsigned char* block = wrow + (size_t)sb * SUPER_BYTES;
        const uint4 hdr = *reinterpret_cast<const uint4*>(block);
        const uint4 q = *reinterpret_cast<const uint4*>(block + qsOff);
        const signed char* xb = xqrow + sb * SUPER_ELEMS + xOff0;
        const int4 xlo = *reinterpret_cast<const int4*>(xb);
        const int4 xhi = *reinterpret_cast<const int4*>(xb + 32);
        const int subBlock = sb * 8 + 2 * pr;
        const float xd0 = xdrow[subBlock], xd1 = xdrow[subBlock + 1];

        const float d = __half2float(__ushort_as_half((unsigned short)(hdr.x & 0xFFFFu)));
        const float dmin = __half2float(__ushort_as_half((unsigned short)(hdr.x >> 16)));
        unsigned int sc0, m0, sc1, m1;
        get_scale_min_k4_words(2 * pr, hdr.y, hdr.z, hdr.w, &sc0, &m0);
        get_scale_min_k4_words(2 * pr + 1, hdr.y, hdr.z, hdr.w, &sc1, &m1);

        int lo = __dp4a((int)(q.x & 0x0F0F0F0Fu), xlo.x, 0);
        lo = __dp4a((int)(q.y & 0x0F0F0F0Fu), xlo.y, lo);
        lo = __dp4a((int)(q.z & 0x0F0F0F0Fu), xlo.z, lo);
        lo = __dp4a((int)(q.w & 0x0F0F0F0Fu), xlo.w, lo);
        int hi = __dp4a((int)((q.x >> 4) & 0x0F0F0F0Fu), xhi.x, 0);
        hi = __dp4a((int)((q.y >> 4) & 0x0F0F0F0Fu), xhi.y, hi);
        hi = __dp4a((int)((q.z >> 4) & 0x0F0F0F0Fu), xhi.z, hi);
        hi = __dp4a((int)((q.w >> 4) & 0x0F0F0F0Fu), xhi.w, hi);

        acc += xd0 * (d * (float)sc0) * (float)lo + xd1 * (d * (float)sc1) * (float)hi;
        if (minLane) acc -= dmin * (xd0 * (float)m0 * xsrow[subBlock] + xd1 * (float)m1 * xsrow[subBlock + 1]);
    }
    return acc;
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
    const int kblocks = K / SUB_ELEMS;
    float acc = q4k_q8_1_row_partial(
        weight + (size_t)n * nsb * SUPER_BYTES,
        xq + (size_t)m * K, xd + (size_t)m * kblocks, xs + (size_t)m * kblocks,
        nsb, 0, 1, lane);

    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        if (bias != nullptr) acc += bias[n];
        output[(size_t)m * N + n] = acc;
    }
}

extern "C" __global__ void mul_mat_vec_q4k_q8_1_ksplit(
    float* __restrict__ output,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ bias,     // may be nullptr
    int N, int K, int M)
{
    const int lane = threadIdx.x;      // 0..31
    const int warp = threadIdx.y;      // all warps on the SAME row
    const int n = blockIdx.x;          // output row
    const int m = blockIdx.y;          // batch row
    if (n >= N || m >= M) return;

    const int nsb = K / SUPER_ELEMS;
    const int kblocks = K / SUB_ELEMS;
    float acc = q4k_q8_1_row_partial(
        weight + (size_t)n * nsb * SUPER_BYTES,
        xq + (size_t)m * K, xd + (size_t)m * kblocks, xs + (size_t)m * kblocks,
        nsb, warp, blockDim.y, lane);

    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);

    __shared__ float partial[16];
    if (lane == 0) partial[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float sum = 0.0f;
        for (int w = 0; w < (int)blockDim.y; ++w) sum += partial[w];
        if (bias != nullptr) sum += bias[n];
        output[(size_t)m * N + n] = sum;
    }
}

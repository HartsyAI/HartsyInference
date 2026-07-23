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
// Layout: one WARP per output row (8 rows/block). Each lane owns 8 elements of a sub-block
// (4 lanes cover one 32-elem sub-block); a warp-shuffle reduction sums the row.
// (A block-per-row K-split variant — llama.cpp mmvq's occupancy shape — was built and measured
// 2026-07-22: net loss at every real production shape once the whole-word nibble unpack landed,
// including the long-K/small-N ffn_down it targeted. Removed per revert-on-no-gain.)
//
// weight [N,K] Q4_K row-major; xq [M,K] int8; xd/xs [M,K/32] F32.
// Launch: blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N/WARPS_PER_BLOCK), M).

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

// Per-lane partial dot over one output row. Each lane owns 8 elements of one sub-block per
// super-block (plane-per-lane: lanes with even/odd j read the same qs bytes for the lo/hi nibble —
// L1 absorbs the second read). A byte-shared variant (each lane processing both nibble planes of
// its uint2, 2 super-blocks/warp-iteration) was built and measured 2026-07-22: ~10% SLOWER on the
// long-K ffn_down shape, flat elsewhere — reverted to this form.
__device__ __forceinline__ float q4k_q8_1_row_partial(
    const unsigned char* __restrict__ wrow,
    const signed char* __restrict__ xqrow,
    const float* __restrict__ xdrow,
    const float* __restrict__ xsrow,
    int nsb, int lane)
{
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
        const unsigned char* qs = block + 16;

        unsigned int sc, mm;
        get_scale_min_k4_words(j, sc32[0], sc32[1], sc32[2], &sc, &mm);
        const float subScale = d * (float)sc;
        const float subMin = dmin * (float)mm;

        // 8 int8 activations (2× int32) + 8 quant bytes (uint2).
        const int2 xqp = *reinterpret_cast<const int2*>(xqrow + sb * SUPER_ELEMS + xElemBase);
        const uint2 qpack = *reinterpret_cast<const uint2*>(qs + subByteBase);

        // Whole-word nibble extraction: shifting the packed word by 0/4 then masking 0x0F0F0F0F
        // yields all four byte-lanes' selected nibbles in one op pair.
        const int wq0 = (int)((qpack.x >> nibbleShift) & 0x0F0F0F0Fu);
        const int wq1 = (int)((qpack.y >> nibbleShift) & 0x0F0F0F0Fu);

        int idot = __dp4a(wq0, xqp.x, 0);
        idot = __dp4a(wq1, xqp.y, idot);

        const int subBlock = sb * 8 + j;
        const float xscale = xdrow[subBlock];
        acc += xscale * subScale * (float)idot;
        if (minLane) acc -= xscale * subMin * xsrow[subBlock];
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
        nsb, lane);

    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        if (bias != nullptr) acc += bias[n];
        output[(size_t)m * N + n] = acc;
    }
}

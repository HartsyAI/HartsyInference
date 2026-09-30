// Single-latent sparse attention and the lightning-indexer scores of DeepSeek-V4.1-Flash.
//
// sparse_latent_attention_f32: one block per (token, head). One latent row is both key and value. Index i < windowSlots
// addresses the window ring, larger indices the main cache (row i - windowSlots); -1 and anything outside both
// sources is skipped. The softmax runs over the valid scaled dot products plus the head's fp32 sink, which joins the
// denominator only (the running maximum includes it for stability). A token with no valid index yields zeros.
//   phase 1  a warp per index: lanes stride the dim, dequantize in place, warp-reduce the dot into shared scores
//   phase 2  block max, exp, sum in a fixed order (deterministic)
//   phase 3  a thread per output element: sum_j p[j] * row[j][d], j ascending, then divide by the denominator
// Launch: grid = (tokens, heads), block = 128, dynamic shared = 4 * k bytes (k <= 12288).
//
// indexer_scores_f32: one warp per (token, key). The key row is dequantized once into registers, then every head's
// dot is a warp reduction; scores[t,n] = scale * sum_h relu(dot) * w[t,h], heads in order. Entries at n >= compressLens[t]
// (which hides the still-open compression group from mid-group queries) or with a zero candidate flag are -inf.
// Launch: grid = (ceil(keys / 8), tokens), block = 256, dim <= 512.

#include "latent_codec.cuh"

#define ATT_THREADS 128
#define IDX_WARPS 8
#define IDX_MAX_PER_LANE 16

__device__ __forceinline__ float att_block_max(float v, float* scratch)
{
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = scratch[0];
    for (int w = 1; w < ATT_THREADS / 32; w++) r = fmaxf(r, scratch[w]);
    __syncthreads();
    return r;
}

__device__ __forceinline__ float att_block_sum(float v, float* scratch)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = scratch[0];
    for (int w = 1; w < ATT_THREADS / 32; w++) r += scratch[w];
    __syncthreads();
    return r;
}

extern "C" __global__ void __launch_bounds__(ATT_THREADS) sparse_latent_attention_f32(
    float* __restrict__ output,
    const float* __restrict__ query,
    const unsigned char* __restrict__ winCodes,
    const unsigned char* __restrict__ winScales,
    const unsigned char* __restrict__ mainCodes,
    const unsigned char* __restrict__ mainScales,
    const int* __restrict__ indices,
    const float* __restrict__ sink,
    int heads, int dim, int k, int windowSlots, int mainRows, int winEnc, int mainEnc, float scale)
{
    extern __shared__ float probs[];
    __shared__ float scratch[ATT_THREADS / 32];
    const int t = blockIdx.x, h = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const long long total = (long long)windowSlots + mainRows;
    const int* idx = indices + (long long)t * k;
    const float* q = query + ((long long)t * heads + h) * dim;
    float* o = output + ((long long)t * heads + h) * dim;
    const float sk = sink[h];

    for (int j = warp; j < k; j += ATT_THREADS / 32)
    {
        const int id = idx[j];
        float dot = 0.0f;
        if (id >= 0 && id < total)
        {
            const bool inWindow = id < windowSlots;
            const long long row = inWindow ? id : id - windowSlots;
            const int enc = inWindow ? winEnc : mainEnc;
            const unsigned char* codes = inWindow ? winCodes : mainCodes;
            const unsigned char* scales = inWindow ? winScales : mainScales;
            for (int d = lane; d < dim; d += 32) dot = __fadd_rn(dot, __fmul_rn(q[d], lc_load(enc, codes, scales, row, dim, d)));
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
            dot = __fmul_rn(dot, scale);
        }
        if (lane == 0) probs[j] = dot;
    }
    __syncthreads();

    float mx = sk;
    for (int j = threadIdx.x; j < k; j += ATT_THREADS)
    {
        const int id = idx[j];
        if (id >= 0 && id < total) mx = fmaxf(mx, probs[j]);
    }
    mx = att_block_max(mx, scratch);
    float denom = 0.0f;
    for (int j = threadIdx.x; j < k; j += ATT_THREADS)
    {
        const int id = idx[j];
        if (id >= 0 && id < total)
        {
            const float p = expf(probs[j] - mx);
            probs[j] = p;
            denom += p;
        }
    }
    denom = att_block_sum(denom, scratch);
    denom = __fadd_rn(denom, expf(sk - mx));

    for (int d = threadIdx.x; d < dim; d += ATT_THREADS)
    {
        float acc = 0.0f;
        for (int j = 0; j < k; j++)
        {
            const int id = idx[j];
            if (id < 0 || id >= total) continue;
            const bool inWindow = id < windowSlots;
            const long long row = inWindow ? id : id - windowSlots;
            acc = __fadd_rn(acc, __fmul_rn(probs[j], lc_load(inWindow ? winEnc : mainEnc, inWindow ? winCodes : mainCodes,
                                                              inWindow ? winScales : mainScales, row, dim, d)));
        }
        o[d] = __fdiv_rn(acc, denom);
    }
}

extern "C" __global__ void __launch_bounds__(IDX_WARPS * 32) indexer_scores_f32(
    float* __restrict__ scores,
    const float* __restrict__ query,
    const unsigned char* __restrict__ keyCodes,
    const unsigned char* __restrict__ keyScales,
    const float* __restrict__ headWeights,
    const int* __restrict__ compressLens,
    const unsigned char* __restrict__ candidates,
    int heads, int dim, int keys, int keyEnc, float scale)
{
    const int t = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * IDX_WARPS + (threadIdx.x >> 5);
    if (n >= keys) return;
    const long long at = (long long)t * keys + n;
    if (n >= compressLens[t] || (candidates != nullptr && candidates[at] == 0))
    {
        if (lane == 0) scores[at] = __int_as_float(0xff800000);
        return;
    }
    float kv[IDX_MAX_PER_LANE];
    for (int i = 0; i < IDX_MAX_PER_LANE; i++)
    {
        const int d = lane + 32 * i;
        kv[i] = d < dim ? lc_load(keyEnc, keyCodes, keyScales, n, dim, d) : 0.0f;
    }
    float sum = 0.0f;
    for (int h = 0; h < heads; h++)
    {
        const float* q = query + ((long long)t * heads + h) * dim;
        float dot = 0.0f;
        for (int i = 0; i < IDX_MAX_PER_LANE; i++)
        {
            const int d = lane + 32 * i;
            if (d < dim) dot = __fadd_rn(dot, __fmul_rn(q[d], kv[i]));
        }
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
        sum = __fadd_rn(sum, __fmul_rn(fmaxf(dot, 0.0f), headWeights[(long long)t * heads + h]));
    }
    if (lane == 0) scores[at] = __fmul_rn(sum, scale);
}

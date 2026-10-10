// Register-resident twins of lm_rmsnorm_q8_1_f32 and lm_add_rmsnorm_q8_1_f32 (Kernels/lm/lm_f32.cu) for rows of 256 * K elements, K a
// power of two up to 32, in a 256-thread block.
//
// BIT-IDENTICAL to the originals: the same per-thread sequential sum of squares (element t + 256 k, k ascending), the same pairwise
// shared-memory tree (stages 128, 64, 32 across warps, then 16 .. 1 inside warp 0), the same rsqrtf and the same
// "in * invRms * weight" product. What changes is only the mechanics: the cross-warp stages are done by warp 0 from one shared-memory
// round trip (two barriers, not nine), the in-warp stages are shuffles, and the output values stay in registers - lane l of warp w owns
// element w * 32 + l + 256 k, so each (warp, k) pair IS one aligned 32-block and quantizes without re-reading global memory.
//
// Launch: grid = rows, block = 256, static shared memory (a 256-float partial array and one scalar).
//
// Build: ../lm/build.sh (nvcc -ptx -arch=sm_80).

#define FAST_THREADS 256

__device__ __forceinline__ float fast_tree_sum(float partial, float* sdata)
{
    const unsigned int t = threadIdx.x;
    const unsigned int lane = t & 31u, warp = t >> 5;
    sdata[t] = partial;
    __syncthreads();
    float r = 0.0f;
    if (warp == 0) {
        // Stage 128: warp w += warp w + 4, stage 64: w += w + 2, stage 32: w0 += w1 - the same pairs the in-place tree adds.
        const float a0 = sdata[lane],       a1 = sdata[32 + lane],  a2 = sdata[64 + lane],  a3 = sdata[96 + lane];
        const float a4 = sdata[128 + lane], a5 = sdata[160 + lane], a6 = sdata[192 + lane], a7 = sdata[224 + lane];
        const float b0 = a0 + a4, b1 = a1 + a5, b2 = a2 + a6, b3 = a3 + a7;
        const float c0 = b0 + b2, c1 = b1 + b3;
        float d = c0 + c1;
        d += __shfl_down_sync(0xffffffffu, d, 16);
        d += __shfl_down_sync(0xffffffffu, d, 8);
        d += __shfl_down_sync(0xffffffffu, d, 4);
        d += __shfl_down_sync(0xffffffffu, d, 2);
        d += __shfl_down_sync(0xffffffffu, d, 1);
        r = d;
    }
    return r;   // valid in lane 0 of warp 0
}

// Quantizes one aligned 32-block held one value per lane (verbatim math of q8_1_quantize_block).
__device__ __forceinline__ void fast_quantize_block(signed char* __restrict__ xq, float* __restrict__ xd, float* __restrict__ xs,
                                                    float v, unsigned int lane)
{
    float amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float scale = amax / 127.0f;
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    int q = __float2int_rn(v * inv);
    q = max(-127, min(127, q));
    xq[lane] = (signed char)q;
    int s = q;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) { xd[0] = scale; xs[0] = (float)s; }
}

template <int K>
__device__ __forceinline__ void rmsnorm_q8_fast(
    float* __restrict__ output, signed char* __restrict__ xq, float* __restrict__ xd, float* __restrict__ xs,
    const float* __restrict__ input, const float* __restrict__ weight, unsigned int normDim, float eps)
{
    __shared__ float sdata[FAST_THREADS];
    __shared__ float sInv;
    const unsigned int row = blockIdx.x, t = threadIdx.x, lane = t & 31u, warp = t >> 5;
    const float* inRow = input + (size_t)row * normDim;
    float* outRow = output + (size_t)row * normDim;

    float x[K];
    float partial = 0.0f;
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        const float v = inRow[t + FAST_THREADS * k];
        x[k] = v;
        partial += v * v;
    }
    const float sum = fast_tree_sum(partial, sdata);
    if (t == 0) sInv = rsqrtf(sum / (float)normDim + eps);
    __syncthreads();
    const float invRms = sInv;

    signed char* xqRow = xq + (size_t)row * normDim;
    float* xdRow = xd + (size_t)row * (normDim / 32u);
    float* xsRow = xs + (size_t)row * (normDim / 32u);
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        const unsigned int i = t + FAST_THREADS * k;
        const float o = x[k] * invRms * weight[i];
        outRow[i] = o;
        const unsigned int blk = warp + 8u * k;
        fast_quantize_block(xqRow + (size_t)blk * 32, xdRow + blk, xsRow + blk, o, lane);
    }
}

template <int K>
__device__ __forceinline__ void add_rmsnorm_q8_fast(
    float* __restrict__ residOut, float* __restrict__ normOut, signed char* __restrict__ xq, float* __restrict__ xd, float* __restrict__ xs,
    const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ weight, unsigned int normDim, float eps)
{
    __shared__ float sdata[FAST_THREADS];
    __shared__ float sInv;
    const unsigned int row = blockIdx.x, t = threadIdx.x, lane = t & 31u, warp = t >> 5;
    const float* aRow = a + (size_t)row * normDim;
    const float* bRow = b + (size_t)row * normDim;
    float* residRow = residOut + (size_t)row * normDim;
    float* normRow = normOut + (size_t)row * normDim;

    float x[K];
    float partial = 0.0f;
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        const unsigned int i = t + FAST_THREADS * k;
        const float v = aRow[i] + bRow[i];
        residRow[i] = v;
        x[k] = v;
        partial += v * v;
    }
    const float sum = fast_tree_sum(partial, sdata);
    if (t == 0) sInv = rsqrtf(sum / (float)normDim + eps);
    __syncthreads();
    const float invRms = sInv;

    signed char* xqRow = xq + (size_t)row * normDim;
    float* xdRow = xd + (size_t)row * (normDim / 32u);
    float* xsRow = xs + (size_t)row * (normDim / 32u);
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        const unsigned int i = t + FAST_THREADS * k;
        const float o = x[k] * invRms * weight[i];
        normRow[i] = o;
        const unsigned int blk = warp + 8u * k;
        fast_quantize_block(xqRow + (size_t)blk * 32, xdRow + blk, xsRow + blk, o, lane);
    }
}

#define RMS_FAST(K) \
extern "C" __global__ void __launch_bounds__(FAST_THREADS) lm_rmsnorm_q8_1_fast_k##K( \
    float* __restrict__ output, signed char* __restrict__ xq, float* __restrict__ xd, float* __restrict__ xs, \
    const float* __restrict__ input, const float* __restrict__ weight, unsigned int normDim, unsigned int totalRows, float eps) \
{ if (blockIdx.x < totalRows) rmsnorm_q8_fast<K>(output, xq, xd, xs, input, weight, normDim, eps); } \
extern "C" __global__ void __launch_bounds__(FAST_THREADS) lm_add_rmsnorm_q8_1_fast_k##K( \
    float* __restrict__ residOut, float* __restrict__ normOut, signed char* __restrict__ xq, float* __restrict__ xd, float* __restrict__ xs, \
    const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ weight, unsigned int normDim, unsigned int totalRows, float eps) \
{ if (blockIdx.x < totalRows) add_rmsnorm_q8_fast<K>(residOut, normOut, xq, xd, xs, a, b, weight, normDim, eps); }

RMS_FAST(2)
RMS_FAST(4)
RMS_FAST(8)
RMS_FAST(16)
RMS_FAST(32)

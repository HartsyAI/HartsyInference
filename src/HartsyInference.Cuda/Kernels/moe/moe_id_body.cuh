// Expert-indexed matrix-vector kernels for MoE decode: the dp4a Q8_1 GEMV of Kernels/lm, with the weight row chosen by
// an expert id read from device memory. Nothing here returns to the host between routing and combine, so the whole
// expert stage can sit inside a captured CUDA graph.
//
// A translation unit defines, before including this file:
//   MOE_ID_SUFFIX                      kernel-name suffix (_q4k, _q6k, _q8_0)
//   moe_id_row_bytes(K)                bytes of one weight row of K elements
//   moe_id_row_partial(w, xq, xd, xs, K, warp, warps, lane)   per-lane partial dot of one weight row against one quantized
//                                      activation row; warp/warps split the row's blocks between the warps of a block (0, 1 = whole row)
// and optionally:
//   MOE_ID_HAS_PAIR, moe_id_row_partial_pair(wg, wu, xq, xd, xs, K, lane, &g, &u)   the gate and up rows in one pass, each
//                                      activation load shared by both (the fused gate/up kernel uses it when defined)
//
// Layout (rows = tokens * topk, row r = token * topk + slot, ids[r] = the expert for that pair):
//   gate/up weights  base of expert 0, experts back to back, expertStride bytes apart, each [N, K] row-major
//   xq/xd/xs         the token's activation quantized to Q8_1 (per-32 scale and sum), one row per token
//   act              [rows, N] F32: act(gate . x) * (up . x)

#define MOE_ID_CAT2(a, b) a##b
#define MOE_ID_CAT(a, b) MOE_ID_CAT2(a, b)
#define MOE_ID_NAME(base) MOE_ID_CAT(base, MOE_ID_SUFFIX)

#define MOE_ID_WARP 32

__device__ __forceinline__ float moe_id_warp_sum(float v)
{
    #pragma unroll
    for (int o = MOE_ID_WARP / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Gate and up projections of every routed (token, slot) pair, fused with the activation:
// one warp per output row computes both dot products against the same activation and writes act(gate) * up.
extern "C" __global__ void MOE_ID_NAME(moe_gateup_id)(
    float* __restrict__ act,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ gateW,
    const unsigned char* __restrict__ upW,
    const int* __restrict__ ids,
    long long expertStride,
    int N, int K, int topk, int numExperts, int gelu)
{
    const int lane = threadIdx.x;
    const int n = blockIdx.x * blockDim.y + threadIdx.y;
    const int row = blockIdx.y;
    if (n >= N) return;

    const int e = ids[row];
    float g = 0.0f, u = 0.0f;
    if (e >= 0 && e < numExperts) {
        const int token = row / topk;
        const long long rowOff = (long long)e * expertStride + (long long)n * moe_id_row_bytes(K);
        const signed char* xqr = xq + (size_t)token * K;
        const float* xdr = xd + (size_t)token * (K / 32);
        const float* xsr = xs + (size_t)token * (K / 32);
#ifdef MOE_ID_HAS_PAIR
        moe_id_row_partial_pair(gateW + rowOff, upW + rowOff, xqr, xdr, xsr, K, lane, &g, &u);
#else
        g = moe_id_row_partial(gateW + rowOff, xqr, xdr, xsr, K, 0, 1, lane);
        u = moe_id_row_partial(upW + rowOff, xqr, xdr, xsr, K, 0, 1, lane);
#endif
    }
    g = moe_id_warp_sum(g);
    u = moe_id_warp_sum(u);
    if (lane == 0) {
        float a;
        if (gelu) {
            const float c = 0.7978845608028654f;   // sqrt(2/pi)
            a = 0.5f * g * (1.0f + tanhf(c * (g + 0.044715f * g * g * g)));
        } else {
            a = g / (1.0f + expf(-g));
        }
        act[(size_t)row * N + n] = a * u;
    }
}

// Down projection of every routed pair: out[row] = W_down[ids[row]] . act[row], the activation already quantized per row.
extern "C" __global__ void MOE_ID_NAME(moe_down_id)(
    float* __restrict__ out,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ downW,
    const int* __restrict__ ids,
    long long expertStride,
    int N, int K, int numExperts)
{
    const int lane = threadIdx.x;
    const int n = blockIdx.x * blockDim.y + threadIdx.y;
    const int row = blockIdx.y;
    if (n >= N) return;

    const int e = ids[row];
    float acc = 0.0f;
    if (e >= 0 && e < numExperts) {
        const long long rowOff = (long long)e * expertStride + (long long)n * moe_id_row_bytes(K);
        acc = moe_id_row_partial(downW + rowOff, xq + (size_t)row * K, xd + (size_t)row * (K / 32),
                                 xs + (size_t)row * (K / 32), K, 0, 1, lane);
    }
    acc = moe_id_warp_sum(acc);
    if (lane == 0) out[(size_t)row * N + n] = acc;
}

// The down projection with each output row split across the warps of a block (blockDim.y warps, grid (N, rows)) and combined
// through shared memory in a fixed order. For long-K/small-N shapes (Mixtral's ffn_down, K = 14336) where one warp per row
// leaves too few loads in flight.
extern "C" __global__ void MOE_ID_NAME(moe_down_id_ksplit)(
    float* __restrict__ out,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ downW,
    const int* __restrict__ ids,
    long long expertStride,
    int N, int K, int numExperts)
{
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int n = blockIdx.x;
    const int row = blockIdx.y;

    const int e = ids[row];
    float acc = 0.0f;
    if (e >= 0 && e < numExperts) {
        const long long rowOff = (long long)e * expertStride + (long long)n * moe_id_row_bytes(K);
        acc = moe_id_row_partial(downW + rowOff, xq + (size_t)row * K, xd + (size_t)row * (K / 32),
                                 xs + (size_t)row * (K / 32), K, warp, (int)blockDim.y, lane);
    }
    acc = moe_id_warp_sum(acc);

    __shared__ float partial[16];
    if (lane == 0) partial[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float sum = 0.0f;
        for (int w = 0; w < (int)blockDim.y; ++w) sum += partial[w];
        out[(size_t)row * N + n] = sum;
    }
}

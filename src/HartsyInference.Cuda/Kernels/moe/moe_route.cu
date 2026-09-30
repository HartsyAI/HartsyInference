// Router for one token per block: score the E logits, pick the top-k experts on score + bias with an optional
// group limit, and emit the expert ids with their renormalised, scaled combine weights.
//
//   logits [T,E] F32   ->   topkIdx [T,k] I32, topkWeight [T,k] F32
//
// Semantics are those of the CPU reference (MoeReference.Route), which mirrors the LLM package's host router:
//   - a pick takes the strictly greatest selection score among the not-yet-taken experts, so the lowest index
//     wins an exact tie; the block argmax below reproduces that with an explicit (value, index) order
//   - the weights are the raw scores at the picked experts, summed in pick order, divided by (sum + eps) when
//     renormalising, then multiplied by scale
//   - a group that loses the group-limit pass has its selection scores replaced by maskedValue (HF masked_fill(0):
//     a dropped expert can still be picked if kept ones score below it), it is never removed from the pool
// Every float add and multiply on the weight path is IEEE (no contraction), and every reduction has a fixed
// order, so a launch is deterministic. exp/log differ from the host libm by a few ulps.
//
// Launch: grid = tokens, block = 128, E <= 1024, k <= E. Static shared memory is ~16 KB.
// softplus_f32 at the end is the elementwise op the sqrtsoftplus score is built from.

#define NEG_INF __int_as_float(0xff800000)
#define ROUTE_MAX_E 1024
#define ROUTE_THREADS 128

__device__ __forceinline__ float route_reduce_max(float v, float* warpMax)
{
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) warpMax[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = warpMax[0];
    for (int w = 1; w < ROUTE_THREADS / 32; w++) r = fmaxf(r, warpMax[w]);
    __syncthreads();
    return r;
}

extern "C" __global__ void __launch_bounds__(ROUTE_THREADS) moe_route_f32(
    int* __restrict__ topkIdx,
    float* __restrict__ topkWeight,
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const float* __restrict__ altBias,
    const int* __restrict__ tokenKinds,
    int numExperts,
    int topK,
    int scoring,
    int groupCount,
    int groupsKept,
    float maskedValue,
    int renormalize,
    float renormEps,
    float scale,
    float logitDivisor)
{
    __shared__ float raw[ROUTE_MAX_E];
    __shared__ float sel[ROUTE_MAX_E];
    __shared__ unsigned char taken[ROUTE_MAX_E];
    __shared__ int pick[ROUTE_MAX_E];
    __shared__ float groupScore[ROUTE_MAX_E / 2];
    __shared__ unsigned char kept[ROUTE_MAX_E / 2];
    __shared__ float warpVal[ROUTE_THREADS / 32];
    __shared__ int warpIdx[ROUTE_THREADS / 32];
    __shared__ float partial[ROUTE_THREADS];
    __shared__ float wsumShared;

    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row = logits + (long long)t * numExperts;

    for (int i = tid; i < numExperts; i += ROUTE_THREADS)
    {
        float v = row[i];
        raw[i] = logitDivisor == 1.0f ? v : v / logitDivisor;
        taken[i] = 0;
    }
    __syncthreads();

    if (scoring == 0)
    {
        float m = NEG_INF;
        for (int i = tid; i < numExperts; i += ROUTE_THREADS) m = fmaxf(m, raw[i]);
        m = route_reduce_max(m, warpVal);
        // Fixed-order sum: thread-strided partials, then a serial combine in thread 0.
        float part = 0.0f;
        for (int i = tid; i < numExperts; i += ROUTE_THREADS)
        {
            float e = expf(raw[i] - m);
            raw[i] = e;
            part += e;
        }
        partial[tid] = part;
        __syncthreads();
        if (tid == 0)
        {
            float s = 0.0f;
            for (int i = 0; i < ROUTE_THREADS; i++) s += partial[i];
            wsumShared = s;
        }
        __syncthreads();
        float sum = wsumShared;
        for (int i = tid; i < numExperts; i += ROUTE_THREADS) raw[i] = raw[i] / sum;
    }
    else if (scoring == 1)
    {
        for (int i = tid; i < numExperts; i += ROUTE_THREADS) raw[i] = 1.0f / (1.0f + expf(-raw[i]));
    }
    else
    {
        for (int i = tid; i < numExperts; i += ROUTE_THREADS)
        {
            float x = raw[i];
            float sp;
            if (x > 20.0f) sp = x;
            else
            {
                double e = exp((double)x);
                double u = 1.0 + e;
                sp = (float)(u == 1.0 ? e : log(u) * e / (u - 1.0));
            }
            raw[i] = sqrtf(sp);
        }
    }
    __syncthreads();

    const float* b = (tokenKinds != nullptr && tokenKinds[t] != 0) ? altBias : bias;
    for (int i = tid; i < numExperts; i += ROUTE_THREADS) sel[i] = b == nullptr ? raw[i] : __fadd_rn(raw[i], b[i]);
    __syncthreads();

    if (groupCount > 1)
    {
        const int per = numExperts / groupCount;
        for (int g = tid; g < groupCount; g += ROUTE_THREADS)
        {
            float top1 = NEG_INF, top2 = NEG_INF;
            for (int j = 0; j < per; j++)
            {
                float v = sel[g * per + j];
                if (v > top1) { top2 = top1; top1 = v; }
                else if (v > top2) top2 = v;
            }
            groupScore[g] = __fadd_rn(top1, top2);
            kept[g] = 0;
        }
        __syncthreads();
        if (tid == 0)
        {
            for (int kk = 0; kk < groupsKept; kk++)
            {
                int bestG = -1;
                float bestV = NEG_INF;
                for (int g = 0; g < groupCount; g++)
                    if (!kept[g] && groupScore[g] > bestV) { bestV = groupScore[g]; bestG = g; }
                if (bestG >= 0) kept[bestG] = 1;
            }
        }
        __syncthreads();
        for (int i = tid; i < numExperts; i += ROUTE_THREADS)
            if (!kept[i / per]) sel[i] = maskedValue;
        __syncthreads();
    }

    const int lane = tid & 31, warp = tid >> 5;
    for (int kk = 0; kk < topK; kk++)
    {
        float bestVal = NEG_INF;
        int bestIdx = 0x7fffffff;
        for (int i = tid; i < numExperts; i += ROUTE_THREADS)
            if (!taken[i] && sel[i] > bestVal) { bestVal = sel[i]; bestIdx = i; }
        for (int o = 16; o > 0; o >>= 1)
        {
            float ov = __shfl_xor_sync(0xffffffffu, bestVal, o);
            int oi = __shfl_xor_sync(0xffffffffu, bestIdx, o);
            if (ov > bestVal || (ov == bestVal && oi < bestIdx)) { bestVal = ov; bestIdx = oi; }
        }
        if (lane == 0) { warpVal[warp] = bestVal; warpIdx[warp] = bestIdx; }
        __syncthreads();
        if (tid == 0)
        {
            float v = warpVal[0];
            int idx = warpIdx[0];
            for (int w = 1; w < ROUTE_THREADS / 32; w++)
                if (warpVal[w] > v || (warpVal[w] == v && warpIdx[w] < idx)) { v = warpVal[w]; idx = warpIdx[w]; }
            if (idx == 0x7fffffff)
            {
                // Nothing scored above -inf (NaN or all -inf): take the lowest free expert like the reference.
                idx = 0;
                while (taken[idx]) idx++;
            }
            taken[idx] = 1;
            pick[kk] = idx;
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        float wsum = 0.0f;
        for (int kk = 0; kk < topK; kk++) wsum = __fadd_rn(wsum, raw[pick[kk]]);
        wsumShared = wsum;
    }
    __syncthreads();
    const float denom = __fadd_rn(wsumShared, renormEps);
    for (int kk = tid; kk < topK; kk += ROUTE_THREADS)
    {
        float w = raw[pick[kk]];
        if (renormalize) w = w / denom;
        w = __fmul_rn(w, scale);
        topkIdx[(long long)t * topK + kk] = pick[kk];
        topkWeight[(long long)t * topK + kk] = w;
    }
}

// Elementwise softplus ln(1 + e^x), evaluated in double like SoftplusReference so the two agree; x > 20 returns x
// (torch's threshold). In-place safe: each thread reads and writes only its own element.
extern "C" __global__ void softplus_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long count)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float x = input[gid];
    float sp;
    if (x > 20.0f) sp = x;
    else
    {
        double e = exp((double)x);
        double u = 1.0 + e;
        sp = (float)(u == 1.0 ? e : log(u) * e / (u - 1.0));
    }
    output[gid] = sp;
}

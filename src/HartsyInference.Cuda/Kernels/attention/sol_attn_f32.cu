// General Sol-Attn in F32: SolAttentionReference (Core) on the device, stage for stage. Operands are token-major
// [B, T, H, D]; tokens are cut into blocks of 64, the same partition for queries and keys.
//
//   1. sol_summaries   per (block, head, batch): mean K, summed V and mean Q over the block's live rows.
//   2. sol_key_stats   per (head, batch): the mean of the block means and their variance, per dimension.
//   3. sol_pooled      per (query block, head, batch): each key block's pooled score, the query block's mean query against
//                      the centered key mean, in base-2 units.
//   4. sol_route       per (query block, head, batch): which key blocks are read token by token. By a threshold of tau
//                      standard deviations, or by rank under a budget; neighbours and sink blocks always; a sink query
//                      block reads everything.
//   5. sol_attend      per (query token, head, batch): one base-2 softmax over the exactly read keys and, when the tail is
//                      on, one pooled entry per unread block weighing its live length.
//
// Stages 1 to 4 decide the routing, and a routing decision is a comparison: a score a rounding apart from the
// reference's can flip a block. So they use the round-to-nearest intrinsics (__fadd_rn, __fmul_rn, __fdiv_rn), which the
// compiler does not contract into fused multiply-adds, in the reference's order of operations. Their results are the
// reference's bit for bit, and so is the routing. Stage 5 is ordinary F32 arithmetic and is compared by tolerance.
//
// Not here: the coarse gate. The backend refuses a call that asks for one.
//
// Build: ../attention/build.sh (nvrtc).

#define SOL_BLOCK 64
#define SOL_NEG_INF __int_as_float(0xff800000)
#define SOL_LOG2E 1.4426950408889634f

__device__ __forceinline__ size_t sol_row(size_t b, size_t t, size_t h, int T, int H, int D)
{
    return ((b * T + t) * H + h) * D;
}

// Grid (blocks, H, B), block 128. Thread d owns dimensions d, d + 128, ...
extern "C" __global__ void sol_summaries(
    float* __restrict__ keyMeans, float* __restrict__ valueSums, float* __restrict__ centroids,
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, const int* __restrict__ lengths,
    int T, int H, int D, int blocks)
{
    const int n = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int len = lengths[n];
    const size_t out = (((size_t)b * H + h) * blocks + n) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float ks = 0.0f, vs = 0.0f, qs = 0.0f;
        for (int t = n * SOL_BLOCK; t < n * SOL_BLOCK + len; ++t) {
            const size_t at = sol_row(b, t, h, T, H, D) + d;
            ks = __fadd_rn(ks, K[at]);
            vs = __fadd_rn(vs, V[at]);
            qs = __fadd_rn(qs, Q[at]);
        }
        keyMeans[out + d] = __fdiv_rn(ks, (float)len);
        valueSums[out + d] = vs;
        centroids[out + d] = __fdiv_rn(qs, (float)len);
    }
}

// Grid (H, B), block 128.
extern "C" __global__ void sol_key_stats(
    float* __restrict__ keyMean, float* __restrict__ variance, const float* __restrict__ keyMeans, int H, int D, int blocks)
{
    const int h = blockIdx.x, b = blockIdx.y;
    const size_t head = (size_t)b * H + h;
    const float* means = keyMeans + head * blocks * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float sum = 0.0f;
        for (int n = 0; n < blocks; ++n) sum = __fadd_rn(sum, means[(size_t)n * D + d]);
        const float mean = __fdiv_rn(sum, (float)blocks);
        float squares = 0.0f;
        for (int n = 0; n < blocks; ++n) {
            const float c = __fsub_rn(means[(size_t)n * D + d], mean);
            squares = __fadd_rn(squares, __fmul_rn(c, c));
        }
        keyMean[head * D + d] = mean;
        variance[head * D + d] = __fdiv_rn(squares, (float)blocks);
    }
}

// Grid (blocks, H, B) over query blocks, block 256. Thread n owns key blocks n, n + 256, ...
extern "C" __global__ void sol_pooled(
    float* __restrict__ pooled, const float* __restrict__ centroids, const float* __restrict__ keyMeans,
    const float* __restrict__ keyMean, int H, int D, int blocks, float log2s)
{
    const int qb = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const size_t head = (size_t)b * H + h;
    const float* c = centroids + (head * blocks + qb) * D;
    const float* mean = keyMean + head * D;
    for (int n = threadIdx.x; n < blocks; n += blockDim.x) {
        const float* k = keyMeans + (head * blocks + n) * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot = __fadd_rn(dot, __fmul_rn(c[d], __fsub_rn(k[d], mean[d])));
        pooled[(head * blocks + qb) * blocks + n] = __fmul_rn(dot, log2s);
    }
}

// Grid (blocks, H, B) over query blocks, block 256. budget < 0 routes by the tau threshold; otherwise by rank: a block is
// read when fewer than budget non-sink blocks score strictly above it, which keeps a group tied at the boundary whole.
extern "C" __global__ void sol_route(
    unsigned char* __restrict__ reads, const float* __restrict__ pooled, const float* __restrict__ centroids,
    const float* __restrict__ variance, int H, int D, int blocks, float log2s, float tau, int budget,
    int sinkStart, int sinkEnd, int sinkQueryStart, int sinkQueryEnd)
{
    __shared__ float threshold;
    const int qb = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const size_t head = (size_t)b * H + h;
    const float* scores = pooled + (head * blocks + qb) * blocks;
    if (budget < 0 && threadIdx.x == 0) {
        const float* c = centroids + (head * blocks + qb) * D;
        const float* var = variance + head * D;
        float spread = 0.0f;
        for (int d = 0; d < D; ++d) spread = __fadd_rn(spread, __fmul_rn(__fmul_rn(c[d], c[d]), var[d]));
        threshold = __fmul_rn(tau, sqrtf(__fadd_rn(__fmul_rn(__fmul_rn(spread, log2s), log2s), 1e-6f)));
    }
    __syncthreads();
    const bool readsAll = qb >= sinkQueryStart && qb < sinkQueryEnd;
    for (int n = threadIdx.x; n < blocks; n += blockDim.x) {
        const bool sink = n >= sinkStart && n < sinkEnd;
        bool read;
        if (budget < 0) {
            read = scores[n] > threshold;
        } else {
            int above = 0;
            for (int m = 0; m < blocks; ++m)
                if (!(m >= sinkStart && m < sinkEnd) && scores[m] > scores[n]) ++above;
            read = budget > 0 && !sink && above < budget;
        }
        const int gap = qb > n ? qb - n : n - qb;
        reads[(head * blocks + qb) * blocks + n] = (read || readsAll || gap <= 1 || sink) ? 1 : 0;
    }
}

// Grid (T, H, B) over query tokens, block 64: thread j is token j of whichever key block is being read.
// bias is null, [T] (biasStride 0) or [B, T] (biasStride T), in natural-log units; -inf masks a key.
extern "C" __global__ void sol_attend(
    float* __restrict__ output, const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ keyMean, const float* __restrict__ pooled, const float* __restrict__ valueSums,
    const unsigned char* __restrict__ reads, const int* __restrict__ lengths, const float* __restrict__ bias, int biasStride,
    int T, int H, int D, int blocks, float log2s, int tail)
{
    __shared__ float qs[256];
    __shared__ float ps[SOL_BLOCK];
    const int i = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int tid = threadIdx.x;
    const int qb = i / SOL_BLOCK;
    const size_t head = (size_t)b * H + h;
    const size_t qAt = sol_row(b, i, h, T, H, D);
    const float* mean = keyMean + head * D;
    const float* scores = pooled + (head * blocks + qb) * blocks;
    const unsigned char* row = reads + (head * blocks + qb) * blocks;
    for (int d = tid; d < D; d += SOL_BLOCK) qs[d] = Q[qAt + d];
    __syncthreads();

    // The logit of token tid of key block n, or -inf when that row is not live or is masked.
    auto logit = [&](int n) {
        if (tid >= lengths[n]) return SOL_NEG_INF;
        const int j = n * SOL_BLOCK + tid;
        const float* k = K + sol_row(b, j, h, T, H, D);
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += qs[d] * (k[d] - mean[d]);
        float value = dot * log2s;
        if (bias != nullptr) value += bias[(size_t)b * biasStride + j] * SOL_LOG2E;
        return value;
    };

    float m = SOL_NEG_INF;
    for (int n = 0; n < blocks; ++n) {
        if (row[n]) m = fmaxf(m, logit(n));
        else if (tail && tid == 0) m = fmaxf(m, scores[n]);
    }
    ps[tid] = m;
    __syncthreads();
    float gmax = SOL_NEG_INF;
    for (int j = 0; j < SOL_BLOCK; ++j) gmax = fmaxf(gmax, ps[j]);
    __syncthreads();

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float denominator = 0.0f;
    if (gmax > SOL_NEG_INF) {
        for (int n = 0; n < blocks; ++n) {
            if (row[n]) {
                const float value = logit(n);
                ps[tid] = value > SOL_NEG_INF ? exp2f(value - gmax) : 0.0f;
                __syncthreads();
                const int len = lengths[n];
                for (int j = 0; j < len; ++j) denominator += ps[j];
                for (int e = 0; e < 4; ++e) {
                    const int d = tid + e * SOL_BLOCK;
                    if (d >= D) break;
                    float sum = 0.0f;
                    for (int j = 0; j < len; ++j) sum += ps[j] * V[sol_row(b, n * SOL_BLOCK + j, h, T, H, D) + d];
                    acc[e] += sum;
                }
                __syncthreads();
            } else if (tail) {
                const float p = exp2f(scores[n] - gmax);
                denominator += p * (float)lengths[n];
                const float* sums = valueSums + (head * blocks + n) * D;
                for (int e = 0; e < 4; ++e) {
                    const int d = tid + e * SOL_BLOCK;
                    if (d >= D) break;
                    acc[e] += p * sums[d];
                }
            }
        }
    }
    const float inverse = 1.0f / fmaxf(denominator, 1e-30f);
    for (int e = 0; e < 4; ++e) {
        const int d = tid + e * SOL_BLOCK;
        if (d >= D) break;
        output[qAt + d] = acc[e] * inverse;
    }
}

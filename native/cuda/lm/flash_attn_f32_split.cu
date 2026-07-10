// Flash-decoding (split-K) for lm_flash_attn_f32 — FP32, grouped-query aware, PLAIN path only.
//
// Why: the monolithic kernel launches one block per (batch, query-head, query-row). In DECODE (Tq=1)
// that is only B·Hq blocks (e.g. 32), which leaves a 128-SM GPU ~75% idle while each block walks all
// kvLen keys sequentially. Flash-decoding splits the key axis into G chunks and runs B·Hq·Tq·G blocks,
// filling the GPU; a tiny combine kernel then merges the G partial online-softmax states per query.
//
// SCOPE: plain attention only — no sink / ALiBi / soft-cap / sliding-window. The host dispatches those
// (Gemma, GPT-OSS, BLOOM, jina-bert) to the proven monolithic kernel, so this path stays simple and is
// covered by the existing causal / non-causal / GQA tests. Splitting is numerically EXACT: each chunk
// computes the same per-key scores as the monolithic loop, just over a sub-range, and the combine is the
// standard online-softmax merge.
//
// Scratch layout, idx = (b*Hq + h)*Tq + r, split g in [0,G):
//   partialM  [N*G]      partialM[idx*G + g]              running max of chunk g
//   partialL  [N*G]      partialL[idx*G + g]              running sum (denominator) of chunk g
//   partialAcc[N*G*D]    partialAcc[(idx*G + g)*D + d]    un-normalised Σ p·V of chunk g (relative to its m)
//
// Build:  ../lm/build.sh   (compiled to flash_attn_f32_split.ptx)

#define INF_F 3.402823466e+38f

// One block per (idx, g): blockIdx.x = idx*G + g. blockDim.x = next pow2 >= D (thread t owns dim t).
extern "C" __global__ void lm_flash_attn_f32_split(
    float* __restrict__ partialM,
    float* __restrict__ partialL,
    float* __restrict__ partialAcc,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    unsigned int B, unsigned int Hq, unsigned int Tq, unsigned int D,
    unsigned int Hkv, unsigned int Lk, unsigned int kvLen, unsigned int kvGroup,
    int causal, int qOffset, float scale,
    unsigned int G, unsigned int chunk,
    const int* __restrict__ dPos)
{
    unsigned int blk = blockIdx.x;
    unsigned int g = blk % G;
    unsigned int idx = blk / G;        // = (b*Hq + h)*Tq + r
    unsigned int r = idx % Tq; unsigned int t2 = idx / Tq;
    unsigned int h = t2 % Hq; unsigned int b = t2 / Hq;
    if (b >= B) return;

    unsigned int hkv = h / kvGroup;
    if (hkv >= Hkv) hkv = Hkv - 1;
    unsigned int tid = threadIdx.x;

    // Graph-capture decode: read this step's position {kvLen, qOffset} from device so one captured graph replays
    // at every step with a FIXED split count (chunk is sized to capacity; splits past kvLen early-exit below).
    if (dPos != nullptr) { kvLen = (unsigned int)dPos[0]; qOffset = dPos[1]; }

    extern __shared__ float sdata[];
    size_t qBase = (((size_t)b * Hq + h) * Tq + r) * D;
    float qv = (tid < D) ? Q[qBase + tid] : 0.0f;

    // Per-query valid key range [0, kMax] (causal) or [0, kvLen-1] (full), intersected with this chunk.
    int kMax = causal ? (int)(qOffset + r) : (int)kvLen - 1;
    if (kMax > (int)kvLen - 1) kMax = (int)kvLen - 1;
    int cStart = (int)(g * chunk);
    int cEnd = (int)(g * chunk + chunk) - 1;            // inclusive
    if (cEnd > (int)kvLen - 1) cEnd = (int)kvLen - 1;
    int kStart = cStart;                                // chunks start at 0 → kMin=0 for the plain path
    int kEnd = (cEnd < kMax) ? cEnd : kMax;

    float m = -INF_F, l = 0.0f, acc = 0.0f;
    for (int k = kStart; k <= kEnd; k++)
    {
        size_t kvBase = (((size_t)b * Hkv + hkv) * Lk + (unsigned int)k) * D;
        sdata[tid] = (tid < D) ? qv * K[kvBase + tid] : 0.0f;
        __syncthreads();
        for (unsigned int s = blockDim.x >> 1; s >= 32; s >>= 1)
        {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid < 32)
        {
            float v = sdata[tid];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, off);
            if (tid == 0) sdata[0] = v;
        }
        __syncthreads();
        float score = sdata[0] * scale;

        float newM = fmaxf(m, score);
        float corr = (m <= -INF_F) ? 0.0f : __expf(m - newM);
        float p = __expf(score - newM);
        l = l * corr + p;
        float vval = (tid < D) ? V[kvBase + tid] : 0.0f;
        acc = acc * corr + p * vval;
        m = newM;
        __syncthreads();   // ensure all threads read sdata[0] before the next key overwrites it
    }

    // Write this chunk's partial online-softmax state. acc is the un-normalised Σ p·V relative to m.
    size_t pBase = (size_t)(idx * G + g);
    if (tid == 0)
    {
        partialM[pBase] = m;
        partialL[pBase] = l;
    }
    if (tid < D) partialAcc[pBase * D + tid] = acc;
}

// One block per idx = (b*Hq + h)*Tq + r. blockDim.x = next pow2 >= D (thread t owns dim t). Merges the
// G chunk partials via online softmax: m* = max m_g, l* = Σ l_g·e^{m_g-m*}, acc*[d] = Σ acc_g[d]·e^{m_g-m*}.
extern "C" __global__ void lm_flash_attn_f32_combine(
    float* __restrict__ out,
    const float* __restrict__ partialM,
    const float* __restrict__ partialL,
    const float* __restrict__ partialAcc,
    unsigned int B, unsigned int Hq, unsigned int Tq, unsigned int D, unsigned int G)
{
    unsigned int idx = blockIdx.x;
    unsigned int total = B * Hq * Tq;
    if (idx >= total) return;
    unsigned int tid = threadIdx.x;

    // Global max over chunks (uniform across threads).
    float m = -INF_F;
    for (unsigned int g = 0; g < G; g++)
    {
        float mg = partialM[idx * G + g];
        m = fmaxf(m, mg);
    }

    float l = 0.0f, acc = 0.0f;
    for (unsigned int g = 0; g < G; g++)
    {
        float mg = partialM[idx * G + g];
        float w = (mg <= -INF_F) ? 0.0f : __expf(mg - m);
        l += partialL[idx * G + g] * w;
        if (tid < D) acc += partialAcc[(size_t)(idx * G + g) * D + tid] * w;
    }

    if (tid < D)
    {
        size_t qBase = (size_t)idx * D;
        out[qBase + tid] = l > 0.0f ? acc / l : 0.0f;
    }
}

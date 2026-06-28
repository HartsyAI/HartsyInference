// FlashAttention (decode + prefill), FP32, grouped-query aware.
//
// Online-softmax attention that never materializes the [Tq x Lk] score matrix and reads K/V grouped-query
// directly (no replicated K/V heads). One CUDA block per (batch, query-head, query-row); blockDim.x = head_dim
// threads (must be a power of two, <= 1024 — true for the 64/128 head dims we run). Each thread owns one output
// dimension; the per-key Q·K dot is a shared-memory tree reduction.
//
// Q:   [B, Hq, Tq, D]   K/V buffer: [B, Hkv, Lk, D] (Lk is the buffer's seq STRIDE; only the first kvLen
// positions are valid, so a fixed-capacity KV buffer can be read in place). out: [B, Hq, Tq, D].
// Query head h reads kv head h / kvGroup. Causal: query row r (abs pos qOffset+r) attends keys 0..qOffset+r.
//
// Build:  ../lm/build.sh   (nvcc --ptx -arch=sm_80)

#define INF_F 3.402823466e+38f

extern "C" __global__ void lm_flash_attn_f32(
    float* __restrict__ out,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    unsigned int B, unsigned int Hq, unsigned int Tq, unsigned int D,
    unsigned int Hkv, unsigned int Lk, unsigned int kvLen, unsigned int kvGroup,
    int causal, int qOffset, float scale, float softcap,
    const float* __restrict__ sink)
{
    unsigned int idx = blockIdx.x;
    unsigned int r = idx % Tq; idx /= Tq;
    unsigned int h = idx % Hq; idx /= Hq;
    unsigned int b = idx;
    if (b >= B) return;

    unsigned int hkv = h / kvGroup;
    if (hkv >= Hkv) hkv = Hkv - 1;
    unsigned int tid = threadIdx.x;

    // blockDim.x is the next power of two >= D (set by the launcher), so the tree reduction below is always
    // power-of-two; threads tid in [D, blockDim.x) are padding (contribute 0, write no output). This lets the
    // kernel handle non-power-of-two head dims (e.g. Phi-3's D=96), not just the 64/128 cases.
    extern __shared__ float sdata[];
    size_t qBase = (((size_t)b * Hq + h) * Tq + r) * D;
    float qv = (tid < D) ? Q[qBase + tid] : 0.0f;

    int kMax = causal ? (int)(qOffset + r) : (int)kvLen - 1;
    if (kMax > (int)kvLen - 1) kMax = (int)kvLen - 1;

    float m = -INF_F, l = 0.0f, acc = 0.0f;
    for (int k = 0; k <= kMax; k++)
    {
        size_t kvBase = (((size_t)b * Hkv + hkv) * Lk + (unsigned int)k) * D;
        sdata[tid] = (tid < D) ? qv * K[kvBase + tid] : 0.0f;
        __syncthreads();
        for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
        {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        float score = sdata[0] * scale;
        // Gemma-2 attention-logit soft-cap: cap·tanh(score/cap), applied to each logit before the softmax.
        if (softcap > 0.0f) score = softcap * tanhf(score / softcap);

        float newM = fmaxf(m, score);
        float corr = (m <= -INF_F) ? 0.0f : __expf(m - newM);
        float p = __expf(score - newM);
        l = l * corr + p;
        float vval = (tid < D) ? V[kvBase + tid] : 0.0f;
        acc = acc * corr + p * vval;
        m = newM;
        __syncthreads();   // ensure all threads read sdata[0] before the next key overwrites it
    }

    // GPT-OSS attention sink: a learned per-head logit that joins the softmax denominator but contributes
    // no value (softmax([scores, sink]) with the sink column dropped from the weighted sum). Seeding the
    // denominator with exp(sink - rowmax) lets a head attend to "nothing" by bleeding probability mass off.
    if (sink != nullptr)
    {
        float sv = sink[h];
        float newM = fmaxf(m, sv);
        float corr = (m <= -INF_F) ? 0.0f : __expf(m - newM);
        float p = __expf(sv - newM);
        l = l * corr + p;
        acc = acc * corr;   // sink value is implicitly zero
        m = newM;
    }

    if (tid < D) out[qBase + tid] = l > 0.0f ? acc / l : 0.0f;
}

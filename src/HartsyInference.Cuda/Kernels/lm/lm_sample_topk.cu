// Draws one token from the sorted top-k candidates of a logits row, on the device, so a sampling decode step can sit inside a captured
// CUDA graph like the greedy one does.
//
// Input: the k largest logits (value-descending, from lm_topk_f32) and their vocabulary ids. When candIdx is given, idx holds positions
// in a candidate list (the merge of a sliced first stage) and candIdx maps a position to its vocabulary id. The chain mirrors the host SamplerChain
// after its top-k step: temperature, softmax over the k survivors, nucleus (top-p) cut on the sorted probabilities, min-p cut relative to
// the best probability, renormalisation and a multinomial draw. temperature <= 0 returns the best id.
//
// rng points at two 64-bit words {seed, counter}; every launch consumes one draw (splitmix64 on seed and counter) and advances the counter
// in device memory, so one captured graph replays with a fresh draw each time.
//
// Launch: 1 block of 1 thread, k <= 64.

#define SAMPLE_MAX_K 64

extern "C" __global__ void lm_sample_from_topk(
    int* __restrict__ outToken,
    const float* __restrict__ vals,
    const int* __restrict__ idx,
    const int* __restrict__ candIdx,
    int k, float temperature, float topP, float minP,
    unsigned long long* __restrict__ rng)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (k > SAMPLE_MAX_K) k = SAMPLE_MAX_K;
    if (temperature <= 0.0f || k <= 1) { outToken[0] = (candIdx != nullptr && idx[0] >= 0) ? candIdx[idx[0]] : idx[0]; return; }

    const float inv = 1.0f / temperature;
    const float top = vals[0] * inv;
    float p[SAMPLE_MAX_K];
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        const float v = vals[i] * inv;
        const float e = (idx[i] < 0 || v == -3.402823466e+38f || !(v > -3.0e+38f)) ? 0.0f : __expf(v - top);
        p[i] = e;
        sum += e;
    }
    const float invSum = 1.0f / sum;
    for (int i = 0; i < k; ++i) p[i] *= invSum;

    // Nucleus: keep the smallest sorted prefix whose mass reaches topP.
    int keep = k;
    if (topP < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < k; ++i) {
            cum += p[i];
            if (cum >= topP) { keep = i + 1; break; }
        }
    }
    // Min-p: drop candidates below minP of the best.
    if (minP > 0.0f) {
        const float cut = minP * p[0];
        int m = 1;
        while (m < keep && p[m] >= cut) ++m;
        keep = m;
    }
    float kept = 0.0f;
    for (int i = 0; i < keep; ++i) kept += p[i];

    unsigned long long seed = rng[0], counter = rng[1];
    unsigned long long z = seed + (counter + 1ull) * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    rng[1] = counter + 1ull;
    const float u = (float)(z >> 40) * (1.0f / 16777216.0f);

    const float target = u * kept;
    float acc = 0.0f;
    int pick = keep - 1;
    for (int i = 0; i < keep; ++i) {
        acc += p[i];
        if (target < acc) { pick = i; break; }
    }
    int sel = idx[pick];
    if (sel < 0) sel = idx[0];
    outToken[0] = candIdx != nullptr ? candIdx[sel] : sel;
}

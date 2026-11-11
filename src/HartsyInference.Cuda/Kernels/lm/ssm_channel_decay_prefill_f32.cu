// The delta-rule recurrence with a per-key-channel log-decay (Kimi KDA; IBackend.SsmChannelDecayPrefill), F32.
// Same warp layout and sequential token order as ssm_delta_prefill_f32 (ssm_prefill_f32.cu): one warp per (state row,
// value head), each lane holds its columns of the row in registers. The only difference is the decay: column c of a row
// decays by exp(logDecay[t, h, c]) instead of one scalar per head. Output is the raw recurrence; no norm, no gate.
// Matches GatedDeltaNetReference.ChannelDecayScan.

#define SSM_MAX_COLS_PER_LANE 8   // sk <= 256

__device__ __forceinline__ float ssm_channel_warp_sum(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

extern "C" {

__global__ void ssm_channel_decay_prefill_f32(
    float* __restrict__ output,          // [seq, hv*sv]
    float* __restrict__ state,           // [hv, sv, sk], in place
    const float* __restrict__ q,         // [seq, (hv/repeat)*sk]
    const float* __restrict__ k,         // [seq, (hv/repeat)*sk]
    const float* __restrict__ v,         // [seq, hv*sv]
    const float* __restrict__ logDecay,  // [seq, hv*sk]
    const float* __restrict__ betaRaw,   // [seq, hv]
    unsigned int seq,
    unsigned int hv,
    unsigned int sv,
    unsigned int sk,
    unsigned int repeat)
{
    unsigned int h = blockIdx.y;
    unsigned int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    unsigned int lane = threadIdx.x & 31u;
    if (row >= sv) return;
    unsigned int kh = h / repeat;
    unsigned int keyDim = (hv / repeat) * sk;
    unsigned int valueDim = hv * sv;
    float* st = state + ((size_t)h * sv + row) * sk;
    float s[SSM_MAX_COLS_PER_LANE];
#pragma unroll
    for (int i = 0; i < SSM_MAX_COLS_PER_LANE; i++)
    {
        unsigned int c = lane + 32u * i;
        s[i] = c < sk ? st[c] : 0.0f;
    }
    for (unsigned int t = 0; t < seq; t++)
    {
        const float* gv = logDecay + (size_t)t * hv * sk + (size_t)h * sk;
        float beta = 1.0f / (1.0f + expf(-betaRaw[(size_t)t * hv + h]));
        const float* kv = k + (size_t)t * keyDim + (size_t)kh * sk;
        const float* qv = q + (size_t)t * keyDim + (size_t)kh * sk;
        float kr[SSM_MAX_COLS_PER_LANE];
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < SSM_MAX_COLS_PER_LANE; i++)
        {
            unsigned int c = lane + 32u * i;
            kr[i] = c < sk ? __ldg(kv + c) : 0.0f;
            if (c < sk) s[i] *= expf(__ldg(gv + c));
            dot += s[i] * kr[i];
        }
        dot = ssm_channel_warp_sum(dot);
        float delta = (__ldg(v + (size_t)t * valueDim + (size_t)h * sv + row) - dot) * beta;
        float o = 0.0f;
#pragma unroll
        for (int i = 0; i < SSM_MAX_COLS_PER_LANE; i++)
        {
            unsigned int c = lane + 32u * i;
            s[i] += kr[i] * delta;
            o += c < sk ? s[i] * __ldg(qv + c) : 0.0f;
        }
        o = ssm_channel_warp_sum(o);
        if (lane == 0) output[(size_t)t * valueDim + (size_t)h * sv + row] = o;
    }
#pragma unroll
    for (int i = 0; i < SSM_MAX_COLS_PER_LANE; i++)
    {
        unsigned int c = lane + 32u * i;
        if (c < sk) st[c] = s[i];
    }
}

}  // extern "C"

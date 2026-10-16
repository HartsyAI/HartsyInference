// Gated-DeltaNet (Qwen3.5 linear attention) prefill over a whole token sequence, F32.
//
//   ssm_conv_split_prefill_f32      causal depthwise conv over [history ; qkvMixed] + SiLU, split to q/k/v
//   ssm_l2norm_rows_f32             per-head L2 norm (then a post-scale) of q or k, one warp per row
//   ssm_conv_history_update_f32     advances the conv carry to this call's last (kernel-1) pre-conv rows
//   ssm_delta_prefill_f32           the delta-rule recurrence, one warp per (state row, value head), sequential
//                                   over tokens: the rows of a head's [sv, sk] state never interact, so every
//                                   row is its own scan and the whole state stays in registers
//   ssm_gated_norm_prefill_f32      rmsnorm(o)·normW·silu(z) per (token, value head), in place
//
// Launch order per layer: conv_split -> l2norm(q) -> l2norm(k) -> history_update -> delta -> gated_norm. The
// history update must follow conv_split, which reads the old carry. Contracts and the host reference:
// IBackend.SsmConvSplitPrefill / SsmDeltaPrefill and GatedDeltaNetReference. Sums are warp-tree ordered, so values
// agree with the sequential host loops to rounding, as the single-token step kernels in lm_f32.cu do.
//
// Build: src/HartsyInference.Cuda/Kernels/lm/build.sh.

#define SSM_MAX_COLS_PER_LANE 8   // sk <= 256

__device__ __forceinline__ float warp_sum(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

extern "C" {

__global__ void ssm_conv_split_prefill_f32(
    float* __restrict__ qOut,            // [seq, keyDim]
    float* __restrict__ kOut,            // [seq, keyDim]
    float* __restrict__ vOut,            // [seq, valueDim]
    const float* __restrict__ qkvMixed,  // [seq, convDim]
    const float* __restrict__ history,   // [kernel-1, convDim]
    const float* __restrict__ convW,     // [convDim, kernel]
    unsigned int convDim,
    unsigned int keyDim,
    unsigned int valueDim,
    unsigned int kernel,
    unsigned int seq)
{
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (unsigned long long)seq * convDim) return;
    unsigned int s = (unsigned int)(idx / convDim);
    unsigned int c = (unsigned int)(idx - (unsigned long long)s * convDim);
    float acc = 0.0f;
    for (unsigned int j = 0; j < kernel; j++)
    {
        unsigned int combined = s + j;
        float src = combined < kernel - 1 ? history[(size_t)combined * convDim + c]
                                          : qkvMixed[(size_t)(combined - (kernel - 1)) * convDim + c];
        acc += convW[(size_t)c * kernel + j] * src;
    }
    float act = acc / (1.0f + expf(-acc));
    if (c < keyDim) qOut[(size_t)s * keyDim + c] = act;
    else if (c < 2u * keyDim) kOut[(size_t)s * keyDim + (c - keyDim)] = act;
    else vOut[(size_t)s * valueDim + (c - 2u * keyDim)] = act;
}

// One warp per contiguous row of `dim` values; blockDim.x = 32 * rowsPerBlock.
__global__ void ssm_l2norm_rows_f32(float* __restrict__ x, unsigned int dim, unsigned int rows, float eps, float postScale)
{
    unsigned int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    unsigned int lane = threadIdx.x & 31u;
    if (row >= rows) return;
    float* r = x + (size_t)row * dim;
    float sumSq = 0.0f;
    for (unsigned int c = lane; c < dim; c += 32u) sumSq += r[c] * r[c];
    float inv = 1.0f / sqrtf(warp_sum(sumSq) + eps);
    for (unsigned int c = lane; c < dim; c += 32u) r[c] = r[c] * inv * postScale;
}

// One thread per channel: shift the carry left by this call's rows, then append its last rows.
__global__ void ssm_conv_history_update_f32(float* __restrict__ history, const float* __restrict__ qkvMixed,
    unsigned int convDim, unsigned int kernel, unsigned int seq)
{
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= convDim || kernel <= 1) return;
    unsigned int carry = seq < kernel - 1 ? seq : kernel - 1;
    for (unsigned int r = 0; r + carry < kernel - 1; r++)
        history[(size_t)r * convDim + c] = history[(size_t)(r + carry) * convDim + c];
    for (unsigned int r = 0; r < carry; r++)
        history[(size_t)(kernel - 1 - carry + r) * convDim + c] = qkvMixed[(size_t)(seq - carry + r) * convDim + c];
}

// grid = (ceil(sv / rowsPerBlock), hv), blockDim.x = 32 * rowsPerBlock. Warp w of block (bx, h) owns state row
// bx * rowsPerBlock + w of value head h; its sk columns live in registers, strided over the lanes.
__global__ void ssm_delta_prefill_f32(
    float* __restrict__ output,          // [seq, hv*sv]  pre-norm readout o = S·q
    float* __restrict__ state,           // [hv, sv, sk]
    const float* __restrict__ q,         // [seq, (hv/repeat)*sk]
    const float* __restrict__ k,         // [seq, (hv/repeat)*sk]
    const float* __restrict__ v,         // [seq, hv*sv]
    const float* __restrict__ alphaRaw,  // [seq, hv]
    const float* __restrict__ betaRaw,   // [seq, hv]
    const float* __restrict__ dtBias,    // [hv]
    const float* __restrict__ ssmA,      // [hv]
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
    float bias = dtBias[h];
    float decay = ssmA[h];
    for (unsigned int t = 0; t < seq; t++)
    {
        float av = alphaRaw[(size_t)t * hv + h] + bias;
        float softplus = av > 20.0f ? av : logf(1.0f + expf(av));
        float gExp = expf(softplus * decay);
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
            s[i] *= gExp;
            dot += s[i] * kr[i];
        }
        dot = warp_sum(dot);
        float delta = (__ldg(v + (size_t)t * valueDim + (size_t)h * sv + row) - dot) * beta;
        float o = 0.0f;
#pragma unroll
        for (int i = 0; i < SSM_MAX_COLS_PER_LANE; i++)
        {
            unsigned int c = lane + 32u * i;
            s[i] += kr[i] * delta;
            o += c < sk ? s[i] * __ldg(qv + c) : 0.0f;
        }
        o = warp_sum(o);
        if (lane == 0) output[(size_t)t * valueDim + (size_t)h * sv + row] = o;
    }
#pragma unroll
    for (int i = 0; i < SSM_MAX_COLS_PER_LANE; i++)
    {
        unsigned int c = lane + 32u * i;
        if (c < sk) st[c] = s[i];
    }
}

// One warp per (token, value head) row of sv values, in place over `output`.
__global__ void ssm_gated_norm_prefill_f32(float* __restrict__ output, const float* __restrict__ z,
    const float* __restrict__ normW, unsigned int rows, unsigned int sv, float eps)
{
    unsigned int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    unsigned int lane = threadIdx.x & 31u;
    if (row >= rows) return;
    float* o = output + (size_t)row * sv;
    const float* zr = z + (size_t)row * sv;
    float var = 0.0f;
    for (unsigned int c = lane; c < sv; c += 32u) var += o[c] * o[c];
    float inv = 1.0f / sqrtf(warp_sum(var) / (float)sv + eps);
    for (unsigned int c = lane; c < sv; c += 32u)
    {
        float zv = zr[c];
        o[c] = o[c] * inv * normW[c] * (zv / (1.0f + expf(-zv)));
    }
}

}

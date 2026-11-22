// The speculative-decode merge: folds a verify step's own rows into an attention result computed over the committed prefix
// alone (DecodeMergeReference in Core, written from Comfy Kitchen's torch definition). A step holds at most 32 rows.
//
// For output row (b, r, h): the prefix result prefix[b, r, h, :] carries the weight exp(lse - m), step row t the weight
// exp(score_t - m) when bit t of mask[r] is set, where score_t = scale * q[b, h, r, :] . k[b, hk, t, :] and
// m = max(lse, largest named score). The merged row is the weighted sum of the prefix result and the named step values over
// the sum of the weights: the softmax over prefix and step keys together, the prefix counted once. A row with no key at all
// (lse == -inf and no bit set) is zeros. A prefix with lse == -inf is not read, and neither is a step row no bit names.
//
// Layouts: Q [B, Hq, S, D]; step K and V [B, Hkv, S, D]; prefix and merged [B, S, Hq * D] (they may be the same buffer);
// lse [B, Hq, S]; mask [S]. All F32 but the mask (I32).
//
// Launch: grid = (B * S * Hq, 1, 1) in the output's row order, block x = (b * S + r) * Hq + h; block = 32, one warp, lane t
// scores step row t.
//
// Build: ../lm/build.sh (nvrtc).

#define MRG_NEG_INF __int_as_float(0xff800000)

extern "C" __global__ void lm_fa_decode_merge_f32(
    float* __restrict__ merged, const float* prefix, const float* __restrict__ lse,
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, const int* __restrict__ mask,
    int Hq, int Hkv, int S, int D, float scale)
{
    __shared__ float w[32];
    const size_t idx = blockIdx.x;
    const int lane = threadIdx.x;
    const int h = (int)(idx % Hq);
    const size_t z = idx / Hq;
    const int r = (int)(z % S);
    const size_t b = z / S;
    const int hk = h / (Hq / Hkv);
    const float* q = Q + ((b * Hq + h) * S + r) * D;
    const float* kBase = K + (b * Hkv + hk) * (size_t)S * D;
    const float* vBase = V + (b * Hkv + hk) * (size_t)S * D;
    const float rowLse = lse[(b * Hq + h) * S + r];
    const int rowMask = mask[r];
    const bool named = lane < S && ((rowMask >> lane) & 1);

    float s = MRG_NEG_INF;
    if (named) {
        const float* k = kBase + (size_t)lane * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += q[d] * k[d];
        s = dot * scale;
    }
    float m = s;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    m = fmaxf(m, rowLse);
    const bool any = m > MRG_NEG_INF;

    const float wt = named && any ? __expf(s - m) : 0.0f;
    w[lane] = wt;
    float total = wt;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) total += __shfl_xor_sync(0xffffffffu, total, o);
    const float wp = any && rowLse > MRG_NEG_INF ? __expf(rowLse - m) : 0.0f;
    total += wp;
    __syncwarp();

    const float* pre = prefix + idx * D;
    float* out = merged + idx * D;
    for (int d = lane; d < D; d += 32) {
        float a = wp > 0.0f ? pre[d] * wp : 0.0f;
        for (int t = 0; t < S; ++t)
            if ((rowMask >> t) & 1) a += w[t] * vBase[(size_t)t * D + d];
        out[d] = total > 0.0f ? a / total : 0.0f;
    }
}

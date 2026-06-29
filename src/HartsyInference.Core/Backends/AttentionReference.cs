using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Pure-F32 CPU reference for FlashAttention, shared by the <see cref="IBackend"/> default implementation
/// and by GPU backends as the fallback when the device kernel can't handle a case (non-F32, or a head dim past
/// the kernel's limit). Kept as a free static so the GPU fallback never re-dispatches into its own override.</summary>
public static class AttentionReference
{
    /// <summary>Online-softmax attention. <paramref name="query"/> is <c>[B, Hq, Tq, D]</c>;
    /// <paramref name="key"/>/<paramref name="value"/> are <c>[B, Hkv, Lk, D]</c>; <paramref name="output"/> is
    /// <c>[B, Hq, Tq, D]</c>. Query head <c>h</c> reads kv head <c>h / kvGroup</c>. When <paramref name="causal"/>,
    /// query row <c>r</c> (absolute position <c>qOffset + r</c>) attends keys <c>0 .. qOffset+r</c>. When
    /// <paramref name="softcap"/> &gt; 0 each logit is soft-capped via <c>softcap·tanh(score/softcap)</c>.
    /// When <paramref name="sink"/> is non-null (a <c>[Hq]</c> F32 tensor of per-head sink logits, GPT-OSS),
    /// each head's sink joins the softmax denominator but contributes no value
    /// (<c>softmax([scores, sink])</c> with the sink column dropped from the weighted sum).
    /// When <paramref name="slidingWindow"/> &gt; 0, query row <c>r</c> additionally only attends the most recent
    /// <c>slidingWindow</c> keys (absolute positions <c>qAbs-slidingWindow+1 .. qAbs</c>) — local-attention layers
    /// of Gemma-2/3, Cohere2, GPT-OSS. 0 = no window (attend the whole causal prefix).</summary>
    public static unsafe void FlashAttention(Tensor output, Tensor query, Tensor key, Tensor value,
        int kvLen, int kvGroup, bool causal, int qOffset, float scale, float softcap = 0f, Tensor? sink = null,
        int slidingWindow = 0, Tensor? alibiSlopes = null)
    {
        if (output.DType != DType.F32 || query.DType != DType.F32 || key.DType != DType.F32 || value.DType != DType.F32)
            throw new NotSupportedException("FlashAttention reference only supports F32.");
        int b = (int)query.Shape[0];
        int hq = (int)query.Shape[1];
        int tq = (int)query.Shape[2];
        int d = (int)query.Shape[3];
        int hkv = (int)key.Shape[1];
        int lk = (int)key.Shape[2];        // buffer seq stride (>= kvLen for a fixed-capacity cache)
        int group = kvGroup <= 0 ? 1 : kvGroup;

        float* qp = (float*)query.DataPointer;
        float* kp = (float*)key.DataPointer;
        float* vp = (float*)value.DataPointer;
        float* op = (float*)output.DataPointer;
        float* sp = sink is null ? null : (float*)sink.DataPointer;
        float* alp = alibiSlopes is null ? null : (float*)alibiSlopes.DataPointer;
        float* acc = stackalloc float[d];

        for (int bi = 0; bi < b; bi++)
            for (int h = 0; h < hq; h++)
            {
                int hkvIdx = h / group;
                if (hkvIdx >= hkv) hkvIdx = hkv - 1;
                for (int r = 0; r < tq; r++)
                {
                    long qBase = (((long)bi * hq + h) * tq + r) * d;
                    int kMax = causal ? Math.Min(kvLen - 1, qOffset + r) : kvLen - 1;
                    // Sliding-window local attention: drop keys older than `slidingWindow` positions back.
                    int kMin = slidingWindow > 0 ? Math.Max(0, qOffset + r - slidingWindow + 1) : 0;
                    float m = float.NegativeInfinity, l = 0f;
                    for (int x = 0; x < d; x++) acc[x] = 0f;
                    for (int k = kMin; k <= kMax; k++)
                    {
                        long kBase = (((long)bi * hkv + hkvIdx) * lk + k) * d;
                        float score = 0f;
                        for (int x = 0; x < d; x++) score += qp[qBase + x] * kp[kBase + x];
                        score *= scale;
                        // ALiBi: per-head linear distance penalty slope·(k_pos − q_pos) (≤ 0 under causality).
                        if (alp != null) score += alp[h] * (k - (qOffset + r));
                        if (softcap > 0f) score = softcap * MathF.Tanh(score / softcap);
                        float newM = MathF.Max(m, score);
                        float corr = m == float.NegativeInfinity ? 0f : MathF.Exp(m - newM);
                        float p = MathF.Exp(score - newM);
                        l = l * corr + p;
                        long vBase = (((long)bi * hkv + hkvIdx) * lk + k) * d;
                        for (int x = 0; x < d; x++) acc[x] = acc[x] * corr + p * vp[vBase + x];
                        m = newM;
                    }
                    if (sp != null)
                    {
                        // Per-head sink: a learned logit in the softmax denominator with zero value.
                        float sv = sp[h];
                        float newM = MathF.Max(m, sv);
                        float corr = m == float.NegativeInfinity ? 0f : MathF.Exp(m - newM);
                        float p = MathF.Exp(sv - newM);
                        l = l * corr + p;
                        for (int x = 0; x < d; x++) acc[x] *= corr;
                        m = newM;
                    }
                    float inv = l > 0f ? 1f / l : 0f;
                    for (int x = 0; x < d; x++) op[qBase + x] = acc[x] * inv;
                }
            }
    }
}

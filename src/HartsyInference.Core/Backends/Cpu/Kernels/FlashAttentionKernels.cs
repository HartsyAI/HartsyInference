using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Cpu.Kernels;

/// <summary>The CPU's online-softmax attention for the Flash route: F32 head-major operands, grouped-query heads,
/// causal and sliding-window exclusions, softcap, sinks and ALiBi, in the order docs/ATTENTION_CONTRACT.md §3 states.</summary>
public static class FlashAttentionKernels
{
    /// <summary>Attends query <c>[B,Hq,Tq,D]</c> over the first <paramref name="kvLen"/> rows of key/value <c>[B,Hkv,Lk,D]</c>;
    /// query head h reads key/value head <c>h / (Hq/Hkv)</c>.</summary>
    /// <param name="causal">When true, query row r (position qOffset+r) attends keys 0..qOffset+r.</param>
    /// <param name="softcap">&gt; 0 soft-caps each logit via <c>softcap·tanh(score/softcap)</c> before the additive terms.</param>
    /// <param name="sink">Non-null <c>[Hq]</c> per-head sink logits: join the softmax denominator, contribute no value.</param>
    /// <param name="slidingWindow">&gt; 0 restricts each query to the most recent N keys; 0 = the full prefix.</param>
    /// <param name="alibiSlopes">Non-null <c>[Hq]</c> slopes: <c>slope·(k−q)</c> for a causal call, <c>−slope·|k−q|</c> otherwise.</param>
    public static unsafe void Run(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, bool causal, int qOffset, float scale,
        float softcap, Tensor? sink, int slidingWindow, Tensor? alibiSlopes)
    {
        if (output.DType != DType.F32 || query.DType != DType.F32 || key.DType != DType.F32 || value.DType != DType.F32)
            throw new NotSupportedException(
                $"The CPU flash kernel computes F32 operands; got Q={query.DType}, K={key.DType}, V={value.DType}, output={output.DType}.");
        if (query.Shape.Rank != 4 || key.Shape.Rank != 4 || !value.Shape.Equals(key.Shape) || !output.Shape.Equals(query.Shape)
            || key.Shape[0] != query.Shape[0] || key.Shape[3] != query.Shape[3] || key.Shape[1] <= 0 || query.Shape[1] % key.Shape[1] != 0)
            throw new ArgumentException(
                $"The CPU flash kernel needs [B,Hq,S,D] operands with Hq a multiple of Hkv; got Q={query.Shape}, K={key.Shape}, V={value.Shape}, output={output.Shape}.");
        ArgumentOutOfRangeException.ThrowIfNegative(slidingWindow);
        if (!float.IsFinite(softcap) || softcap < 0f)
            throw new ArgumentOutOfRangeException(nameof(softcap), "The softcap must be finite and non-negative.");
        int b = (int)query.Shape[0];
        int hq = (int)query.Shape[1];
        int tq = (int)query.Shape[2];
        int d = (int)query.Shape[3];
        int hkv = (int)key.Shape[1];
        int lk = (int)key.Shape[2];
        if (kvLen < 0 || kvLen > lk)
            throw new ArgumentOutOfRangeException(nameof(kvLen), $"kvLen {kvLen} is outside the cache's {lk} rows.");
        int group = hq / hkv;

        float* qp = (float*)query.DataPointer;
        float* kp = (float*)key.DataPointer;
        float* vp = (float*)value.DataPointer;
        float* op = (float*)output.DataPointer;
        float* sp = sink is null ? null : (float*)sink.DataPointer;
        float* alp = alibiSlopes is null ? null : (float*)alibiSlopes.DataPointer;
        float* acc = stackalloc float[d];

        for (int bi = 0; bi < b; bi++)
        {
            for (int h = 0; h < hq; h++)
            {
                int hkvIdx = h / group;
                for (int r = 0; r < tq; r++)
                {
                    long qBase = (((long)bi * hq + h) * tq + r) * d;
                    int kMax = causal ? Math.Min(kvLen - 1, qOffset + r) : kvLen - 1;
                    int kMin = slidingWindow > 0 ? Math.Max(0, qOffset + r - slidingWindow + 1) : 0;
                    float m = float.NegativeInfinity, l = 0f;
                    for (int x = 0; x < d; x++) acc[x] = 0f;
                    for (int k = kMin; k <= kMax; k++)
                    {
                        long kBase = (((long)bi * hkv + hkvIdx) * lk + k) * d;
                        float score = 0f;
                        for (int x = 0; x < d; x++) score += qp[qBase + x] * kp[kBase + x];
                        score *= scale;
                        // Contract §3: softcap first, then the additive terms, so a capped score still takes the full ALiBi penalty.
                        if (softcap > 0f) score = softcap * MathF.Tanh(score / softcap);
                        if (alp != null)
                        {
                            int rel = k - (qOffset + r);
                            score += alp[h] * (causal ? rel : -Math.Abs(rel));
                        }
                        float newM = MathF.Max(m, score);
                        float corr = m == float.NegativeInfinity ? 0f : MathF.Exp(m - newM);
                        float p = MathF.Exp(score - newM);
                        l = l * corr + p;
                        for (int x = 0; x < d; x++) acc[x] = acc[x] * corr + p * vp[kBase + x];
                        m = newM;
                    }
                    if (sp != null)
                    {
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
}

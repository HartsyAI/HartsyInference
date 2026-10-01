using System;

namespace HartsyInference.LLM.Sampling;

/// <summary>Shared numeric helpers for sampler steps: a numerically stable softmax that respects masked (negative-infinity) logits and a descending argsort.</summary>
internal static class SamplerMath
{
    /// <summary>Writes the softmax of <paramref name="logits"/> into <paramref name="probs"/>; entries at negative infinity contribute zero probability.</summary>
    public static void Softmax(ReadOnlySpan<float> logits, Span<float> probs)
    {
        int count = logits.Length;
        float max = float.NegativeInfinity;
        for (int i = 0; i < count; i++)
        {
            if (logits[i] > max)
            {
                max = logits[i];
            }
        }
        double sum = 0.0;
        for (int i = 0; i < count; i++)
        {
            float logit = logits[i];
            if (float.IsNegativeInfinity(logit))
            {
                probs[i] = 0.0f;
                continue;
            }
            float e = MathF.Exp(logit - max);
            probs[i] = e;
            sum += e;
        }
        if (sum <= 0.0)
        {
            return;
        }
        float inv = (float)(1.0 / sum);
        for (int i = 0; i < count; i++)
        {
            probs[i] *= inv;
        }
    }

    /// <summary>Sorts <paramref name="values"/> into descending order in place and fills <paramref name="order"/>
    /// (same length) with the original index of each now-relocated element — i.e. <c>order[rank]</c> is the
    /// pre-sort index of the rank-th largest value, same contract a caller would get from sorting
    /// <c>Enumerable.Range(0, n)</c> by <c>values[i]</c> descending.</summary>
    /// <remarks>Both buffers are caller-owned (reused across calls — e.g. one <see cref="TopPStep"/> instance's
    /// per-generation scratch arrays — so this allocates nothing): a prior <see cref="ArgumentException"/>-style
    /// length mismatch is the caller's bug, not guarded here, since every caller sizes both from the same vocab
    /// count.
    ///
    /// <para>Previously a delegate comparer (<c>Array.Sort(order, (a, b) =&gt; values[b].CompareTo(values[a]))</c>):
    /// correct, but every one of the O(n log n) comparisons paid a virtual delegate dispatch, which dominates the
    /// per-token sampler cost at a ~152K-token vocabulary (measured ~20+ ms/token — see
    /// <c>benchmarks/results/2026-10-01_llm_short_reply_decode.md</c>). <see cref="Array.Sort{TKey, TValue}(TKey[],
    /// TValue[])"/> sorts the primitive <c>float</c> keys directly (no delegate indirection), but only ascending —
    /// negating before and after is exact (sign-bit flip, no rounding, including the softmax's signed-zero masked
    /// entries: <c>-(-0.0f) == 0.0f</c> bit-for-bit) and turns "ascending by -value" into "descending by value"
    /// without a second array or a reversal pass. Ties (two different vocab indices with bit-identical softmax
    /// probability — not seen in practice with real model logits) can land in either relative order under either
    /// algorithm; neither the old nor the new sort documents or guarantees a specific tie-break.</para></remarks>
    public static void SortDescendingByValue(float[] values, int[] order)
    {
        int n = order.Length;
        for (int i = 0; i < n; i++)
        {
            order[i] = i;
            values[i] = -values[i];
        }
        Array.Sort(values, order);
        for (int i = 0; i < n; i++)
        {
            values[i] = -values[i];
        }
    }
}

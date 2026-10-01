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

    /// <summary>Sorts <paramref name="values"/>[0, <paramref name="count"/>) into descending order in place and
    /// permutes <paramref name="order"/>[0, <paramref name="count"/>) (CALLER-populated — see below) to match:
    /// after the call, <c>order[rank]</c> is whatever identity <paramref name="order"/>[rank] carried in BEFORE
    /// the call for the element that is now the rank-th largest value.</summary>
    /// <remarks>Both buffers are caller-owned (reused across calls — e.g. one <see cref="TopPStep"/> instance's
    /// per-generation scratch arrays — so this allocates nothing) and at least <paramref name="count"/> long; a
    /// shorter buffer is the caller's bug, not guarded here. <paramref name="order"/> is NOT initialized to
    /// <c>0..count-1</c> here — a plain argsort-from-identity caller does that itself first (<see cref="TopPStep"/>
    /// 's full-vocabulary fallback); a caller sorting an already-compacted candidate subset instead passes each
    /// candidate's ORIGINAL (pre-compaction) index already in place, which this must preserve through the
    /// permutation rather than overwrite.
    ///
    /// <para>Previously a delegate comparer (<c>Array.Sort(order, (a, b) =&gt; values[b].CompareTo(values[a]))</c>):
    /// correct, but every one of the O(n log n) comparisons paid a virtual delegate dispatch, which dominates the
    /// per-token sampler cost at a ~152K-token vocabulary (measured ~20+ ms/token — see
    /// <c>benchmarks/results/2026-10-01_llm_short_reply_decode.md</c>). <see cref="Array.Sort{TKey, TValue}(TKey[],
    /// TValue[], int, int)"/> sorts the primitive <c>float</c> keys directly (no delegate indirection), but only
    /// ascending — negating before and after is exact (sign-bit flip, no rounding, including the softmax's
    /// signed-zero masked entries: <c>-(-0.0f) == 0.0f</c> bit-for-bit) and turns "ascending by -value" into
    /// "descending by value" without a second array or a reversal pass. Even restricted to primitives this sort
    /// is still the dominant per-token cost at full vocabulary size (measured ~11 ms of the ~12.4 ms this step
    /// took right after that fix — see the same results file's "Tier 2" section), which is why
    /// <see cref="TopPStep"/> sorts only a pre-filtered candidate subset when it safely can, rather than the
    /// whole vocabulary every time. Ties (two different vocab indices with bit-identical softmax probability —
    /// not seen in practice with real model logits) can land in either relative order under either algorithm;
    /// neither the old nor the new sort documents or guarantees a specific tie-break.</para></remarks>
    public static void SortDescendingByValue(float[] values, int[] order, int count)
    {
        for (int i = 0; i < count; i++)
        {
            values[i] = -values[i];
        }
        Array.Sort(values, order, 0, count);
        for (int i = 0; i < count; i++)
        {
            values[i] = -values[i];
        }
    }
}

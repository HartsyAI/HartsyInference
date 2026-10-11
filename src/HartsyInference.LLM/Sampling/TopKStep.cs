using System;
using System.Collections.Generic;

namespace HartsyInference.LLM.Sampling;

/// <summary>Keeps the k highest logits and masks the remainder to negative infinity. Values of <paramref name="k"/> at or below 0 disable the filter at apply time.</summary>
/// <remarks>One instance lives for exactly one generation (see <see cref="TopPStep"/>'s remarks) — the scratch
/// buffer is sized once and reused for every decode step instead of being reallocated per token.</remarks>
public sealed class TopKStep(int k) : ISamplerStep
{
    private readonly int _k = k;
    private float[]? _top;

    /// <inheritdoc/>
    public void Apply(Span<float> logits, IReadOnlyList<int> history)
    {
        if (_k <= 0 || _k >= logits.Length)
        {
            return;
        }
        // The k-th largest value, found with a bounded descending buffer: one pass, and an insertion only when a value beats the
        // current k-th (a few hundred times over a vocabulary-sized scan). The threshold is the same value a full sort would give.
        float threshold = KthLargest(logits);
        // Tokens at exactly the threshold can exceed k when there are ties; cap survivors at k.
        int kept = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] >= threshold && kept < _k)
            {
                kept++;
            }
            else
            {
                logits[i] = float.NegativeInfinity;
            }
        }
    }

    /// <summary>The <c>_k</c>-th largest value of <paramref name="logits"/> (NaN counts as the smallest, as <see cref="Array.Sort(Array)"/> orders it).</summary>
    private float KthLargest(ReadOnlySpan<float> logits)
    {
        float[] top = _top ??= new float[_k];
        int filled = 0;
        foreach (float v in logits)
        {
            if (float.IsNaN(v)) continue;
            if (filled == _k)
            {
                if (!(v > top[_k - 1])) continue;
                filled--;   // the new value displaces the current k-th
            }
            int at = filled++;
            while (at > 0 && top[at - 1] < v)
            {
                top[at] = top[at - 1];
                at--;
            }
            top[at] = v;
        }
        return filled == 0 ? float.NegativeInfinity : top[filled - 1];
    }
}

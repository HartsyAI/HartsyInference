using System;
using System.Collections.Generic;

namespace HartsyInference.LLM.Sampling;

/// <summary>Keeps the k highest logits and masks the remainder to negative infinity. Values of <paramref name="k"/> at or below 0 disable the filter at apply time.</summary>
/// <remarks>One instance lives for exactly one generation (see <see cref="TopPStep"/>'s remarks) — the scratch
/// buffer is sized once and reused for every decode step instead of being reallocated per token.</remarks>
public sealed class TopKStep(int k) : ISamplerStep
{
    private readonly int _k = k;
    private float[]? _sorted;

    /// <inheritdoc/>
    public void Apply(Span<float> logits, IReadOnlyList<int> history)
    {
        if (_k <= 0 || _k >= logits.Length)
        {
            return;
        }
        // Find the k-th largest value via a copy-and-sort; survivors are those at or above it.
        if (_sorted is null || _sorted.Length != logits.Length)
        {
            _sorted = new float[logits.Length];
        }
        float[] sorted = _sorted;
        logits.CopyTo(sorted);
        Array.Sort(sorted);
        float threshold = sorted[logits.Length - _k];
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
}

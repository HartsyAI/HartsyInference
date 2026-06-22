using System;
using System.Collections.Generic;

namespace HartsyInference.LLM.Sampling;

/// <summary>Keeps the k highest logits and masks the remainder to negative infinity.</summary>
public sealed class TopKStep : ISamplerStep
{
    private readonly int _k;

    /// <summary>Creates a top-k step; values of 0 or below disable the filter at apply time.</summary>
    public TopKStep(int k)
    {
        _k = k;
    }

    /// <inheritdoc/>
    public void Apply(Span<float> logits, IReadOnlyList<int> history)
    {
        if (_k <= 0 || _k >= logits.Length)
        {
            return;
        }
        // Find the k-th largest value via a copy-and-sort; survivors are those at or above it.
        float[] sorted = new float[logits.Length];
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

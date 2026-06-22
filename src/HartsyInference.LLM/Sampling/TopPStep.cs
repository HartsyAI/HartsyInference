using System;
using System.Collections.Generic;

namespace HartsyInference.LLM.Sampling;

/// <summary>Nucleus filter: keeps the smallest set of highest-probability tokens whose cumulative softmax mass reaches p, masking the rest to negative infinity.</summary>
public sealed class TopPStep : ISamplerStep
{
    private readonly float _p;

    /// <summary>Creates a top-p step; values of 1.0 or above disable the filter at apply time.</summary>
    public TopPStep(float p)
    {
        _p = p;
    }

    /// <inheritdoc/>
    public void Apply(Span<float> logits, IReadOnlyList<int> history)
    {
        if (_p >= 1.0f)
        {
            return;
        }
        int count = logits.Length;
        float[] probs = new float[count];
        SamplerMath.Softmax(logits, probs);
        int[] order = SamplerMath.ArgsortDescending(probs);
        float cumulative = 0.0f;
        int keep = 0;
        for (int rank = 0; rank < count; rank++)
        {
            cumulative += probs[order[rank]];
            keep = rank + 1;
            if (cumulative >= _p)
            {
                break;
            }
        }
        for (int rank = keep; rank < count; rank++)
        {
            logits[order[rank]] = float.NegativeInfinity;
        }
    }
}

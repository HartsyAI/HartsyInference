using System;
using System.Collections.Generic;

namespace HartsyInference.LLM.Sampling;

/// <summary>Nucleus filter: keeps the smallest set of highest-probability tokens whose cumulative softmax mass reaches p, masking the rest to negative infinity. Values of <paramref name="p"/> at or above 1.0 disable the filter at apply time.</summary>
public sealed class TopPStep(float p) : ISamplerStep
{
    private readonly float _p = p;

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

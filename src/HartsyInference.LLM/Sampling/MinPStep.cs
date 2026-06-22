using System;
using System.Collections.Generic;

namespace HartsyInference.LLM.Sampling;

/// <summary>Masks tokens whose softmax probability falls below a fraction of the most probable token's probability.</summary>
public sealed class MinPStep : ISamplerStep
{
    private readonly float _minP;

    /// <summary>Creates a min-p step; values of 0.0 or below disable the filter at apply time.</summary>
    public MinPStep(float minP)
    {
        _minP = minP;
    }

    /// <inheritdoc/>
    public void Apply(Span<float> logits, IReadOnlyList<int> history)
    {
        if (_minP <= 0.0f)
        {
            return;
        }
        int count = logits.Length;
        float[] probs = new float[count];
        SamplerMath.Softmax(logits, probs);
        float maxProb = 0.0f;
        for (int i = 0; i < count; i++)
        {
            if (probs[i] > maxProb)
            {
                maxProb = probs[i];
            }
        }
        float threshold = _minP * maxProb;
        for (int i = 0; i < count; i++)
        {
            if (probs[i] < threshold)
            {
                logits[i] = float.NegativeInfinity;
            }
        }
    }
}

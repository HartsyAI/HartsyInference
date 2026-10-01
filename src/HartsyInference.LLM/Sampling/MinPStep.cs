using System;
using System.Collections.Generic;

namespace HartsyInference.LLM.Sampling;

/// <summary>Masks tokens whose softmax probability falls below a fraction of the most probable token's probability. Values of <paramref name="minP"/> at or below 0.0 disable the filter at apply time.</summary>
/// <remarks>One instance lives for exactly one generation (see <see cref="TopPStep"/>'s remarks) — the scratch
/// buffer is sized once and reused for every decode step instead of being reallocated per token.</remarks>
public sealed class MinPStep(float minP) : ISamplerStep
{
    private readonly float _minP = minP;
    private float[]? _probs;

    /// <inheritdoc/>
    public void Apply(Span<float> logits, IReadOnlyList<int> history)
    {
        if (_minP <= 0.0f)
        {
            return;
        }
        int count = logits.Length;
        if (_probs is null || _probs.Length != count)
        {
            _probs = new float[count];
        }
        float[] probs = _probs;
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

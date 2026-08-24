using System;
using System.Collections.Generic;

namespace HartsyInference.LLM.Sampling;

/// <summary>Scales every logit by the inverse temperature to sharpen or flatten the distribution. Values of <paramref name="temperature"/> at or below 1.0 leave logits unchanged at apply time.</summary>
public sealed class TemperatureStep(float temperature) : ISamplerStep
{
    private readonly float _temperature = temperature;

    /// <inheritdoc/>
    public void Apply(Span<float> logits, IReadOnlyList<int> history)
    {
        if (_temperature == 1.0f || _temperature <= 0.0f)
        {
            return;
        }
        float inv = 1.0f / _temperature;
        for (int i = 0; i < logits.Length; i++)
        {
            logits[i] *= inv;
        }
    }
}

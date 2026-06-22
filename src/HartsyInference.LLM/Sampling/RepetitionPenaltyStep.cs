using System;
using System.Collections.Generic;

namespace HartsyInference.LLM.Sampling;

/// <summary>Penalizes the logits of already-generated tokens following the Hugging Face convention (divide positive logits, multiply negative ones).</summary>
public sealed class RepetitionPenaltyStep : ISamplerStep
{
    private readonly float _penalty;

    /// <summary>Creates a repetition-penalty step; a penalty of 1.0 leaves logits unchanged at apply time.</summary>
    public RepetitionPenaltyStep(float penalty)
    {
        _penalty = penalty;
    }

    /// <inheritdoc/>
    public void Apply(Span<float> logits, IReadOnlyList<int> history)
    {
        if (_penalty == 1.0f)
        {
            return;
        }
        for (int i = 0; i < history.Count; i++)
        {
            int token = history[i];
            if ((uint)token >= (uint)logits.Length)
            {
                continue;
            }
            float logit = logits[token];
            logits[token] = logit > 0.0f ? logit / _penalty : logit * _penalty;
        }
    }
}

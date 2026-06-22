using System;
using System.Collections.Generic;

namespace HartsyInference.LLM.Sampling;

/// <summary>One stage of a sampler chain that mutates the full-vocabulary logit buffer in place.</summary>
public interface ISamplerStep
{
    /// <summary>Transforms <paramref name="logits"/> in place over the entire vocabulary; <paramref name="history"/> holds the token ids generated so far.</summary>
    void Apply(Span<float> logits, IReadOnlyList<int> history);
}

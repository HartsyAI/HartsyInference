using System;
using System.Collections.Generic;

namespace HartsyInference.LLM.Sampling;

/// <summary>Nucleus filter: keeps the smallest set of highest-probability tokens whose cumulative softmax mass reaches p, masking the rest to negative infinity. Values of <paramref name="p"/> at or above 1.0 disable the filter at apply time.</summary>
/// <remarks>One <see cref="TopPStep"/> instance lives for exactly one generation (built fresh per
/// <see cref="SamplerChain.FromOptions"/> call, which <c>TextGenerationPipeline.Generate</c> calls once per
/// request) and its <see cref="Apply"/> runs once per decode step of that SAME generation — so the scratch
/// buffers below are sized once, on the first token, and reused for every later token instead of being
/// reallocated (two full vocab-sized arrays, ~1.2 MB total at Qwen3's ~152K-token vocabulary) on every single
/// one.</remarks>
public sealed class TopPStep(float p) : ISamplerStep
{
    private readonly float _p = p;
    private float[]? _probs;
    private int[]? _order;

    /// <inheritdoc/>
    public void Apply(Span<float> logits, IReadOnlyList<int> history)
    {
        if (_p >= 1.0f)
        {
            return;
        }
        int count = logits.Length;
        if (_probs is null || _probs.Length != count)
        {
            _probs = new float[count];
            _order = new int[count];
        }
        float[] probs = _probs;
        int[] order = _order!;
        SamplerMath.Softmax(logits, probs);
        SamplerMath.SortDescendingByValue(probs, order);
        float cumulative = 0.0f;
        int keep = 0;
        for (int rank = 0; rank < count; rank++)
        {
            cumulative += probs[rank];
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

using HartsyInference.LLM.Sampling;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>Sampler step that forces the skeleton token grammar: ids the <see cref="UniRigGrammar"/> forbids next are set to negative infinity.</summary>
/// <remarks>The grammar must already have consumed the start tokens (BOS, class); the step consumes <c>history</c> incrementally.</remarks>
public sealed class UniRigGrammarStep(UniRigGrammar grammar) : ISamplerStep
{
    private int _consumed;

    /// <inheritdoc/>
    public void Apply(Span<float> logits, IReadOnlyList<int> history)
    {
        if (history.Count < _consumed)
        {
            throw new InvalidOperationException("UniRig grammar step saw a history shorter than one it already consumed.");
        }

        while (_consumed < history.Count)
        {
            grammar.Advance(history[_consumed++]);
        }

        for (int token = 0; token < logits.Length; token++)
        {
            if (!grammar.Allows(token))
            {
                logits[token] = float.NegativeInfinity;
            }
        }
    }
}

namespace HartsyInference.Diffusion.Prompting;

/// <summary>A tokenized prompt plus the per-token emphasis weight that applies to each id, in the same order.
/// <see cref="Tokens"/> and <see cref="Weights"/> always have the same length; template ids contributed by the
/// caller's prefix/suffix always carry weight <c>1</c>.</summary>
public sealed record WeightedTokenSequence(int[] Tokens, float[] Weights)
{
    /// <summary>True when every token carries weight <c>1</c>, i.e. the sequence is indistinguishable from a plain
    /// encode and every weighting step downstream must be skipped rather than applied as a multiply by one.</summary>
    public bool IsUniformlyUnweighted
    {
        get
        {
            foreach (float weight in Weights)
            {
                if (weight != 1f)
                {
                    return false;
                }
            }
            return true;
        }
    }

    /// <summary>The single weight shared by every non-empty span, or null when the spans disagree — SwarmUI's
    /// <c>uniform_weight</c> (<c>SwarmText.py:464-472</c>), which selects the whole-cond + pooled fallback.</summary>
    public float? UniformWeight { get; init; }

    /// <summary>Cuts the sequence to the encoder's token limit, ids and weights together. Truncating the ids alone
    /// would leave a weight array longer than what it describes, and every weighting step matches the two by
    /// position — so the emphasis would land on a neighbouring word rather than simply be dropped.</summary>
    public WeightedTokenSequence Truncate(int maxTokens) =>
        Tokens.Length <= maxTokens
            ? this
            : new WeightedTokenSequence(Tokens[..maxTokens], Weights[..maxTokens]) { UniformWeight = UniformWeight };
}

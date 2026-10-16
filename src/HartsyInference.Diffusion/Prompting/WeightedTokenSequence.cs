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

    /// <summary>Right-pads to <paramref name="length"/> with <paramref name="padId"/> at weight <c>1</c>; a sequence
    /// already that long is returned unchanged, so truncate first for a fixed window.</summary>
    public WeightedTokenSequence PadTo(int length, int padId)
    {
        if (Tokens.Length >= length)
        {
            return this;
        }
        int[] tokens = new int[length];
        float[] weights = new float[length];
        Array.Fill(weights, 1f);
        Tokens.CopyTo(tokens, 0);
        Weights.CopyTo(weights, 0);
        Array.Fill(tokens, padId, Tokens.Length, length - Tokens.Length);
        return new WeightedTokenSequence(tokens, weights) { UniformWeight = UniformWeight };
    }

    /// <summary>Right-pads to the next multiple of <paramref name="multiple"/> (FLUX.3 pads prompts to 80).</summary>
    public WeightedTokenSequence PadToMultiple(int multiple, int padId)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(multiple, 1);
        int remainder = Tokens.Length % multiple;
        return remainder == 0 ? this : PadTo(Tokens.Length + multiple - remainder, padId);
    }

    /// <summary>Surrounds the sequence with template ids, which always carry weight <c>1</c>.</summary>
    /// <remarks>Separate from <see cref="WeightedTokenBuilder"/>'s own wrapping so a caller can truncate the BODY
    /// first. A tokenizer that caps the text and keeps its specials — <c>ErnieTokenizer.Encode</c> reserves room
    /// for BOS/EOS — cannot be reproduced by building the whole thing and cutting the tail, which would drop the
    /// terminator the encoder expects.</remarks>
    public WeightedTokenSequence Wrap(ReadOnlySpan<int> prefix, ReadOnlySpan<int> suffix)
    {
        if (prefix.IsEmpty && suffix.IsEmpty)
        {
            return this;
        }
        int[] tokens = new int[prefix.Length + Tokens.Length + suffix.Length];
        float[] weights = new float[tokens.Length];
        Array.Fill(weights, 1f);
        prefix.CopyTo(tokens);
        Tokens.CopyTo(tokens, prefix.Length);
        Weights.CopyTo(weights, prefix.Length);
        suffix.CopyTo(tokens.AsSpan(prefix.Length + Tokens.Length));
        return new WeightedTokenSequence(tokens, weights) { UniformWeight = UniformWeight };
    }
}

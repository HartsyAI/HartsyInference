namespace HartsyInference.Diffusion.Prompting;

/// <summary>SwarmUI's <see cref="PromptWeightingMode.CondScaleWithAttention"/> half, as data: the joint-attention
/// positions that carry a non-unit weight, and the two things the patch does with them
/// (<c>attn1_token_weight_patch</c>, <c>SwarmText.py:281-312</c>).</summary>
/// <remarks><para>The patch is not a second way of scaling conditioning — it runs <em>alongside</em>
/// <see cref="CondTokenWeights"/>, inside attention, and splits by direction: a weight below 1 multiplies that
/// token's VALUE rows post-projection, while a weight above 1 adds <c>(w−1)·2</c> to every query's logit for that
/// KEY position. Down-weighting removes what the token contributes; up-weighting makes every other token look at
/// it harder. Neither is expressible as the other.</para>
/// <para>Positions are indices into the JOINT sequence. SwarmUI's guard for that is
/// <c>seq == img_slice[1]</c> — the patch fires only when the image tokens end the sequence, i.e. text is first —
/// so a text index is already a joint index and needs no remapping. A family that concatenates the other way
/// round would need one, which is why the guard exists rather than being assumed.</para></remarks>
public sealed class TextTokenWeights
{
    private readonly (int Position, float Weight)[] _entries;

    private TextTokenWeights((int Position, float Weight)[] entries) => _entries = entries;

    /// <summary>The positions and weights, joint-sequence indexed. Never empty.</summary>
    public IReadOnlyList<(int Position, float Weight)> Entries => _entries;

    /// <summary>Right-aligns per-token weights against the conditioning rows that actually reach attention, the
    /// same <c>offset = cond_len − len(batch)</c> as <c>collect_token_weight_pairs</c> (<c>SwarmText.py:270-278</c>).
    /// Returns null when nothing is weighted, which is what keeps an unweighted prompt on the untouched path.</summary>
    /// <param name="tokenWeights">One weight per token as tokenized, template positions included.</param>
    /// <param name="condLength">Rows the encoder handed the transformer, after any template trim.</param>
    public static TextTokenWeights? TryBuild(ReadOnlySpan<float> tokenWeights, int condLength)
    {
        if (tokenWeights.IsEmpty || condLength <= 0)
        {
            return null;
        }
        int offset = condLength - tokenWeights.Length;
        List<(int, float)> entries = [];
        for (int i = 0; i < tokenWeights.Length; i++)
        {
            int pos = offset + i;
            if (tokenWeights[i] != 1f && pos >= 0 && pos < condLength)
            {
                entries.Add((pos, tokenWeights[i]));
            }
        }
        return entries.Count == 0 ? null : new TextTokenWeights([.. entries]);
    }

    /// <summary>Per-position multipliers for the attention VALUE rows, or null when nothing is down-weighted.
    /// Length <paramref name="jointSeq"/>, 1.0 everywhere the patch leaves alone.</summary>
    /// <remarks>Only <c>w &lt; 1</c> lands here. An up-weight scaling V would raise that token's contribution
    /// wherever attention already looked at it, which is not what SwarmUI does — it changes where attention looks.</remarks>
    public float[]? BuildValueRowScale(int jointSeq)
    {
        float[]? scale = null;
        foreach ((int position, float weight) in _entries)
        {
            if (weight >= 1f || position >= jointSeq)
            {
                continue;
            }
            scale ??= BuildFilled(jointSeq, 1f);
            scale[position] = weight;
        }
        return scale;
    }

    /// <summary>Additive per-key logit bias <c>(w−1)·2</c>, or null when nothing is up-weighted. Length
    /// <paramref name="jointSeq"/>, stored as the single query row an SDPA mask of shape <c>[1,1,1,Skv]</c>
    /// broadcasts over every query.</summary>
    public float[]? BuildKeyLogitBias(int jointSeq)
    {
        float[]? bias = null;
        foreach ((int position, float weight) in _entries)
        {
            if (weight <= 1f || position >= jointSeq)
            {
                continue;
            }
            bias ??= BuildFilled(jointSeq, 0f);
            bias[position] = (weight - 1f) * 2f;
        }
        return bias;
    }

    private static float[] BuildFilled(int length, float value)
    {
        float[] array = new float[length];
        if (value != 0f)
        {
            Array.Fill(array, value);
        }
        return array;
    }
}

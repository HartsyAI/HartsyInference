namespace HartsyInference.Diffusion.Prompting;

/// <summary>Chooses between a family's own whole-template encode and the per-span weighted build, for the
/// <see cref="PromptWeightingMode.CondScale"/> families whose tokenizer renders the chat template as text.</summary>
/// <remarks><para>The choice is not cosmetic. A template renderer typically BPEs the prompt together with the
/// text immediately before it — <c>Qwen3Tokenizer</c> concatenates <c>"user\n"</c> with the prompt in one call,
/// deliberately, so that a prompt starting with whitespace merges its newline with the template's. Tokenizing the
/// prompt on its own puts a pre-tokenization boundary there and can produce different ids for the same text.</para>
/// <para>SwarmUI has the same property and accepts it, because splicing template ids around a separately
/// tokenized leaf is what <c>calc_leaf</c> does. But it is only acceptable where a weight actually exists: an
/// UNWEIGHTED prompt must keep the family's own encode, or wiring weighting would move every existing
/// generation. That is the whole job of this type.</para></remarks>
public static class TemplatedPromptTokens
{
    /// <summary>Tokenizes <paramref name="prompt"/>, weights and all.</summary>
    /// <param name="encodeTemplated">The family's existing encode: prompt in, full templated ids out. Used
    /// verbatim whenever nothing is weighted, so that path stays byte-identical.</param>
    /// <param name="encodeRaw">Span tokenizer: text in, ids out, no template and no specials.</param>
    /// <param name="templatePrefix">The ids <paramref name="encodeTemplated"/> puts before the prompt.</param>
    /// <param name="templateSuffix">The ids it puts after.</param>
    public static WeightedTokenSequence Build(string? prompt, Func<string, int[]> encodeTemplated,
        Func<string, IReadOnlyList<int>> encodeRaw, ReadOnlySpan<int> templatePrefix, ReadOnlySpan<int> templateSuffix)
    {
        ArgumentNullException.ThrowIfNull(encodeTemplated);
        ArgumentNullException.ThrowIfNull(encodeRaw);
        string text = prompt ?? "";
        IReadOnlyList<WeightedSpan> spans = PromptWeighting.Parse(text);
        if (PromptWeighting.HasWeights(spans))
        {
            return WeightedTokenBuilder.Build(spans, encodeRaw, templatePrefix, templateSuffix);
        }
        int[] ids = encodeTemplated(text)
            ?? throw new InvalidOperationException("The templated encoder returned null for a prompt.");
        float[] weights = new float[ids.Length];
        Array.Fill(weights, 1f);
        return new WeightedTokenSequence(ids, weights) { UniformWeight = 1f };
    }
}

using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Features;

/// <summary>SwarmUI's <see cref="PromptWeightingMode.ComfyBlend"/> for the T5-style encoders, the way
/// <see cref="WeightedConditioning"/> is for CLIP. The blend itself is <see cref="ComfyBlend"/>; this owns the two
/// things every caller of it gets wrong.</summary>
/// <remarks><para><b>The empty baseline must match the conditioning exactly in shape</b>, because it is subtracted
/// from it row by row. For a T5 tokenizer that pads to a fixed window that is free — one empty encode fits any
/// prompt, and it can be cached for the lifetime of the pipeline because it does not depend on the prompt at all.
/// A family whose conditioning length tracks its prompt cannot reuse one and needs a per-prompt baseline.</para>
/// <para><b>Conditioning caches keyed on token ids will serve the wrong tensor.</b> Once the recipe has taken the
/// emphasis off the text, <c>(cat:1.5)</c> and <c>cat</c> tokenize identically, so the second generation hits the
/// cache and gets the unweighted encode. Callers must cache the PLAIN conditioning and blend a per-request copy
/// after the fetch, which is what <see cref="Blend"/> being a separate step is for.</para></remarks>
public static class T5WeightedConditioning
{
    /// <summary>Tokenizes to the encoder's fixed window with one weight per row, or <c>(tokens, null)</c> when
    /// nothing is weighted — which is what keeps an ordinary prompt on its original single-encode path.</summary>
    /// <remarks>The unweighted case still goes through the spans rather than the raw string, because the grammar
    /// has to come off the text whether or not it does anything: <c>(red:1.0)</c> weighs 1, but its parens are not
    /// part of the prompt and would reach the encoder as prose.</remarks>
    public static (int[] Tokens, float[]? Weights) Tokenize(T5Tokenizer tokenizer, string? prompt) =>
        TokenizeSpans(tokenizer, PromptWeighting.Parse(PromptTagFlattening.Flatten(prompt)));

    /// <summary>The spans form, for a caller that already parsed the prompt.</summary>
    public static (int[] Tokens, float[]? Weights) TokenizeSpans(
        T5Tokenizer tokenizer, IReadOnlyList<WeightedSpan> spans)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(spans);
        if (!PromptWeighting.HasWeights(spans))
        {
            return (tokenizer.Encode(PromptWeighting.Join(spans)), null);
        }
        WeightedTokenSequence built = WeightedTokenBuilder.Build(spans, tokenizer.EncodeRaw, [], []);
        int window = tokenizer.MaxLength;
        int[] tokens = new int[window];
        float[] weights = new float[window];
        Array.Fill(weights, 1f);
        // Pad and EOS rows weigh 1: they are not part of the prompt, and blending them would pull the padding
        // toward the empty encode along with the words — a whole-sequence drift, not a per-word emphasis.
        int real = Math.Min(built.Tokens.Length, window - 1);
        Array.Copy(built.Tokens, tokens, real);
        Array.Copy(built.Weights, weights, real);
        tokens[real] = T5Tokenizer.EosTokenId;
        for (int i = real + 1; i < window; i++)
        {
            tokens[i] = T5Tokenizer.PadTokenId;
        }
        return (tokens, weights);
    }

    /// <summary>The token batch for the empty-prompt baseline — the same padding the prompt got, which is the
    /// whole point of it.</summary>
    /// <remarks>This is ComfyUI's <c>gen_empty_tokens</c>: start (none for T5) + end + padding. Its pad id is a
    /// per-family convention and ours need not match, because <see cref="ComfyBlend"/> only rewrites rows whose
    /// weight differs from 1 and pad rows always weigh 1 — so the baseline's padding region is never read. What
    /// must match is the prefix: a baseline whose row 0 holds a different token from the reference's shifts the
    /// blend wherever the first word is weighted.</remarks>
    public static int[] EmptyTokens(T5Tokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        return tokenizer.Encode("");
    }

    /// <summary>Replaces <paramref name="embeds"/> with its blend toward <paramref name="empty"/>, or leaves it
    /// alone when there is nothing to apply. The replacement is the caller's to dispose.</summary>
    public static void Blend(IBackend backend, ref Tensor embeds, Tensor empty, float[]? weights)
    {
        if (weights is null)
        {
            return;
        }
        Tensor? blended = ComfyBlend.Apply(backend, embeds, empty, weights);
        if (blended is null)
        {
            return;
        }
        embeds.Dispose();
        embeds = blended;
    }

    /// <summary>A blended COPY, for a caller holding conditioning it does not own — the cached-tensor case. Null
    /// when there is nothing to apply, meaning "keep using the original".</summary>
    public static Tensor? BlendCopy(IBackend backend, Tensor embeds, Tensor empty, float[]? weights) =>
        weights is null ? null : ComfyBlend.Apply(backend, embeds, empty, weights);
}

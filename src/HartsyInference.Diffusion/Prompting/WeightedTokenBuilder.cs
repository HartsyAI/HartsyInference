using System.Collections.Generic;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>Tokenizes <see cref="WeightedSpan"/>s into one id sequence plus a parallel per-token weight array, wrapped
/// in whatever instruction-template ids the caller's encoder needs. This is SwarmUI's <c>calc_leaf</c>
/// (<c>SwarmText.py:401-411</c>): each span is tokenized ALONE with the tokenizer's own weighting disabled, and the
/// span's weight is then stamped onto every id it produced — so the weights survive even on the tokenizers ComfyUI
/// configures with <c>disable_weights=True</c>, which is what makes <see cref="PromptWeightingMode.CondScale"/>
/// possible at all.</summary>
/// <remarks>Per-span tokenization is the parity behaviour, not an approximation, but it does mean a BPE/SPM merge can
/// no longer span a weight boundary: <c>(cat:1.2)flap</c> tokenizes differently from <c>catflap</c>. SwarmUI has the
/// same property. The invariant that matters is the one callers gate on — a prompt with no weighting syntax produces
/// exactly one span and therefore exactly the plain <c>EncodeRaw</c> ids.</remarks>
public static class WeightedTokenBuilder
{
    /// <summary>Builds the templated id/weight sequence for <paramref name="spans"/>.</summary>
    /// <param name="encodeRaw">The encoder's raw tokenizer: text in, ids out, no specials and no padding.</param>
    /// <param name="templatePrefix">Ids the encoder prepends (chat-template system/user header, BOS, …); weight 1.</param>
    /// <param name="templateSuffix">Ids the encoder appends (turn end, assistant header, EOS, …); weight 1.</param>
    public static WeightedTokenSequence Build(IReadOnlyList<WeightedSpan> spans,
        Func<string, IReadOnlyList<int>> encodeRaw, ReadOnlySpan<int> templatePrefix, ReadOnlySpan<int> templateSuffix)
    {
        ArgumentNullException.ThrowIfNull(spans);
        ArgumentNullException.ThrowIfNull(encodeRaw);
        List<int> tokens = new List<int>(templatePrefix.Length + templateSuffix.Length + 64);
        List<float> weights = new List<float>(tokens.Capacity);
        foreach (int id in templatePrefix)
        {
            tokens.Add(id);
            weights.Add(1f);
        }
        float? uniform = null;
        bool firstContent = true;
        foreach (WeightedSpan span in spans)
        {
            if (span.Text.Length == 0)
            {
                continue;
            }
            IReadOnlyList<int> ids = encodeRaw(span.Text)
                ?? throw new InvalidOperationException("encodeRaw returned null for a prompt span.");
            for (int i = 0; i < ids.Count; i++)
            {
                tokens.Add(ids[i]);
                weights.Add(span.Weight);
            }
            // SwarmUI's uniform_weight ignores whitespace-only leaves, so "a (b:1.5)" is not uniform but
            // "(a:1.5) (b:1.5)" is — the separating space must not veto the whole-cond fallback.
            if (span.Text.AsSpan().IsWhiteSpace())
            {
                continue;
            }
            if (firstContent)
            {
                uniform = span.Weight;
                firstContent = false;
            }
            else if (uniform != span.Weight)
            {
                uniform = null;
            }
        }
        foreach (int id in templateSuffix)
        {
            tokens.Add(id);
            weights.Add(1f);
        }
        return new WeightedTokenSequence([.. tokens], [.. weights]) { UniformWeight = firstContent ? null : uniform };
    }

    /// <summary>Parses <paramref name="prompt"/>'s <c>(text:N)</c> grammar and builds the sequence in one step.</summary>
    public static WeightedTokenSequence Build(string prompt, Func<string, IReadOnlyList<int>> encodeRaw,
        ReadOnlySpan<int> templatePrefix, ReadOnlySpan<int> templateSuffix) =>
        Build(PromptWeighting.Parse(prompt ?? ""), encodeRaw, templatePrefix, templateSuffix);
}

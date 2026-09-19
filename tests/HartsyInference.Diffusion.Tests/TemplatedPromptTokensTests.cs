using Xunit;
using HartsyInference.Diffusion.Prompting;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Which encode a chat-template family uses for a given prompt. The choice is load-bearing in both
/// directions: routing an unweighted prompt through the per-span builder moves ids that a template renderer
/// merges, and routing a weighted one through the family's own encode drops the emphasis entirely.</summary>
public sealed class TemplatedPromptTokensTests
{
    private static readonly int[] Prefix = [100, 101];
    private static readonly int[] Suffix = [200];

    /// <summary>Stand-in for a renderer that BPEs the prompt together with the template text before it — the
    /// property <c>Qwen3Tokenizer</c> has, and the reason the split exists. The marker id 999 is what a per-span
    /// build can never produce, so a test can see which path ran.</summary>
    private static int[] Templated(string prompt) => [.. Prefix, 999, .. Raw(prompt), .. Suffix];

    private static int[] Raw(string text) => [.. text.Select(c => (int)c)];

    /// <summary>An ordinary prompt keeps the family's own encode, merge marker and all.</summary>
    [Fact]
    public void AnUnweightedPromptKeepsTheFamilysOwnEncode()
    {
        WeightedTokenSequence sequence = TemplatedPromptTokens.Build("ab", Templated, Raw, Prefix, Suffix);

        Assert.Equal(Templated("ab"), sequence.Tokens);
        Assert.True(sequence.IsUniformlyUnweighted);
    }

    /// <summary>The one the real-weight gate caught: a weight of exactly 1 changes nothing, but the grammar that
    /// expressed it is still in the string. Encoding it verbatim feeds <c>(a:1.0)</c> to the model as prose, so
    /// <c>(word:1.0)</c> stops being byte-identical to <c>word</c> — which is the first thing a weighting gate
    /// checks and the first thing a user tries.</summary>
    [Fact]
    public void AUnitWeightIsStrippedRatherThanEncodedAsProse()
    {
        WeightedTokenSequence weighted = TemplatedPromptTokens.Build("(ab:1.0)", Templated, Raw, Prefix, Suffix);

        Assert.Equal(TemplatedPromptTokens.Build("ab", Templated, Raw, Prefix, Suffix).Tokens, weighted.Tokens);
        Assert.DoesNotContain('(', string.Concat(weighted.Tokens.Select(t => (char)t)));
    }

    /// <summary>A real weight switches to the per-span build, which splices the template ids around a separately
    /// tokenized prompt — so the merge marker is gone and the weights are per token.</summary>
    [Fact]
    public void ARealWeightSwitchesToThePerSpanBuild()
    {
        WeightedTokenSequence sequence = TemplatedPromptTokens.Build("a(b:1.5)", Templated, Raw, Prefix, Suffix);

        Assert.DoesNotContain(999, sequence.Tokens);
        Assert.Equal([100, 101, 'a', 'b', 200], sequence.Tokens);
        Assert.Equal([1f, 1f, 1f, 1.5f, 1f], sequence.Weights);
    }

    /// <summary>Template ids always weigh 1, so a uniform prompt weight never leaks onto the chat scaffolding —
    /// scaling the system-block rows would emphasize the instructions, not the subject.</summary>
    [Fact]
    public void TemplateIdsAreNeverWeighted()
    {
        WeightedTokenSequence sequence = TemplatedPromptTokens.Build("(ab:2.0)", Templated, Raw, Prefix, Suffix);

        Assert.Equal([1f, 1f, 2f, 2f, 1f], sequence.Weights);
        Assert.Equal(2f, sequence.UniformWeight);
    }

    /// <summary>A null or empty prompt is the empty-negative case every CFG pipeline passes, and must not throw.
    /// </summary>
    [Theory]
    [InlineData(null)]
    [InlineData("")]
    public void AnEmptyPromptStillProducesTheTemplate(string? prompt)
    {
        WeightedTokenSequence sequence = TemplatedPromptTokens.Build(prompt, Templated, Raw, Prefix, Suffix);

        Assert.Equal(Templated(""), sequence.Tokens);
    }
}

using HartsyInference.Diffusion.Prompting;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The two helpers a family needs when its tokenizer merges the prompt into the chat template in ONE BPE call
/// (<c>Qwen3Tokenizer.EncodeChatWithLength:178</c>, kept whole so a whitespace-leading prompt merges correctly). Such a
/// family cannot split per emphasis span without moving a token boundary, so an unweighted prompt keeps the whole-string
/// call and only a weighted one splits — which is what SwarmUI itself does, since <c>calc_leaf</c> tokenizes each leaf
/// alone.</summary>
public sealed class ChatTemplateWeightingTests
{
    [Theory]
    [InlineData("a red cat")]
    [InlineData("")]
    [InlineData("  leading and trailing  ")]
    [InlineData("commas, colons: and <angle> brackets")]
    public void JoiningParsedSpansRebuildsAPromptThatCarriedNoEmphasis(string prompt) =>
        Assert.Equal(prompt, PromptWeighting.Join(PromptWeighting.Parse(prompt)));

    [Fact]
    public void JoiningDropsTheEmphasisMarkersTheEncoderWouldOtherwiseReadAsProse() =>
        Assert.Equal("a red cat", PromptWeighting.Join(PromptWeighting.Parse("a (red:1.5) cat")));

    [Fact]
    public void HasWeightsIsFalseForAPromptWrittenEntirelyAtWeightOne()
    {
        Assert.False(PromptWeighting.HasWeights(PromptWeighting.Parse("a red cat")));
        Assert.False(PromptWeighting.HasWeights(PromptWeighting.Parse("a (red:1.0) cat")));
        Assert.True(PromptWeighting.HasWeights(PromptWeighting.Parse("a (red:1.5) cat")));
    }

    /// <summary>The documented load-bearing case: <c>"user\n"</c> and a prompt starting with a newline must merge into
    /// one token. The template accessor splits them, so it may only be used on the weighted path — this pins that the
    /// two really do differ, so the split is never quietly adopted for unweighted prompts.</summary>
    [Fact]
    public void TheTemplateAccessorAndTheMergedEncodeDifferOnAWhitespaceLeadingPrompt()
    {
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 64);
        (int[] prefix, _) = tokenizer.ChatTemplateIds();
        int[] merged = tokenizer.EncodeChat("\ncat");
        int[] split = [.. prefix, .. tokenizer.EncodeRaw("\ncat")];
        Assert.NotEqual(merged[..split.Length], split);
    }

    /// <summary>The same divergence through the path that actually runs — the builder. A weighted prompt is tokenized
    /// per leaf, so a leaf beginning with a newline emits its own newline token instead of merging with the
    /// <c>user\n</c> before it. SwarmUI does the same (<c>calc_leaf</c> tokenizes each leaf alone), so this is the
    /// parity behaviour and is pinned here rather than left to be "fixed" into a boundary shift later.</summary>
    [Fact]
    public void AWeightedWhitespaceLeadingPromptTokenizesPerLeaf()
    {
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 64);
        (int[] prefix, int[] suffix) = tokenizer.ChatTemplateIds();
        WeightedTokenSequence weighted =
            WeightedTokenBuilder.Build("\n(cat:1.5)", tokenizer.EncodeRaw, prefix, suffix);
        int[] merged = tokenizer.EncodeChat("\ncat");
        int overlap = Math.Min(merged.Length, weighted.Tokens.Length);
        Assert.NotEqual(merged[..overlap], weighted.Tokens[..overlap]);
        // The weight still lands on the word, not on the newline that precedes it.
        Assert.Equal(1f, weighted.Weights[prefix.Length]);
        Assert.Contains(weighted.Weights, weight => weight == 1.5f);
    }

    /// <summary>…and that they agree for an ordinary prompt, so the weighted path's ids are the expected ones whenever
    /// the boundary is not in play.</summary>
    [Fact]
    public void TheTemplateAccessorReproducesTheMergedEncodeForAnOrdinaryPrompt()
    {
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 64);
        (int[] prefix, int[] suffix) = tokenizer.ChatTemplateIds();
        int[] merged = tokenizer.EncodeChat("a red cat");
        int[] split = [.. prefix, .. tokenizer.EncodeRaw("a red cat"), .. suffix];
        Assert.Equal(merged[..split.Length], split);
    }
}

using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>MiniMax-H3 tokenizes inside its own encoder and interleaves vision/audio blocks with text, so the
/// thing that makes CondScale work here is an ordering property rather than a length: the user prompt is appended
/// LAST. These pin that, because if a future condition kind were appended after the prompt the weights would
/// silently land on its rows instead.</summary>
public sealed class MiniMaxH3WeightedPromptTests
{
    /// <summary>No emphasis means no weights and the plain ids, including for `(fox:1.0)` — a weight of exactly
    /// one does nothing, but the grammar that expressed it must not reach the encoder as prose.</summary>
    [Theory]
    [InlineData("a red fox in snow, cinematic")]
    [InlineData(" leading space")]
    [InlineData("a red (fox:1.0) in snow")]
    public void AnUnweightedPromptCarriesNoWeightsAndThePlainIds(string prompt)
    {
        using Qwen2Tokenizer tokenizer = new Qwen2Tokenizer();
        string stripped = PromptWeighting.Join(PromptWeighting.Parse(prompt));
        MiniMaxH3TextEncoding.Encoded plain = MiniMaxH3TextEncoding.Build(tokenizer, stripped);
        MiniMaxH3TextEncoding.Encoded built = MiniMaxH3TextEncoding.Build(tokenizer, prompt);
        Assert.Null(built.PromptWeights);
        Assert.Equal(plain.TokenIds, built.TokenIds);
    }

    /// <summary>The weights describe the PROMPT only, not the whole sequence — right-alignment supplies the
    /// offset, so a full-length array would be both redundant and wrong the moment a condition is present.</summary>
    [Fact]
    public void TheWeightsCoverThePromptTokensAlone()
    {
        using Qwen2Tokenizer tokenizer = new Qwen2Tokenizer();
        MiniMaxH3TextEncoding.Encoded built = MiniMaxH3TextEncoding.Build(tokenizer, "a red (fox:1.5) in snow");
        Assert.NotNull(built.PromptWeights);
        Assert.Equal(built.TokenIds.Length, built.PromptWeights!.Length);
        Assert.Contains(built.PromptWeights, w => w == 1.5f);
        Assert.Contains(built.PromptWeights, w => w == 1f);
    }

    /// <summary>With a condition present the prompt is a SUFFIX of the sequence, and the weight array must be
    /// shorter than it by exactly the conditioning's length. That difference is the right-alignment offset, and
    /// it is what keeps the vision rows at weight 1 without naming them.</summary>
    [Fact]
    public void AConditionMakesThePromptASuffixAndTheWeightsStayPromptLength()
    {
        using Qwen2Tokenizer tokenizer = new Qwen2Tokenizer();
        MiniMaxH3TextEncoding.Condition image = new MiniMaxH3TextEncoding.Condition
        {
            Kind = MiniMaxH3TextEncoding.ConditionKind.Image,
            Blocks = [new MiniMaxH3TextEncoding.VisionBlock(4)],
        };
        MiniMaxH3TextEncoding.Encoded withImage = MiniMaxH3TextEncoding.Build(tokenizer, "a red (fox:1.5) in snow", [image]);
        MiniMaxH3TextEncoding.Encoded bare = MiniMaxH3TextEncoding.Build(tokenizer, "a red (fox:1.5) in snow");
        Assert.NotNull(withImage.PromptWeights);
        Assert.Equal(bare.PromptWeights!.Length, withImage.PromptWeights!.Length);
        Assert.True(withImage.TokenIds.Length > withImage.PromptWeights.Length);
        // The prompt's ids really are the tail, which is the property right-alignment depends on.
        Assert.Equal(
            bare.TokenIds[^withImage.PromptWeights.Length..],
            withImage.TokenIds[^withImage.PromptWeights.Length..]);
    }
}

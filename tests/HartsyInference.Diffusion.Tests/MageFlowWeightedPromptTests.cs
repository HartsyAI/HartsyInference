using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Mage-Flow is the second family on SwarmUI's <see cref="PromptWeightingMode.CondScale"/> path and the
/// simplest: it caches no conditioning at all, so the scale applies to a tensor built fresh each generation. What still
/// has to hold is the alignment — its chat template is dropped from the encoder output like Qwen-Image's.</summary>
public sealed class MageFlowWeightedPromptTests
{
    /// <summary>With no emphasis syntax the templated ids must be exactly what the plain tokenizer produced before the
    /// weighting path existed; the per-span builder must not introduce a tokenization boundary of its own.</summary>
    [Fact]
    public void AnUnweightedPromptProducesTheSameIdsAsBefore()
    {
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer();
        (WeightedTokenSequence sequence, int dropIndex) =
            MageFlowRecipePipeline.EncodeWithTemplate(tokenizer, "a red cat on a mat");
        int[] plain = [.. tokenizer.EncodeRaw("a red cat on a mat")];
        Assert.Equal(plain, sequence.Tokens[dropIndex..(dropIndex + plain.Length)]);
        Assert.True(sequence.IsUniformlyUnweighted);
    }

    [Fact]
    public void TheTemplateIdsAroundThePromptCarryNoWeight()
    {
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer();
        (WeightedTokenSequence sequence, int dropIndex) =
            MageFlowRecipePipeline.EncodeWithTemplate(tokenizer, "(red:1.5) cat");
        for (int i = 0; i < dropIndex; i++)
        {
            Assert.Equal(1f, sequence.Weights[i]);
        }
        Assert.Equal(1.5f, sequence.Weights[dropIndex]);
        Assert.Equal(1f, sequence.Weights[^1]);
        Assert.Equal(sequence.Tokens.Length, sequence.Weights.Length);
    }
}

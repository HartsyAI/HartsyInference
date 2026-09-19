using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Qwen-Image is the reference case for SwarmUI's <see cref="PromptWeightingMode.CondScale"/>: it wraps the
/// prompt in a chat template and then DROPS that template's hidden states, which is exactly the situation
/// <c>multiply_cond_by_token_weights</c>' negative right-alignment offset exists for. These pin the two halves that
/// have to agree — the ids the template produces and the weights that describe them.</summary>
public sealed class QwenImageWeightedPromptTests
{
    /// <summary>The invariant that keeps every unweighted generation byte-identical: with no emphasis syntax the
    /// templated ids must be the ones the plain tokenizer would have produced, not a per-span reassembly of them.</summary>
    [Fact]
    public void AnUnweightedPromptProducesTheSameIdsAsBefore()
    {
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer();
        (WeightedTokenSequence sequence, int dropIndex) =
            QwenImageRecipePipeline.EncodeWithTemplate(tokenizer, "a red cat on a mat");
        int[] plain = [.. tokenizer.EncodeRaw("a red cat on a mat")];
        Assert.Equal(plain, sequence.Tokens[dropIndex..(dropIndex + plain.Length)]);
        Assert.True(sequence.IsUniformlyUnweighted);
    }

    /// <summary>Template ids sit on both sides of the prompt and are not part of it; an emphasis that leaked onto the
    /// system block or the assistant header would scale rows the model reads as instructions.</summary>
    [Fact]
    public void OnlyThePromptsOwnTokensCarryItsWeight()
    {
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer();
        (WeightedTokenSequence sequence, int dropIndex) =
            QwenImageRecipePipeline.EncodeWithTemplate(tokenizer, "(red:1.5) cat");
        for (int i = 0; i < dropIndex; i++)
        {
            Assert.Equal(1f, sequence.Weights[i]);
        }
        int promptTokens = tokenizer.EncodeRaw("red").Count + tokenizer.EncodeRaw(" cat").Count;
        for (int i = dropIndex + promptTokens; i < sequence.Weights.Length; i++)
        {
            Assert.Equal(1f, sequence.Weights[i]);
        }
        Assert.Equal(1.5f, sequence.Weights[dropIndex]);
        Assert.Contains(sequence.Weights, weight => weight == 1.5f);
    }

    [Fact]
    public void TheWeightArrayAlwaysDescribesTheIdArray()
    {
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer();
        (WeightedTokenSequence sequence, _) =
            QwenImageRecipePipeline.EncodeWithTemplate(tokenizer, "a (very:1.3) (long:0.7) prompt");
        Assert.Equal(sequence.Tokens.Length, sequence.Weights.Length);
    }

    /// <summary>diffusers truncates the templated sequence at 512; the weights have to be cut with it, or every
    /// emphasis would land one token further along than the word it belongs to.</summary>
    [Fact]
    public void TruncationCutsTheWeightsWithTheIds()
    {
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer();
        string longPrompt = string.Join(' ', Enumerable.Repeat("(cat:1.5)", 600));
        (WeightedTokenSequence sequence, _) = QwenImageRecipePipeline.EncodeWithTemplate(tokenizer, longPrompt);
        Assert.Equal(512, sequence.Tokens.Length);
        Assert.Equal(sequence.Tokens.Length, sequence.Weights.Length);
    }

    /// <summary>The edit template puts the <c>Picture N:</c> blocks after the drop index, so they DO reach the
    /// conditioning — but they are template, and an emphasis must not scale the rows the vision tokens occupy.</summary>
    [Fact]
    public void TheEditTemplatesVisionBlocksCarryNoWeight()
    {
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer();
        (WeightedTokenSequence sequence, int dropIndex) =
            QwenImageEditConditioning.BuildTokens(tokenizer, "(make it red:1.5)", [4]);
        for (int i = 0; i < sequence.Tokens.Length; i++)
        {
            if (sequence.Tokens[i] == Qwen25VlMultimodalEncoder.ImageTokenId)
            {
                Assert.Equal(1f, sequence.Weights[i]);
            }
        }
        Assert.True(dropIndex > 0);
        Assert.Contains(sequence.Weights, weight => weight == 1.5f);
    }
}

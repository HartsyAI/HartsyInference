using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>HunyuanImage pads to a fixed 1034-token window and its encoder then trims to the mask's real length
/// and slices the 34-token chat template off the front. That makes the weight array's LENGTH the thing most
/// likely to be wrong — and wrong in the silent direction, since a mis-sized array right-aligns to a different
/// offset and scales nothing rather than throwing.</summary>
public sealed class HunyuanImageWeightedPromptTests
{
    /// <summary>The prefix check inside <c>TokenizePadded</c> runs on EVERY generation. If the assembled template
    /// disagreed with the encoder's hard-coded 34-token drop, plain HunyuanImage would throw.</summary>
    [Theory]
    [InlineData("a red fox in snow, cinematic")]
    [InlineData(" a prompt that starts with a space")]
    [InlineData(".")]
    public void APlainPromptDoesNotTripThePrefixCheck(string prompt)
    {
        using Qwen2Tokenizer tokenizer = new Qwen2Tokenizer();
        (int[] ids, _, float[]? weights) = HunyuanImageRecipePipeline.TokenizePadded(tokenizer, prompt);
        Assert.Equal(HunyuanImageQwenTextEncoder.PaddedLength, ids.Length);
        Assert.Null(weights);
    }

    /// <summary>The unweighted path must reproduce the chat-template encode byte for byte, including the
    /// `(fox:1.0)` case — a weight of exactly 1 does nothing, but the grammar that expressed it must not reach
    /// the encoder as prose.</summary>
    [Theory]
    [InlineData("a red fox in snow, cinematic")]
    [InlineData(" leading space")]
    [InlineData("a red (fox:1.0) in snow")]
    public void AnUnweightedPromptProducesTheChatTemplateIds(string prompt)
    {
        using Qwen2Tokenizer tokenizer = new Qwen2Tokenizer();
        string stripped = PromptWeighting.Join(PromptWeighting.Parse(prompt));
        int[] expected = Qwen2Tokenizer.PadToLength(
            tokenizer.EncodeChat(stripped, systemPrompt: HunyuanImageQwenTextEncoder.SystemPrompt, addGenerationPrompt: false),
            HunyuanImageQwenTextEncoder.PaddedLength);
        (int[] ids, _, _) = HunyuanImageRecipePipeline.TokenizePadded(tokenizer, prompt);
        Assert.Equal(expected, ids);
    }

    /// <summary>The weights describe the TRIMMED sequence, not the padded window. Returning 1034 of them would
    /// right-align to <c>keep − 1034</c> and push every prompt weight off the front — a no-op that looks like
    /// working coverage.</summary>
    [Fact]
    public void TheWeightsAreTheRealLengthNotThePaddedLength()
    {
        using Qwen2Tokenizer tokenizer = new Qwen2Tokenizer();
        (_, int[] mask, float[]? weights) = HunyuanImageRecipePipeline.TokenizePadded(tokenizer, "a red (fox:1.5) in snow");
        Assert.NotNull(weights);
        int realLen = 0;
        foreach (int m in mask)
        {
            realLen += m;
        }
        Assert.Equal(realLen, weights!.Length);
        Assert.True(realLen < HunyuanImageQwenTextEncoder.PaddedLength);
    }

    /// <summary>Every template row weighs 1. The first 34 are dropped by the encoder, but a weight that leaked
    /// onto the trailing turn-end would scale a row the model reads as structure.</summary>
    [Fact]
    public void OnlyThePromptRowsCarryWeight()
    {
        using Qwen2Tokenizer tokenizer = new Qwen2Tokenizer();
        (_, _, float[]? weights) = HunyuanImageRecipePipeline.TokenizePadded(tokenizer, "a red (fox:1.5) in snow");
        Assert.NotNull(weights);
        // 34 is what the encoder DROPS; the template is really 33 ids long on this tokenizer. Asserting to 34
        // is deliberately one stricter, and holds because this prompt's first span is unweighted. See the TODO
        // on TemplatePrefix for why the two numbers differ.
        for (int i = 0; i < HunyuanImageQwenTextEncoder.TemplatePrefixTokens; i++)
        {
            Assert.Equal(1f, weights![i]);
        }
        Assert.Equal(1f, weights![^1]);
        Assert.Contains(weights, w => w == 1.5f);
    }
}

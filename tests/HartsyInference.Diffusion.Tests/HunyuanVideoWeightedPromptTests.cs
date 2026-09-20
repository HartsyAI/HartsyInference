using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>HunyuanVideo crops a hard-coded 95 template tokens off the front of its conditioning, so the split
/// between "template prefix" and "prompt" has to agree with what the whole-string encode produces — to the token.
/// These run on the embedded Llama-3 tokenizer, no checkpoint required, because the real-weight gate can only
/// sample a few prompts and the failure mode here is a silently misaligned crop.</summary>
public sealed class HunyuanVideoWeightedPromptTests
{
    /// <summary>The prefix-length check inside <c>BuildWeightedTokens</c> runs on EVERY generation, weighted or
    /// not. If the template split disagreed with <c>CropStart</c> this would throw for a plain prompt, which is a
    /// regression on a family that works today rather than a gap in a new feature.</summary>
    [Theory]
    [InlineData("a red fox in snow, cinematic")]
    [InlineData(" a prompt that starts with a space")]
    [InlineData("")]
    public void APlainPromptDoesNotTripThePrefixCheck(string prompt)
    {
        WeightedTokenSequence sequence = HunyuanVideoRecipe.BuildWeightedTokens(prompt);
        Assert.True(sequence.IsUniformlyUnweighted);
    }

    /// <summary>The unweighted path must reproduce the whole-string encode byte for byte. Per-span tokenization is
    /// the parity behaviour where a weight exists, but using it for a prompt with no emphasis would move every
    /// existing generation — a byte-level tokenizer merges a leading space into the following word, so the split
    /// changes ids that the plain path never split.</summary>
    [Theory]
    [InlineData("a red fox in snow, cinematic")]
    [InlineData(" leading space")]
    [InlineData("a red (fox:1.0) in snow")]
    public void AnUnweightedPromptProducesTheWholeStringIds(string prompt)
    {
        // `(fox:1.0)` is in this list deliberately: a weight of exactly 1 does nothing, but the grammar that
        // expressed it is still in the string and must not reach the encoder as prose.
        string stripped = PromptWeighting.Join(PromptWeighting.Parse(prompt));
        Assert.Equal(HunyuanVideoRecipe.BuildTemplatedTokens(stripped),
            HunyuanVideoRecipe.BuildWeightedTokens(prompt).Tokens);
    }

    /// <summary>A weighted prompt must leave every template row at weight 1 — the first <c>CropStart</c> rows are
    /// cropped away, and the trailing <c>&lt;|eot_id|&gt;</c> is not part of the caption.</summary>
    [Fact]
    public void OnlyThePromptRowsCarryWeight()
    {
        WeightedTokenSequence sequence = HunyuanVideoRecipe.BuildWeightedTokens("a red (fox:1.5) in snow");
        Assert.False(sequence.IsUniformlyUnweighted);
        for (int i = 0; i < HunyuanVideoRecipe.CropStart; i++)
        {
            Assert.Equal(1f, sequence.Weights[i]);
        }
        Assert.Equal(1f, sequence.Weights[^1]);
        Assert.Contains(sequence.Weights, w => w == 1.5f);
    }

    /// <summary>The ComfyBlend baseline is one start token then pad — <c>gen_empty_tokens</c> emits start + end +
    /// padding and <c>LLAMAModel</c> declares no end. Row 0 is a real prompt position once the crop is applied, so
    /// a wrong first id would shift the blend rather than fail.</summary>
    [Fact]
    public void TheEmptyBaselineIsOneStartTokenThenPad()
    {
        int[] baseline = HunyuanVideoRecipe.EmptyBaseline(8);
        Assert.Equal([128000, 128258, 128258, 128258, 128258, 128258, 128258, 128258], baseline);
    }
}

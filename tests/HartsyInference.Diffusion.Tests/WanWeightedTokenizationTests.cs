using Xunit;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The token/weight pairing every Wan-family recipe now shares. All five — the base family and the
/// Animate, Animate-2, S2V and VACE variants — call one helper, so these invariants hold once for all of them
/// rather than being re-checked per recipe.</summary>
/// <remarks>umT5 pads to a fixed window, which is what makes one empty-prompt baseline valid for any prompt and
/// ComfyBlend cheap here. These pin the two things that would break silently: a weight array that stops lining up
/// with its tokens, and emphasis leaking onto the EOS/pad rows, which would drag the padding toward the empty
/// encode along with the words.</remarks>
public sealed class WanWeightedTokenizationTests
{
    private static T5Tokenizer Umt5() => T5Tokenizer.CreateUmt5(maxLength: 512);

    /// <summary>An ordinary prompt returns no weight array at all, which is what keeps it on the original
    /// single-encode path — no empty-prompt baseline, no blend, byte-identical output.</summary>
    [Fact]
    public void AnUnweightedPromptAsksForNoBlend()
    {
        using T5Tokenizer tokenizer = Umt5();

        (int[] tokens, float[]? weights) =
            VideoRecipeUtils.TokenizeWeightedWan(tokenizer, PromptWeighting.Parse("a red fox in snow"));

        Assert.Null(weights);
        Assert.Equal(tokenizer.MaxLength, tokens.Length);
    }

    /// <summary>A weight of exactly 1 still does nothing, and — the part worth pinning — its grammar does not
    /// reach the encoder as prose. `(red:1.0)` must tokenize to what `red` tokenizes to.</summary>
    [Fact]
    public void AUnitWeightMatchesThePlainPromptExactly()
    {
        using T5Tokenizer tokenizer = Umt5();

        (int[] weighted, float[]? weights) =
            VideoRecipeUtils.TokenizeWeightedWan(tokenizer, PromptWeighting.Parse("a (red:1.0) fox"));
        (int[] plain, _) = VideoRecipeUtils.TokenizeWeightedWan(tokenizer, PromptWeighting.Parse("a red fox"));

        Assert.Null(weights);
        Assert.Equal(plain, weighted);
    }

    /// <summary>A real weight produces one weight per token position, filling the whole fixed window so the array
    /// can be matched to the encoder's rows without an offset.</summary>
    [Fact]
    public void AWeightedPromptFillsTheWindowWithOneWeightPerRow()
    {
        using T5Tokenizer tokenizer = Umt5();

        (int[] tokens, float[]? weights) =
            VideoRecipeUtils.TokenizeWeightedWan(tokenizer, PromptWeighting.Parse("a (red:1.5) fox"));

        Assert.NotNull(weights);
        Assert.Equal(tokenizer.MaxLength, tokens.Length);
        Assert.Equal(tokens.Length, weights!.Length);
        Assert.Contains(1.5f, weights);
    }

    /// <summary>EOS and every pad row weigh exactly 1. They are not part of the prompt, and blending them would
    /// pull the padding toward the empty encode along with the words — a whole-sequence drift that reads as the
    /// weighting being far too strong.</summary>
    [Fact]
    public void TheEosAndPadRowsAreNeverWeighted()
    {
        using T5Tokenizer tokenizer = Umt5();

        (int[] tokens, float[]? weights) =
            VideoRecipeUtils.TokenizeWeightedWan(tokenizer, PromptWeighting.Parse("(fox:2.0)"));

        Assert.NotNull(weights);
        int eos = Array.IndexOf(tokens, T5Tokenizer.EosTokenId);
        Assert.True(eos > 0, "The sequence must be terminated for the encoder to see its length.");
        for (int i = eos; i < tokens.Length; i++)
        {
            Assert.Equal(1f, weights![i]);
            Assert.Equal(i == eos ? T5Tokenizer.EosTokenId : T5Tokenizer.PadTokenId, tokens[i]);
        }
    }

    /// <summary>A prompt long enough to fill the window still terminates. Losing the EOS would leave the encoder
    /// reading padding as content, which is a worse failure than dropping the last word.</summary>
    [Fact]
    public void AnOverlongWeightedPromptStillTerminates()
    {
        using T5Tokenizer tokenizer = Umt5();
        string huge = "(fox:1.5) " + string.Join(' ', Enumerable.Repeat("snow", 2000));

        (int[] tokens, float[]? weights) =
            VideoRecipeUtils.TokenizeWeightedWan(tokenizer, PromptWeighting.Parse(huge));

        Assert.NotNull(weights);
        Assert.Equal(tokenizer.MaxLength, tokens.Length);
        Assert.Equal(T5Tokenizer.EosTokenId, tokens[^1]);
        Assert.Equal(1f, weights![^1]);
    }
}

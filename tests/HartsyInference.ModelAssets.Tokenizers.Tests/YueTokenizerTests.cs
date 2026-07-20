using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary>Asset-free tests for the YuE Stage-1 prompt text construction (the special-token ID resolution
/// and full encode need the real tokenizer.model, verified at runtime).</summary>
public sealed class YueTokenizerTests
{
    [Fact]
    public void BuildStage1PromptText_MatchesYueInferFormat()
    {
        string text = YueTokenizer.BuildStage1PromptText("emotional piano slow ballad sad female", "[verse]\nhello world");
        Assert.Equal(
            "Generate music from the given lyrics segment by segment.\n[Genre] emotional piano slow ballad sad female\n[verse]\nhello world",
            text);
    }

    [Fact]
    public void BuildStage1PromptText_TrimsGenreAndLyrics_AndToleratesNull()
    {
        Assert.Equal(
            "Generate music from the given lyrics segment by segment.\n[Genre] pop\n[chorus]",
            YueTokenizer.BuildStage1PromptText("  pop  ", "  [chorus]  "));
        Assert.Equal(
            "Generate music from the given lyrics segment by segment.\n[Genre] \n",
            YueTokenizer.BuildStage1PromptText(null!, null!));
    }
}

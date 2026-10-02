using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary>Asset-free tests for the YuE Stage-1 prompt text construction (the special-token ID resolution
/// and full encode need the real tokenizer.model, verified at runtime). The lyrics part is infer.py's
/// <c>split_lyrics</c>: each <c>[label]</c> section becomes <c>"[label]\n{text}\n\n"</c>, and the sections are joined
/// with one more newline.</summary>
public sealed class YueTokenizerTests
{
    [Fact]
    public void BuildStage1PromptText_MatchesYueInferFormat()
    {
        string text = YueTokenizer.BuildStage1PromptText("emotional piano slow ballad sad female", "[verse]\nhello world");
        Assert.Equal(
            "Generate music from the given lyrics segment by segment.\n[Genre] emotional piano slow ballad sad female\n[verse]\nhello world\n\n",
            text);
    }

    [Fact]
    public void BuildStage1PromptText_JoinsSectionsTheWayInferPyDoes()
    {
        string text = YueTokenizer.BuildStage1PromptText("rock", "[verse]\n  first line \n[chorus]\nsecond line\n");
        Assert.Equal(
            "Generate music from the given lyrics segment by segment.\n[Genre] rock\n[verse]\nfirst line\n\n\n[chorus]\nsecond line\n\n",
            text);
    }

    [Fact]
    public void BuildStage1PromptText_TrimsGenreAndLyrics_AndToleratesNull()
    {
        Assert.Equal(
            "Generate music from the given lyrics segment by segment.\n[Genre] pop\n[chorus]\n\n\n",
            YueTokenizer.BuildStage1PromptText("  pop  ", "  [chorus]  "));
        Assert.Equal(
            "Generate music from the given lyrics segment by segment.\n[Genre] \n",
            YueTokenizer.BuildStage1PromptText(null!, null!));
    }
}

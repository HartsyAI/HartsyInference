using HartsyInference.Audio.Frontends;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>What a voice should and should not read aloud from a chat model's reply.</summary>
public sealed class SpokenTextNormalizerTests
{
    [Fact]
    public void ThinkBlocks_AreRemoved_ClosedOrNot()
    {
        Assert.Equal("The answer is four.",
            SpokenTextNormalizer.ToSpeakable("<think>Two plus two... let me count.</think>The answer is four."));
        Assert.Equal("Sure.", SpokenTextNormalizer.ToSpeakable("Sure.<THINK>\nstill thinking\nacross lines"));
        Assert.Equal("", SpokenTextNormalizer.ToSpeakable("<think>only thoughts</think>"));
    }

    [Fact]
    public void Markdown_KeepsTheWordsAndDropsTheSyntax()
    {
        Assert.Equal("This is bold and italic and code and more and gone.",
            SpokenTextNormalizer.ToSpeakable("This is **bold** and _italic_ and `code` and *more* and ~~gone~~."));
        Assert.Equal("Strong and emphasis.", SpokenTextNormalizer.ToSpeakable("__Strong__ and *emphasis*."));
    }

    [Fact]
    public void FencedCode_IsNotReadAloud()
    {
        Assert.Equal("Run this: then check the output.",
            SpokenTextNormalizer.ToSpeakable("Run this:\n```bash\nls -la\n```\nthen check the output."));
    }

    [Fact]
    public void Digits_AndUnderscoredIdentifiers_AreLeftAlone()
    {
        Assert.Equal("Call 555-0100 at 3.5 pm about order 42.",
            SpokenTextNormalizer.ToSpeakable("Call 555-0100 at 3.5 pm about order 42."));
        Assert.Equal("Use snake_case_name here.", SpokenTextNormalizer.ToSpeakable("Use snake_case_name here."));
        Assert.Equal("2 * 3 = 6 and 4 * 5 = 20", SpokenTextNormalizer.ToSpeakable("2 * 3 = 6 and 4 * 5 = 20"));
        // A year opening a line is prose, not a list marker.
        Assert.Equal("1999. The year it happened.", SpokenTextNormalizer.ToSpeakable("1999. The year it happened."));
    }

}

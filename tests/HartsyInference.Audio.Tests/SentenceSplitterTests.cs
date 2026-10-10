using HartsyInference.Audio.Frontends;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Where a reply gets cut before it is spoken.
///
/// <para>Every failure here is audible rather than visible. Cutting inside "Dr. Chen" or "3.5 seconds" splits
/// one sentence into two separately-voiced halves with their own prosody and their own trailing silence, and
/// the seam lands in the middle of a phrase. Failing to cut is only slower. So the tests are weighted the way
/// the splitter is: a great deal of care about false boundaries, and one test that it finds the real ones.</para></summary>
public sealed class SentenceSplitterTests
{
    [Fact]
    public void PlainProse_SplitsAtEveryStop()
    {
        IReadOnlyList<string> parts = SentenceSplitter.Split(
            "The weather tomorrow is clear and mild. Rain arrives on Thursday evening. Bring a coat with you.");

        Assert.Equal(3, parts.Count);
        Assert.Equal("The weather tomorrow is clear and mild.", parts[0]);
        Assert.Equal("Rain arrives on Thursday evening.", parts[1]);
        Assert.Equal("Bring a coat with you.", parts[2]);
    }

    [Theory]
    [InlineData("Dr. Chen said the results were unremarkable and sent us home again.")]
    [InlineData("Bring a coat, an umbrella, boots, etc. and we will be ready for whatever arrives.")]
    public void ThingsThatLookLikeStopsButAreNot_StayInOnePiece(string text)
    {
        Assert.Single(SentenceSplitter.Split(text));
    }

    [Fact]
    public void ClosingQuotesStayWithTheSentenceTheyClose()
    {
        IReadOnlyList<string> parts = SentenceSplitter.Split(
            "She said \"the meeting is cancelled.\" Everyone went home for the afternoon.");

        Assert.Equal(2, parts.Count);
        Assert.EndsWith("\"", parts[0]);
        Assert.StartsWith("Everyone", parts[1]);
    }

    [Fact]
    public void EveryWordSurvivesTheSplit()
    {
        const string text = "Good morning. The forecast today calls for scattered clouds with a high of "
            + "seventy two degrees. You have three meetings on your calendar, the first at nine thirty. "
            + "Traffic on your usual route is moving normally.";

        IReadOnlyList<string> parts = SentenceSplitter.Split(text);

        Assert.True(parts.Count > 1, "a four-sentence passage came back as one piece");
        Assert.Equal(
            string.Join(" ", text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)),
            string.Join(" ", string.Join(" ", parts).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)));
    }

    [Theory]
    [InlineData(null)]
    public void NothingIn_NothingOut(string? text) => Assert.Empty(SentenceSplitter.Split(text));

}

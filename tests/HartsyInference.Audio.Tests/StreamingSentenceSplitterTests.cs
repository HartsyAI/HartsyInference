using HartsyInference.Audio.Frontends;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The streaming splitter must agree with <see cref="SentenceSplitter.Split"/> on the joined text for
/// every way the deltas can fall: mid-word, between "Dr" and its period, between "3." and the "5". A disagreement
/// is an audible seam in a spoken reply, which no other test would catch.</summary>
public sealed class StreamingSentenceSplitterTests
{
    private const string Tricky =
        "Dr. Chen said the results were unremarkable and sent us home. It measured 3.5 metres across, give or take. "
        + "J. R. R. Tolkien wrote it over the better part of two decades! What time is the meeting tomorrow? "
        + "Bring a coat, an umbrella, boots, etc. and we will be ready. Yes. The meeting was moved to three o'clock.";

    public static IEnumerable<object[]> Cuttings =>
    [
        [Tricky, 1],
        [Tricky, 2],
        [Tricky, 3],
        [Tricky, 7],
        [Tricky, 11],
        [Tricky, 1000],
        ["Hello world. The end", 4],
        ["No terminator anywhere in this reply at all", 5],
    ];

    [Theory]
    [MemberData(nameof(Cuttings))]
    public void Deltas_ProduceTheSameSentencesAsSplitOnTheJoinedText(string text, int step)
    {
        StreamingSentenceSplitter splitter = new();
        List<string> streamed = [];
        for (int i = 0; i < text.Length; i += step)
        {
            streamed.AddRange(splitter.Push(text.Substring(i, Math.Min(step, text.Length - i))));
        }
        string? tail = splitter.Flush();
        if (tail is not null)
        {
            streamed.Add(tail);
        }
        Assert.Equal(SentenceSplitter.Split(text), streamed);
    }

    [Fact]
    public void Push_NeverEmitsThePieceStillOpenAtTheEndOfTheBuffer()
    {
        StreamingSentenceSplitter splitter = new();
        Assert.Empty(splitter.Push("It costs 3."));
        Assert.Empty(splitter.Push("5 dollars in total, sir."));
        IReadOnlyList<string> completed = splitter.Push(" Then more text arrives here for you.");
        Assert.Equal(["It costs 3.5 dollars in total, sir."], completed);
        Assert.Equal("Then more text arrives here for you.", splitter.Flush());
    }

    [Fact]
    public void Push_HoldsAnAbbreviationUntilTheNextWordDecidesIt()
    {
        StreamingSentenceSplitter splitter = new();
        Assert.Empty(splitter.Push("The appointment is with Dr."));
        Assert.Empty(splitter.Push(" Chen at nine."));
        Assert.Equal(["The appointment is with Dr. Chen at nine."], splitter.Push(" Please arrive a little early."));
    }

    [Fact]
    public void Flush_ReturnsTheTailOnceAndNullWhenEmpty()
    {
        StreamingSentenceSplitter splitter = new();
        Assert.Null(splitter.Flush());
        splitter.Push("  a trailing fragment  ");
        Assert.Equal("a trailing fragment", splitter.Flush());
        Assert.Null(splitter.Flush());
        Assert.Equal(0, splitter.PendingLength);
    }

    [Fact]
    public void FirstSentence_MayBeShorterThanTheRest()
    {
        // "Yes." comes out alone; the rest splits under the normal minimum, and "Go." is the held tail.
        StreamingSentenceSplitter eager = new(firstSentenceMinLength: 1);
        Assert.Equal(["Yes.", "The meeting was moved to three o'clock."], eager.Push("Yes. The meeting was moved to three o'clock. Go."));
        Assert.Equal("Go.", eager.Flush());

        // Under the normal minimum "Yes." merges forward, and the merged sentence is complete once "Go." follows.
        StreamingSentenceSplitter normal = new();
        Assert.Equal(["Yes. The meeting was moved to three o'clock."], normal.Push("Yes. The meeting was moved to three o'clock. Go."));
        Assert.Equal("Go.", normal.Flush());
    }

    [Fact]
    public void FirstSentenceMinimum_AppliesToTheFirstSentenceOnly()
    {
        // With the small minimum applied to the whole buffer, "No." would come out alone too; it must merge forward.
        StreamingSentenceSplitter splitter = new(firstSentenceMinLength: 1);
        IReadOnlyList<string> completed = splitter.Push("Yes. No. The meeting was moved to three o'clock today. Fine.");
        Assert.Equal(["Yes.", "No. The meeting was moved to three o'clock today."], completed);
        Assert.Equal("Fine.", splitter.Flush());
    }

    [Fact]
    public void Reset_DropsPendingTextAndArmsTheFirstSentenceAgain()
    {
        StreamingSentenceSplitter splitter = new(firstSentenceMinLength: 1);
        splitter.Push("Yes. The rest of this");
        splitter.Reset();
        Assert.Equal(0, splitter.PendingLength);
        Assert.Equal(["Ok."], splitter.Push("Ok. And then a longer sentence follows it here."));
    }

    [Fact]
    public void Push_IgnoresNullAndEmptyDeltas()
    {
        StreamingSentenceSplitter splitter = new();
        Assert.Empty(splitter.Push(null));
        Assert.Empty(splitter.Push(""));
        Assert.Equal(0, splitter.PendingLength);
    }

    [Fact]
    public void SplitClauses_ShortSentence_ComesBackWhole()
    {
        Assert.Equal(["Short and sweet, really."], SentenceSplitter.SplitClauses("  Short and sweet, really.  ", 40));
        Assert.Empty(SentenceSplitter.SplitClauses("   ", 40));
        Assert.Empty(SentenceSplitter.SplitClauses(null, 40));
        Assert.Throws<ArgumentOutOfRangeException>(() => SentenceSplitter.SplitClauses("x", 0));
    }

    [Fact]
    public void SplitClauses_CutsAtTheLastClauseMarkBeforeTheLimit()
    {
        const string Long = "First we gather the ingredients, then we mix the batter, then we pour it into the tin, "
            + "and finally we bake it for forty minutes; after that it cools: patience is the hard part.";
        IReadOnlyList<string> pieces = SentenceSplitter.SplitClauses(Long, 60);
        Assert.True(pieces.Count >= 3);
        Assert.All(pieces, p => Assert.True(p.Length <= 60, $"\"{p}\" is over 60 chars"));
        Assert.Equal("First we gather the ingredients, then we mix the batter,", pieces[0]);
        Assert.Equal(Long, string.Join(" ", pieces));
    }

    [Fact]
    public void SplitClauses_DoesNotCutInsideANumber()
    {
        const string Text = "The total came to 1,000,000 dollars and change, which surprised everyone in the room that day.";
        IReadOnlyList<string> pieces = SentenceSplitter.SplitClauses(Text, 50);
        Assert.All(pieces, p => Assert.True(p.Length <= 50));
        Assert.Contains(pieces, p => p.Contains("1,000,000", StringComparison.Ordinal));
        Assert.Equal(Text, string.Join(" ", pieces));
    }

    [Fact]
    public void SplitClauses_FallsBackToASpaceThenToAHardCut()
    {
        const string NoClauses = "no clause marks anywhere in this sentence just words and words and words";
        IReadOnlyList<string> bySpace = SentenceSplitter.SplitClauses(NoClauses, 30);
        Assert.All(bySpace, p => Assert.True(p.Length <= 30));
        Assert.Equal(NoClauses, string.Join(" ", bySpace));
        Assert.All(bySpace, p => Assert.DoesNotContain("  ", p, StringComparison.Ordinal));

        const string OneWord = "supercalifragilisticexpialidocious";
        IReadOnlyList<string> hard = SentenceSplitter.SplitClauses(OneWord, 10);
        Assert.Equal(["supercalif", "ragilistic", "expialidoc", "ious"], hard);
    }
}

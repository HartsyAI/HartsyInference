using System.Text;
using HartsyInference.Audio.Text;
using Xunit;

namespace HartsyInference.Audio.Tests.Text;

/// <summary>Chunk boundaries for text-to-speech. Word loss or a chunk over budget is silent in the audio,
/// so the conservation test checks the full word sequence rather than a sample.</summary>
public sealed class TtsTextSegmenterTests
{
    [Fact]
    public void SentencesMergeUpToTheBudget()
    {
        IReadOnlyList<string> chunks = TtsTextSegmenter.Pack(
            "The weather is clear. Rain comes later. Bring a coat.", maxWords: 40);

        Assert.Single(chunks);
        Assert.Equal("The weather is clear. Rain comes later. Bring a coat.", chunks[0]);
    }

    [Fact]
    public void SentenceBoundaryIsKeptWhenTheNextSentenceDoesNotFit()
    {
        IReadOnlyList<string> chunks = TtsTextSegmenter.Pack(
            "One two three. Four five six. Seven eight nine.", maxWords: 6);

        Assert.Equal(2, chunks.Count);
        Assert.Equal("One two three. Four five six.", chunks[0]);
        Assert.Equal("Seven eight nine.", chunks[1]);
    }

    [Fact]
    public void MultiSentencePassageRespectsTheFortyWordCap()
    {
        const string text = "The first morning of the trip began before dawn with cold coffee and a long drive "
            + "toward the coast. Rain met us at the border, so we stopped for lunch in a small town. "
            + "The harbour was quiet. Gulls circled over the boats while the fishermen mended their nets. "
            + "By evening the clouds had cleared, and we walked along the pier until the lights came on.";

        IReadOnlyList<string> chunks = TtsTextSegmenter.Pack(text);

        Assert.True(chunks.Count > 1);
        Assert.All(chunks, chunk => Assert.True(WordsOf(chunk).Length <= 40));
        Assert.All(chunks, chunk => Assert.False(string.IsNullOrWhiteSpace(chunk)));
    }

    [Fact]
    public void SingleLongSentenceIsCutAtClauseBoundaries()
    {
        const string text = "We packed the car early, drove through the hills, stopped for tea, "
            + "watched the sunset over the lake, and arrived home after midnight";

        IReadOnlyList<string> chunks = TtsTextSegmenter.Pack(text, maxWords: 10);

        Assert.True(chunks.Count >= 2);
        Assert.All(chunks, chunk => Assert.True(WordsOf(chunk).Length <= 10));
        Assert.EndsWith("hills,", chunks[0]);
        Assert.Equal(WordsOf(text), chunks.SelectMany(chunk => WordsOf(chunk)).ToArray());
    }

    [Fact]
    public void ClauseLongerThanBudgetIsCutIntoExactWordPieces()
    {
        const string text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu";

        IReadOnlyList<string> chunks = TtsTextSegmenter.Pack(text, maxWords: 5);

        Assert.Equal(3, chunks.Count);
        Assert.Equal("alpha beta gamma delta epsilon", chunks[0]);
        Assert.Equal("zeta eta theta iota kappa", chunks[1]);
        Assert.Equal("lambda mu nu", chunks[2]);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("\n\t  \r\n")]
    public void EmptyAndWhitespaceInputYieldNothing(string text) => Assert.Empty(TtsTextSegmenter.Pack(text));

    [Fact]
    public void MaxWordsOfOneEmitsOneWordPerChunk()
    {
        IReadOnlyList<string> chunks = TtsTextSegmenter.Pack("Hello there. Friends, welcome!", maxWords: 1);

        Assert.Equal(new[] { "Hello", "there.", "Friends,", "welcome!" }, chunks);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-3)]
    public void MaxWordsBelowOneIsRejected(int maxWords)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => TtsTextSegmenter.Pack("Some text here.", maxWords));
    }

    [Fact]
    public void EveryWordAppearsOnceInOrderOnALongPassage()
    {
        string text = BuildPassage(176);

        IReadOnlyList<string> chunks = TtsTextSegmenter.Pack(text);

        Assert.All(chunks, chunk => Assert.True(WordsOf(chunk).Length <= 40));
        Assert.All(chunks, chunk => Assert.False(string.IsNullOrWhiteSpace(chunk)));
        Assert.Equal(WordsOf(text), chunks.SelectMany(chunk => WordsOf(chunk)).ToArray());
    }

    [Fact]
    public void UnicodeTextIsPreservedWordForWord()
    {
        const string text = "Grüße aus Zürich. 今日は晴れです、とても暖かい。Ça va très bien ; merci beaucoup.";

        IReadOnlyList<string> chunks = TtsTextSegmenter.Pack(text, maxWords: 4);

        Assert.All(chunks, chunk => Assert.True(WordsOf(chunk).Length <= 4));
        Assert.Equal(WordsOf(text), chunks.SelectMany(chunk => WordsOf(chunk)).ToArray());
    }

    private static string[] WordsOf(string text) => text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

    private static string BuildPassage(int wordCount)
    {
        StringBuilder builder = new();
        for (int i = 0; i < wordCount; i++)
        {
            if (i > 0)
            {
                builder.Append(' ');
            }
            builder.Append("Word").Append(i);
            if (i % 4 == 2)
            {
                builder.Append(',');
            }
            if (i % 9 == 8)
            {
                builder.Append('.');
            }
        }
        return builder.ToString();
    }
}

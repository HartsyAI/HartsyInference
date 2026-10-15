using HartsyInference.Audio.Models.F5Tts;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>F5 long-text chunking. Expected chunks are upstream F5-TTS <c>chunk_text</c> output, run on CPU.</summary>
public sealed class F5TextChunkerTests
{
    private const string Paragraph = "Our water bill arrived late this month, so I called the office to ask why. "
        + "The clerk said a meter reading was missed in March. "
        + "She promised to fix the record and send a corrected statement within two weeks. "
        + "I thanked her, hung up, and wrote the date on my calendar so I would remember to check. "
        + "Last time this happened it took three phone calls, so I am hoping this one goes faster.";

    private const string SweepSentence = "After the meeting ended, the team gathered in the kitchen to talk about "
        + "the new product launch, the budget, and the schedule for the coming quarter.";

    private const string NoPunctuation = "the quick brown fox jumps over the lazy dog while the farmer counts "
        + "the cows in the far field and the children run home before dark";

    private const string Cjk = "我们的水费本月迟到了。我打电话到办公室询问原因。职员说三月份漏记了一次抄表。"
        + "她答应修正记录，并在两周内寄出更正的账单。我谢过她，挂断电话，并在日历上写下日期，以免忘记再查。";

    [Theory]
    [InlineData(294)]
    [InlineData(327)]
    [InlineData(344)]
    public void Chunk_ParagraphSplitsOnceAtTheLastSentenceThatFits(int maxChars)
        => Assert.Equal(
            new[]
            {
                "Our water bill arrived late this month, so I called the office to ask why. "
                + "The clerk said a meter reading was missed in March. "
                + "She promised to fix the record and send a corrected statement within two weeks. "
                + "I thanked her, hung up, and wrote the date on my calendar so I would remember to check.",
                "Last time this happened it took three phone calls, so I am hoping this one goes faster.",
            },
            F5TextChunker.Chunk(Paragraph, maxChars));

    [Fact]
    public void Chunk_SplitsALongSentenceAtItsCommas()
        => Assert.Equal(
            new[]
            {
                "After the meeting ended,",
                "the team gathered in the kitchen to talk about the new product launch,",
                "the budget, and the schedule for the coming quarter.",
            },
            F5TextChunker.Chunk(SweepSentence, 80));

    [Fact]
    public void Chunk_SentenceWithinTheLimitIsOneChunk()
        => Assert.Equal(new[] { SweepSentence }, F5TextChunker.Chunk(SweepSentence, 327));

    [Fact]
    public void Chunk_KeepsASentenceWithoutPunctuationWhole()
        => Assert.Equal(new[] { NoPunctuation }, F5TextChunker.Chunk(NoPunctuation, 80));

    [Fact]
    public void Chunk_SplitsCjkAfterFullWidthPunctuation()
        => Assert.Equal(
            new[]
            {
                "我们的水费本月迟到了。",
                "我打电话到办公室询问原因。",
                "职员说三月份漏记了一次抄表。",
                "她答应修正记录，",
                "并在两周内寄出更正的账单。我谢过她，",
                "挂断电话，并在日历上写下日期，",
                "以免忘记再查。",
            },
            F5TextChunker.Chunk(Cjk, 60));

    [Fact]
    public void Chunk_CollapsesWhitespaceAfterSentencePunctuation()
        => Assert.Equal(
            new[] { "First  one. Second one! Third?" },
            F5TextChunker.Chunk("First  one.   Second one!  Third?", 100));

    [Theory]
    [InlineData(1.0, 328)]
    [InlineData(0.5, 164)]
    public void MaxChars_MatchesUpstreamInferProcess(double speed, int expected)
        => Assert.Equal(expected, F5TextChunker.MaxChars(65, 341 * 256 / 24000.0, speed));

    [Fact]
    public void MaxChars_RejectsAReferenceWithoutDuration()
        => Assert.Throws<ArgumentOutOfRangeException>(() => F5TextChunker.MaxChars(65, 0.0, 1.0));

    [Fact]
    public void CrossFade_LinearOverlapOfThreeSamples()
        => Assert.Equal(
            new[] { 1f, 2f, 3f, 4f, 5f, 6f, 13.5f, 30f, 40f, 50f, 60f, 70f, 80f },
            F5TextChunker.CrossFade(
                new[]
                {
                    new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f },
                    new[] { 10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f },
                },
                3));

    [Fact]
    public void CrossFade_ClampsTheOverlapToTheShorterWave()
        => Assert.Equal(
            new[] { 1f, 11f, 30f, 40f, 50f, 60f, 70f, 80f },
            F5TextChunker.CrossFade(
                new[]
                {
                    new[] { 1f, 2f, 3f },
                    new[] { 10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f },
                },
                5));

    [Fact]
    public void CrossFade_OneSampleOverlapKeepsTheEarlierWaveAtTheJoin()
        => Assert.Equal(
            new[] { 5f, 6f, 8f },
            F5TextChunker.CrossFade(new[] { new[] { 5f, 6f }, new[] { 7f, 8f } }, 1));

    [Fact]
    public void CrossFade_ZeroOverlapConcatenates()
        => Assert.Equal(
            new[] { 1f, 2f, 10f, 20f },
            F5TextChunker.CrossFade(new[] { new[] { 1f, 2f }, new[] { 10f, 20f } }, 0));

    [Fact]
    public void CrossFade_UsesTheUpstreamOverlapAtTwentyFourKilohertz()
    {
        int overlap = F5TextChunker.CrossFadeSamples(24_000);
        Assert.Equal(3_600, overlap);
        float[][] waves = { new float[4_000], new float[4_000] };
        Assert.Equal(4_000 + 4_000 - overlap, F5TextChunker.CrossFade(waves, overlap).Length);
    }
}

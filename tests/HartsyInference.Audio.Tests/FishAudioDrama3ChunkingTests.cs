using HartsyInference.Audio.Models.FishAudio;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Drama 3's long-text chunking and chunk history, checked against the cases of fish-speech's
/// <c>tests/test_drama3_chunking.py</c>.</summary>
public sealed class FishAudioDrama3ChunkingTests
{
    [Fact]
    public void ShortText_IsOneUnchangedChunk_AndBlankTextNone()
    {
        Assert.Equal(["  Hello there.  "], FishAudioDrama3Chunking.SplitByLength("  Hello there.  ", 300));
        Assert.Empty(FishAudioDrama3Chunking.SplitByLength(" \n ", 300));
    }

    [Fact]
    public void Splits_PreferSentenceEnds()
    {
        string text = string.Join(" ", Enumerable.Range(0, 40).Select(i => $"Sentence number {i} is here."));
        List<string> chunks = FishAudioDrama3Chunking.SplitByLength(text, 100);
        Assert.True(chunks.Count > 1);
        Assert.All(chunks, chunk =>
        {
            Assert.True(FishAudioDrama3Chunking.Utf8Length(chunk) <= 100);
            Assert.EndsWith(".", chunk);
        });
        Assert.Equal(text, string.Join(" ", chunks));

        string chinese = string.Concat(Enumerable.Repeat("今天天气很好，我们去公园散步。", 20));
        chunks = FishAudioDrama3Chunking.SplitByLength(chinese, 100);
        Assert.All(chunks, chunk =>
        {
            Assert.True(FishAudioDrama3Chunking.Utf8Length(chunk) <= 100);
            Assert.EndsWith("。", chunk);
        });
        Assert.Equal(chinese, string.Concat(chunks));
    }

    [Fact]
    public void Splits_FallBackToClausesSpacesThenLength()
    {
        string clauses = string.Join("，", Enumerable.Repeat("这是一个很长的分句", 20));
        List<string> chunks = FishAudioDrama3Chunking.SplitByLength(clauses, 60);
        Assert.All(chunks[..^1], chunk => Assert.EndsWith("，", chunk));
        Assert.Equal(clauses, string.Concat(chunks));

        string words = string.Join(" ", Enumerable.Repeat("word", 100));
        chunks = FishAudioDrama3Chunking.SplitByLength(words, 50);
        Assert.All(chunks, chunk => Assert.True(FishAudioDrama3Chunking.Utf8Length(chunk) <= 50));
        Assert.Equal(words, string.Join(" ", chunks));

        string unbroken = new('字', 100);
        chunks = FishAudioDrama3Chunking.SplitByLength(unbroken, 50);
        Assert.All(chunks, chunk => Assert.True(FishAudioDrama3Chunking.Utf8Length(chunk) <= 50));
        Assert.Equal(unbroken, string.Concat(chunks));
    }

    [Fact]
    public void Splits_KeepDecimalsAndProtectedSpansWhole()
    {
        string whisper = "<whisper>" + string.Concat(Enumerable.Repeat("secret words. ", 10)) + "</whisper>";
        string text = string.Concat(Enumerable.Repeat("Pi is 3.14159 roughly. ", 10)) + whisper + " [laughing] done.";
        List<string> chunks = FishAudioDrama3Chunking.SplitByLength(text, 60);
        Assert.Contains(whisper, chunks);
        Assert.True(FishAudioDrama3Chunking.Utf8Length(whisper) > 60);
        Assert.DoesNotContain(chunks, chunk => chunk.EndsWith("3.", StringComparison.Ordinal));
        Assert.Contains(chunks, chunk => chunk.Contains("[laughing]", StringComparison.Ordinal));
    }

    [Fact]
    public void SpeakerSegments_StartUntaggedTextAsSpeakerZero()
    {
        Assert.Equal([("0", "intro "), ("1", "Hi"), ("2", " Yo")],
            FishAudioDrama3Chunking.ParseSpeakerSegments("intro <|speaker:1|>Hi<|speaker:2|> Yo"));
    }

    [Fact]
    public void ANewChunk_StartsAtTheFourthSpeaker()
    {
        string text = "<|speaker:0|>A1<|speaker:1|>B1<|speaker:0|>A2<|speaker:2|>C1<|speaker:3|>D1";
        Assert.Equal(
            ["<|speaker:0|> A1\n<|speaker:1|> B1\n<|speaker:0|> A2\n<|speaker:2|> C1", "<|speaker:3|> D1"],
            FishAudioDrama3Chunking.Split(text, maxBytes: 300, maxSpeakers: 3));
    }

    [Fact]
    public void LongSegments_AreCutAndKeepTheirTag()
    {
        string longText = string.Concat(Enumerable.Repeat("This sentence is long enough. ", 10));
        List<string> chunks = FishAudioDrama3Chunking.Split($"<|speaker:0|>{longText}<|speaker:0|>Short.", 100);
        Assert.True(chunks.Count > 2);
        Assert.All(chunks.SelectMany(chunk => chunk.Split('\n')), segment => Assert.StartsWith("<|speaker:0|> ", segment));
        Assert.EndsWith("Short.", chunks[^1]);
    }

    [Fact]
    public void UntaggedText_IsSplitByLength()
    {
        Assert.Equal(["Hello."], FishAudioDrama3Chunking.Split("Hello."));
        Assert.Empty(FishAudioDrama3Chunking.Split("  "));
    }

    [Fact]
    public void MultiSpeakerScripts_ChunkWithinThreeSpeakers_AndPassValidation()
    {
        string script = string.Concat(Enumerable.Range(0, 30).Select(i => $"<|speaker:{i % 5}|>Line {i} of the scene goes here. "));
        List<string> chunks = FishAudioDrama3Chunking.Split(script);
        Assert.True(chunks.Count > 1);
        foreach (string chunk in chunks)
        {
            FishAudioDrama3Text.Validate(chunk, "chunk");
            int speakers = FishAudioDrama3Text.SpeakerTag().Matches(chunk).Select(m => m.Groups[1].Value).Distinct().Count();
            Assert.True(speakers <= 3);
        }
    }

    [Fact]
    public void History_PinsTheFirstChunk_AndKeepsTheLatest()
    {
        FishAudioDrama3History history = new(recentTurns: 1);
        Assert.Equal([[]], history.Candidates().Select(turns => turns.Select(t => t.Text).ToArray()));
        for (int i = 0; i < 3; i++) history.Push(new FishAudioDrama3Turn($"chunk {i}", new int[2, 1]));
        Assert.Equal([["chunk 0", "chunk 2"], ["chunk 0"], []], history.Candidates().Select(turns => turns.Select(t => t.Text).ToArray()));
    }
}

using HartsyInference.Cli.Dispatch;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>Unit coverage for <see cref="GenerationDispatch.FormatTimestamps"/> -- the fix for <c>hartsy transcribe
/// --timestamps</c> printing only a word count ("segments N") and never the actual timestamps, so the flag visibly
/// did nothing to the transcript itself.</summary>
public sealed class TranscribeTimestampFormattingTests
{
    [Fact]
    public void OneWord_FormatsAsASingleBracketedLine()
    {
        WordSegment[] words = [new() { Word = "Hello", Start = 0.0, End = 1.5 }];

        string result = GenerationDispatch.FormatTimestamps(words);

        Assert.Equal("[0.000s --> 1.500s]  Hello", result);
    }

    [Fact]
    public void MultipleWords_OneLinePerWord_InOrder()
    {
        WordSegment[] words =
        [
            new() { Word = "Hello", Start = 0.0, End = 0.4 },
            new() { Word = "world", Start = 0.4, End = 0.9 },
        ];

        string result = GenerationDispatch.FormatTimestamps(words);

        Assert.Equal("[0.000s --> 0.400s]  Hello\n[0.400s --> 0.900s]  world", result);
    }

    [Fact]
    public void SegmentGranularity_OneWholeSentenceAsOneEntry_StillFormats()
    {
        // The engine's own TranscriptResult doc: granularity may be segment-level, not per-word, depending on
        // what the model produced -- a single "word" spanning the whole utterance must format the same way.
        WordSegment[] words = [new() { Word = "The quick brown fox.", Start = 0.0, End = 10.68 }];

        string result = GenerationDispatch.FormatTimestamps(words);

        Assert.Equal("[0.000s --> 10.680s]  The quick brown fox.", result);
    }

    [Fact]
    public void NoTrailingNewline()
    {
        WordSegment[] words = [new() { Word = "Hi", Start = 0.0, End = 0.2 }];

        string result = GenerationDispatch.FormatTimestamps(words);

        Assert.False(result.EndsWith('\n'), "the printed transcript shouldn't carry a trailing blank line.");
    }

    [Fact]
    public void WordsAreTrimmedAndWhitespaceOnlyWordsSkipped()
    {
        WordSegment[] words = [new() { Word = " Hello", Start = 0.0, End = 0.4 }, new() { Word = "  ", Start = 0.4, End = 0.5 }];

        string result = GenerationDispatch.FormatTimestamps(words);

        Assert.Equal("[0.000s --> 0.400s]  Hello", result);
    }
}

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
}

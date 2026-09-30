using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.OutputParsing;
using Xunit;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>The streaming parser against completions parsed by upstream encoding.py, at every split point.</summary>
public sealed class DeepSeekV41OutputParserReferenceTests
{
    public static IEnumerable<object[]> OkCases() => ParserCase.Names(ok: true);

    public static IEnumerable<object[]> ErrorCases() => ParserCase.Names(ok: false);

    public static IEnumerable<object[]> AllCases() => ParserCase.Names();

    [Fact]
    public void FixtureHasBothKindsOfCase()
    {
        Assert.True(ParserCase.All.Count(c => c.Ok) >= 20);
        Assert.True(ParserCase.All.Count(c => !c.Ok) >= 10);
    }

    [Theory]
    [MemberData(nameof(OkCases))]
    public void EveryTwoWaySplitEqualsReference(string name)
    {
        ParserCase c = ParserCase.ByName(name);
        int length = System.Text.Encoding.UTF8.GetByteCount(c.Text);
        for (int cut = 0; cut <= length; cut++)
        {
            PieceTokenizer tok = new();
            ParsedAssistant got = ReplayedTurn.Run(
                ParserTestHelpers.NewParser(tok, c.Thinking), tok.SplitBytes(c.Text, cut), out ReplayedTurn replay);
            ParserTestHelpers.AssertMatches(c, got, replay, $"cut@{cut}");
        }
    }

    [Theory]
    [MemberData(nameof(OkCases))]
    public void SingleBytePiecesEqualReference(string name)
    {
        ParserCase c = ParserCase.ByName(name);
        PieceTokenizer tok = new();
        int length = System.Text.Encoding.UTF8.GetByteCount(c.Text);
        int[] cuts = Enumerable.Range(1, length - 1).ToArray();
        ParsedAssistant got = ReplayedTurn.Run(ParserTestHelpers.NewParser(tok, c.Thinking), tok.SplitBytes(c.Text, cuts), out ReplayedTurn replay);
        ParserTestHelpers.AssertMatches(c, got, replay, "single-byte");
    }

    [Theory]
    [MemberData(nameof(OkCases))]
    public void ControlLiteralsAsSpecialIdsWithRandomSegmentationEqualReference(string name)
    {
        ParserCase c = ParserCase.ByName(name);
        for (int seed = 0; seed < 60; seed++)
        {
            PieceTokenizer tok = new();
            int[] ids = tok.RandomWithSpecials(c.Text, new Random(seed), 1 + seed % 9);
            ParsedAssistant got = ReplayedTurn.Run(ParserTestHelpers.NewParser(tok, c.Thinking), ids, out ReplayedTurn replay);
            ParserTestHelpers.AssertMatches(c, got, replay, $"seed{seed}");
            Assert.True(got.Completed);
        }
    }

    [Theory]
    [MemberData(nameof(ErrorCases))]
    public void MalformedCompletionsAreFlaggedAtEverySplitWithoutThrowing(string name)
    {
        ParserCase c = ParserCase.ByName(name);
        int length = System.Text.Encoding.UTF8.GetByteCount(c.Text);
        PieceTokenizer oneShotTok = new();
        ParsedAssistant oneShot = ReplayedTurn.Run(ParserTestHelpers.NewParser(oneShotTok, c.Thinking), oneShotTok.SplitBytes(c.Text), out _);
        Assert.True(oneShot.Malformed, $"{name}: one-shot parse not flagged");
        for (int cut = 0; cut <= length; cut++)
        {
            PieceTokenizer tok = new();
            ParsedAssistant got = ReplayedTurn.Run(
                ParserTestHelpers.NewParser(tok, c.Thinking), tok.SplitBytes(c.Text, cut), out ReplayedTurn replay);
            Assert.True(got.Malformed, $"{name} cut@{cut}: not flagged");
            Assert.True(replay.Malformed > 0, $"{name} cut@{cut}: no Malformed event");
            Assert.Equal(oneShot.Reasoning, got.Reasoning);
            Assert.Equal(oneShot.Content, got.Content);
            Assert.Equal(oneShot.ToolCalls.Count, got.ToolCalls.Count);
            Assert.Equal(1, replay.Stops);
        }
    }

    [Theory]
    [MemberData(nameof(AllCases))]
    public void EveryRandomSegmentationOfAnyCompletionNeverThrows(string name)
    {
        ParserCase c = ParserCase.ByName(name);
        for (int seed = 100; seed < 130; seed++)
        {
            PieceTokenizer tok = new();
            ParsedAssistant got = ReplayedTurn.Run(ParserTestHelpers.NewParser(tok, c.Thinking),
                tok.RandomWithSpecials(c.Text, new Random(seed), 1 + seed % 7), out ReplayedTurn replay);
            ParserTestHelpers.AssertReplayMatches(got, replay, $"{name} seed{seed}");
        }
    }

    [Theory]
    [MemberData(nameof(OkCases))]
    public void EncodingTheParsedTurnReproducesThePrompt(string name)
    {
        ParserCase c = ParserCase.ByName(name);
        if (c.Prompt is null) return;
        PieceTokenizer tok = new();
        ParsedAssistant got = ReplayedTurn.Run(ParserTestHelpers.NewParser(tok, c.Thinking), tok.SplitBytes(c.Text, 7, 41), out _);
        List<ChatToolCall> calls = got.ToolCalls
            .Select((t, i) => new ChatToolCall($"c{i}", t.Name, t.ArgumentsJson) { Namespace = t.Namespace }).ToList();
        ChatMessage assistant = new ChatMessage("assistant", got.Content)
        {
            ReasoningContent = c.Thinking ? got.Reasoning : null,
            ToolCalls = calls.Count > 0 ? calls : null,
        };
        string rendered = DeepSeekV41Encoder.RenderText(
            [ChatMessage.User("q"), assistant],
            new EncodeOptions { Thinking = c.Thinking, DropThinking = false });
        Assert.Equal(c.Prompt, rendered);
    }
}

using HartsyInference.LLM.OutputParsing;
using Xunit;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>Hand-checked event sequences: what a streaming client sees, when, and how broken output is handled.</summary>
public sealed class DeepSeekV41OutputParserBehaviorTests
{
    private const string Call = "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"ns::f\">\n" +
        "<｜DSML｜ parameter name=\"a\" string=\"true\">x\"y</｜DSML｜ parameter>\n" +
        "<｜DSML｜ parameter name=\"n\" string=\"false\">5</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";

    private static List<ParsedEvent> Events(bool thinking, string text)
    {
        PieceTokenizer tok = new();
        DeepSeekV41OutputParser parser = ParserTestHelpers.NewParser(tok, thinking);
        List<ParsedEvent> events = [];
        foreach (int id in tok.RandomWithSpecials(text, new Random(1), 3)) parser.Push(id, events.Add);
        parser.Finish(events.Add);
        return events;
    }

    private static string Join(IEnumerable<ParsedEvent> events, ParsedEventKind kind) =>
        string.Concat(events.Where(e => e.Kind == kind).Select(e => e.Text));

    [Fact]
    public void ThinkingTurnStreamsReasoningThenContentThenCallThenStop()
    {
        List<ParsedEvent> events = Events(true, "hmm</think>Sure." + Call + PieceTokenizer.Eos);
        Assert.Equal("hmm", Join(events, ParsedEventKind.ReasoningDelta));
        Assert.Equal("Sure.", Join(events, ParsedEventKind.ContentDelta));
        ParsedEvent begin = Assert.Single(events, e => e.Kind == ParsedEventKind.ToolCallBegin);
        Assert.Equal(("f", "ns", 0), (begin.Text, begin.Namespace, begin.ToolCallIndex));
        Assert.Equal("{\"a\": \"x\\\"y\", \"n\": 5}", Join(events, ParsedEventKind.ToolCallArgsDelta));
        Assert.Equal(ParsedEventKind.ToolCallEnd, events[^2].Kind);
        Assert.Equal(ParsedEventKind.Stop, events[^1].Kind);
        Assert.DoesNotContain(events, e => e.Kind == ParsedEventKind.Malformed);
    }

    [Fact]
    public void ThinkingDoesNotLeakPartialMarkerText()
    {
        PieceTokenizer tok = new();
        DeepSeekV41OutputParser parser = ParserTestHelpers.NewParser(tok, false);
        List<ParsedEvent> events = [];
        foreach (int id in tok.SplitBytes("Hi\n\n<｜DSML｜ ca", 2, 3, 4, 5, 6)) parser.Push(id, events.Add);
        Assert.Equal("Hi", Join(events, ParsedEventKind.ContentDelta));
    }

    [Fact]
    public void TruncationInTextSectionsIsCompletedNotMalformed()
    {
        PieceTokenizer tok = new();
        DeepSeekV41OutputParser parser = ParserTestHelpers.NewParser(tok, true);
        ParsedAssistant got = ReplayedTurn.Run(parser, tok.SplitBytes("still thinking\n\n<"), out ReplayedTurn replay);
        Assert.Equal("still thinking\n\n<", got.Reasoning);
        Assert.False(got.Malformed);
        Assert.Equal(1, replay.Stops);
    }

    [Fact]
    public void TruncatedToolCallIsMalformedAndDropped()
    {
        string cut = Call[..(Call.IndexOf("</｜DSML｜ invoke", StringComparison.Ordinal) - 10)];
        PieceTokenizer tok = new();
        ParsedAssistant got = ReplayedTurn.Run(ParserTestHelpers.NewParser(tok, false), tok.SplitBytes(cut), out ReplayedTurn replay);
        Assert.True(got.Malformed);
        Assert.Empty(got.ToolCalls);
        Assert.Contains(replay.Calls, c => !c.Ended);
        Assert.Equal([0], replay.Aborted);
    }

    [Fact]
    public void FaultedToolBlockSwallowsTheRestButStillStops()
    {
        List<ParsedEvent> events = Events(false, "a\n\n<｜DSML｜ calls>junk more junk" + PieceTokenizer.Eos + "ignored");
        Assert.Contains(events, e => e.Kind == ParsedEventKind.Malformed);
        Assert.Equal(ParsedEventKind.Stop, events[^1].Kind);
        Assert.Single(events, e => e.Kind == ParsedEventKind.Stop);
    }

    [Fact]
    public void FinishIsIdempotent()
    {
        PieceTokenizer tok = new();
        DeepSeekV41OutputParser parser = ParserTestHelpers.NewParser(tok, false);
        List<ParsedEvent> events = [];
        parser.Finish(events.Add);
        parser.Finish(events.Add);
        Assert.Single(events);
    }

    [Fact]
    public void TokenizerWithoutThePinnedSpecialsIsRejected()
    {
        Assert.Throws<InvalidOperationException>(() => new DeepSeekV41OutputParser(new NoSpecialsTokenizer(), OutputParserState.Content));
    }

}

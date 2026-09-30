using HartsyInference.LLM.OutputParsing;
using Xunit;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>Non-structured templates must stream exactly what a plain decode of the ids produces.</summary>
public sealed class PassthroughOutputParserTests
{
    [Fact]
    public void ContentEqualsPlainDecodeAndControlTokensAreSkipped()
    {
        PieceTokenizer tok = new();
        string text = "Hello <think>ignored-tag-text</think> wörld 日本語 😀 <｜DSML｜ calls> done";
        int[] ids = tok.RandomWithSpecials(text, new Random(5), 4);
        PassthroughOutputParser parser = new(tok);
        List<ParsedEvent> events = [];
        foreach (int id in ids) parser.Push(id, events.Add);
        parser.Finish(events.Add);
        Assert.Equal(tok.Decode(ids), string.Concat(events.Where(e => e.Kind == ParsedEventKind.ContentDelta).Select(e => e.Text)));
        Assert.Equal(tok.Decode(ids), parser.Result.Content);
        Assert.Equal(ParsedEventKind.Stop, events[^1].Kind);
        Assert.All(events.SkipLast(1), e => Assert.Equal(ParsedEventKind.ContentDelta, e.Kind));
        Assert.True(parser.Result.Completed);
    }

    [Fact]
    public void ChunkBoundariesFollowTokensAndNeverSplitACharacter()
    {
        PieceTokenizer tok = new();
        PassthroughOutputParser parser = new(tok);
        List<ParsedEvent> events = [];
        int[] ids = tok.SplitBytes("aé", 1, 2);
        parser.Push(ids[0], events.Add);
        parser.Push(ids[1], events.Add);
        Assert.Equal(["a"], events.Select(e => e.Text));
        parser.Push(ids[2], events.Add);
        Assert.Equal(["a", "é"], events.Select(e => e.Text));
    }

    [Fact]
    public void UnfinishedSequenceFlushesAsReplacementCharacter()
    {
        PieceTokenizer tok = new();
        PassthroughOutputParser parser = new(tok);
        List<ParsedEvent> events = [];
        parser.Push(tok.Add([0xC3]), events.Add);
        parser.Finish(events.Add);
        Assert.Equal("�", parser.Result.Content);
    }
}

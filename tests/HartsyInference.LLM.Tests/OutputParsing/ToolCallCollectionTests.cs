using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.OutputParsing;
using Xunit;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>Every completed call is kept in order, so a non-streaming reply can report all of them, not just the last.</summary>
public sealed class ToolCallCollectionTests
{
    private static void Call(ParsedEventTranslator translator, int index, string name, string args)
    {
        translator.Handle(new ParsedEvent(ParsedEventKind.ToolCallBegin, name, index));
        translator.Handle(new ParsedEvent(ParsedEventKind.ToolCallArgsDelta, args, index));
        translator.Handle(new ParsedEvent(ParsedEventKind.ToolCallEnd, null, index));
    }

    [Fact]
    public void TranslatorCollectsEveryCompletedCallWithRequestScopedIds()
    {
        ParsedEventTranslator translator = new(_ => { }, requestId: 9);
        Call(translator, 0, "get_time", "{}");
        Call(translator, 1, "hang_up", "{\"reason\":\"done\"}");
        Assert.Equal(["get_time", "hang_up"], translator.Calls.Select(c => c.Name));
        Assert.Equal(["call_9_0", "call_9_1"], translator.Calls.Select(c => c.Id));
        Assert.Equal("{\"reason\":\"done\"}", translator.Calls[1].Arguments);
    }

    [Fact]
    public void AbortedCallIsNotCollected()
    {
        ParsedEventTranslator translator = new(_ => { }, requestId: 1);
        translator.Handle(new ParsedEvent(ParsedEventKind.ToolCallBegin, "get_time", 0));
        translator.Handle(new ParsedEvent(ParsedEventKind.ToolCallAbort, null, 0));
        Assert.Empty(translator.Calls);
    }

    [Fact]
    public void FilterSinkListsEveryCompletedCallInOrder()
    {
        // A filter that completes one call on each of two deltas; the sink keeps both and exposes the last as ToolCall.
        TextFilterResult First() => new("", new NativeToolCall { Id = "a", Name = "one" });
        TextFilterResult Second() => new("", new NativeToolCall { Id = "b", Name = "two" }, Stop: true);
        Queue<Func<TextFilterResult>> script = new([First, Second]);
        ScriptedFilter filter = new(script);
        TextFilterSink sink = new(filter, _ => { }, () => { });
        sink.Handle(new TextChunk { Kind = TextChunkKind.Chunk, Text = "x" });
        sink.Handle(new TextChunk { Kind = TextChunkKind.Chunk, Text = "y" });
        Assert.Equal(["a", "b"], sink.ToolCalls.Select(c => c.Id));
        Assert.Equal("b", sink.ToolCall!.Id);
    }

    private sealed class ScriptedFilter(Queue<Func<TextFilterResult>> script) : ITextStreamFilter
    {
        public TextFilterResult OnDelta(string delta) => script.Dequeue()();
        public TextFilterResult OnEnd() => TextFilterResult.Empty;
    }
}

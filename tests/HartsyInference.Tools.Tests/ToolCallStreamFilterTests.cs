using HartsyInference.Engine;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tools.Parsing;
using HartsyInference.Tools.Tests.Parsing;
using Xunit;

namespace HartsyInference.Tools.Tests;

/// <summary>The filter driven through the engine's own <see cref="TextFilterSink"/>, the way <c>TextService.RunText</c> wires it: content deltas arrive in random pieces, a completed call becomes a <see cref="TextChunkKind.NativeToolCall"/> chunk and (by default) a stop relay, plain text is untouched, and the seam's one-call-per-delta limit is bridged by queuing.</summary>
public sealed class ToolCallStreamFilterTests
{
    private const string Completion = "Sure. <tool_call>{\"name\": \"hang_up\", \"arguments\": {\"reason\": \"done\"}}</tool_call> trailing";

    private static (List<TextChunk> Chunks, TextFilterSink Sink, bool StopRequested) Drive(ToolCallStreamFilter filter, string completion, int seed)
    {
        List<TextChunk> chunks = [];
        bool stopRequested = false;
        TextFilterSink sink = new(filter, chunks.Add, () => stopRequested = true);
        foreach (string piece in ParserDriver.Pieces(completion, new Random(seed), 4))
        {
            sink.Handle(new TextChunk { Kind = TextChunkKind.Chunk, Text = piece });
            if (sink.Stopped) break;
        }
        if (!sink.Stopped) sink.End();
        return (chunks, sink, stopRequested);
    }

    [Theory]
    [InlineData(1)]
    public void CompletedCallEmitsANativeToolCallChunkAndStopsGeneration(int seed)
    {
        (List<TextChunk> chunks, TextFilterSink sink, bool stopRequested) = Drive(new ToolCallStreamFilter(), Completion, seed);
        Assert.True(sink.Stopped);
        Assert.True(stopRequested);
        Assert.Equal("Sure. ", sink.Text);
        Assert.Equal("Sure. ", string.Concat(chunks.Where(c => c.Kind == TextChunkKind.Chunk).Select(c => c.Text)));
        TextChunk call = Assert.Single(chunks, c => c.Kind == TextChunkKind.NativeToolCall);
        Assert.Equal("hang_up", call.ToolCall!.Name);
        Assert.Equal("{\"reason\": \"done\"}", call.ToolCall.Arguments);
        Assert.Equal("call_0", call.ToolCall.Id);
        Assert.Equal(0, call.ToolCallIndex);
        Assert.Same(call.ToolCall, sink.ToolCall);
        Assert.Equal(TextChunkKind.NativeToolCall, chunks[^1].Kind);
    }

    [Fact]
    public void WithoutStopAfterFirstCallEveryCallIsEmittedAndTextBetweenIsForwarded()
    {
        const string text = "<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call> and <tool_call>{\"name\": \"b\", \"arguments\": {}}</tool_call> done";
        ToolCallStreamFilter filter = new(ToolCallFormat.Hermes, stopAfterFirstCall: false);
        (List<TextChunk> chunks, TextFilterSink sink, bool stopRequested) = Drive(filter, text, 5);
        Assert.False(sink.Stopped);
        Assert.False(stopRequested);
        Assert.Equal("and done", sink.Text);
        Assert.Equal(["a", "b"], chunks.Where(c => c.Kind == TextChunkKind.NativeToolCall).Select(c => c.ToolCall!.Name));
        Assert.Equal([0, 1], chunks.Where(c => c.Kind == TextChunkKind.NativeToolCall).Select(c => c.ToolCallIndex!.Value));
        Assert.Equal(2, filter.Calls.Count);
    }

    [Fact]
    public void PlainTextLeavesTheChunkSequenceUnchanged()
    {
        const string text = "plain answer with { braces } and <tags>, no tools 😀";
        (List<TextChunk> chunks, TextFilterSink sink, bool stopRequested) = Drive(new ToolCallStreamFilter(), text, 8);
        Assert.False(sink.Stopped);
        Assert.False(stopRequested);
        Assert.Null(sink.ToolCall);
        Assert.Equal(text, sink.Text);
        Assert.All(chunks, c => Assert.Equal(TextChunkKind.Chunk, c.Kind));
    }


    [Fact]
    public void DefaultStopWaitsUntilEveryCallClosedByOneDeltaHasBeenEmitted()
    {
        ToolCallStreamFilter filter = new(ToolCallFormat.Mistral);
        List<TextChunk> chunks = [];
        bool stopRequested = false;
        TextFilterSink sink = new(filter, chunks.Add, () => stopRequested = true);
        sink.Handle(new TextChunk { Kind = TextChunkKind.Chunk, Text = "[{\"name\": \"a\", \"arguments\": {}}, {\"name\": \"b\", \"arguments\": {}}]" });
        Assert.Equal(["a"], chunks.Where(c => c.Kind == TextChunkKind.NativeToolCall).Select(c => c.ToolCall!.Name));
        Assert.False(sink.Stopped);
        sink.Handle(new TextChunk { Kind = TextChunkKind.Chunk, Text = "\n" });
        Assert.Equal(["a", "b"], chunks.Where(c => c.Kind == TextChunkKind.NativeToolCall).Select(c => c.ToolCall!.Name));
        Assert.True(sink.Stopped);
        Assert.True(stopRequested);
    }

    [Fact]
    public void InstalledFilterRestrictsBareFormsToTheOfferedTools()
    {
        EngineOptions options = new();
        ToolCalling.Install(options);
        TextRequest request = new()
        {
            Messages = [new TextMessage { Role = TextRole.User, Content = "hi" }],
            Tools = [new ToolDefinition { Name = "hang_up" }],
        };
        TextStreamFilterContext context = new() { Request = request, RequestId = 7, TemplateName = "jinja" };
        ToolCallStreamFilter filter = Assert.IsType<ToolCallStreamFilter>(options.TextStreamFilterFactory!(context));
        List<TextChunk> chunks = [];
        TextFilterSink sink = new(filter, chunks.Add, static () => { });
        sink.Handle(new TextChunk { Kind = TextChunkKind.Chunk, Text = "{\"name\": \"Bob\", \"age\": 3}\n" });
        Assert.False(sink.Stopped);
        Assert.Equal("{\"name\": \"Bob\", \"age\": 3}\n", sink.Text);
        sink.Handle(new TextChunk { Kind = TextChunkKind.Chunk, Text = "{\"name\": \"hang_up\", \"arguments\": {}}" });
        Assert.True(sink.Stopped);
        Assert.Equal("hang_up", sink.ToolCall!.Name);
        Assert.Equal("call_7_0", sink.ToolCall.Id);
    }

    [Fact]
    public void StructuredParserTemplateGetsNoFilter()
    {
        TextRequest request = new()
        {
            Messages = [new TextMessage { Role = TextRole.User, Content = "hi" }],
            Tools = [new ToolDefinition { Name = "hang_up" }],
        };
        TextStreamFilterContext context = new() { Request = request, RequestId = 1, TemplateName = "deepseek", HasStructuredParser = true };
        Assert.Null(ToolCalling.CreateFilter(context));
    }

    [Fact]
    public void RequestWithoutToolsGetsNoFilter()
    {
        TextRequest request = new() { Messages = [new TextMessage { Role = TextRole.User, Content = "hi" }] };
        Assert.Null(ToolCalling.CreateFilter(new TextStreamFilterContext { Request = request, RequestId = 1, TemplateName = "jinja" }));
    }
}

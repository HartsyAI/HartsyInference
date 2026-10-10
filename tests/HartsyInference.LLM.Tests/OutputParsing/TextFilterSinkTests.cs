using System.Text;
using System.Text.Json;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.OutputParsing;
using Xunit;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>The stream-filter seam driven exactly as <c>TextService.RunText</c> wires it (parser → translator → sink): a fake filter that recognises a <c>&lt;tool_call&gt;</c> span proves the forwarded text, the <see cref="TextChunkKind.NativeToolCall"/> chunk and the stop relay.</summary>
public sealed class TextFilterSinkTests
{
    private const string Completion = "Sure. <tool_call>{\"name\": \"hang_up\", \"arguments\": {\"reason\": \"done\"}}</tool_call> trailing text";

    private static (List<TextChunk> Chunks, TextFilterSink Sink, bool StopRequested) Drive(ITextStreamFilter filter, string completion, int seed)
    {
        PieceTokenizer tok = new();
        PassthroughOutputParser parser = new(tok);
        List<TextChunk> chunks = [];
        bool stopRequested = false;
        TextFilterSink sink = new(filter, chunks.Add, () => stopRequested = true);
        ParsedEventTranslator translator = new(sink.Handle, 7);
        foreach (int id in tok.RandomWithSpecials(completion, new Random(seed), 4))
        {
            parser.Push(id, translator.Handle);
            if (sink.Stopped) break;
        }
        if (!sink.Stopped)
        {
            parser.Finish(translator.Handle);
            sink.End();
        }
        return (chunks, sink, stopRequested);
    }

    [Theory]
    [InlineData(1)]
    public void CompletedToolCallEmitsANativeToolCallChunkAndStopsGeneration(int seed)
    {
        (List<TextChunk> chunks, TextFilterSink sink, bool stopRequested) = Drive(new SentinelFilter(), Completion, seed);
        Assert.True(sink.Stopped);
        Assert.True(stopRequested, "the sink must relay the filter's stop to the generation loop");
        Assert.Equal("Sure. ", sink.Text);
        Assert.Equal("Sure. ", string.Concat(chunks.Where(c => c.Kind == TextChunkKind.Chunk).Select(c => c.Text)));
        TextChunk call = Assert.Single(chunks, c => c.Kind == TextChunkKind.NativeToolCall);
        Assert.Equal("hang_up", call.ToolCall!.Name);
        Assert.Equal("{\"reason\": \"done\"}", call.ToolCall.Arguments);
        Assert.Equal(0, call.ToolCallIndex);
        Assert.Same(call.ToolCall, sink.ToolCall);
        Assert.Equal(TextChunkKind.NativeToolCall, chunks[^1].Kind);
    }

    [Fact]
    public void HeldTextIsFlushedAtEndAndACallCompletedThereIsReported()
    {
        (List<TextChunk> chunks, TextFilterSink sink, bool stopRequested) = Drive(new HoldAllFilter(), "held until the end", 9);
        Assert.False(stopRequested);
        Assert.Equal("held until the end", sink.Text);
        Assert.Equal("held until the end", Assert.Single(chunks, c => c.Kind == TextChunkKind.Chunk).Text);
        Assert.Equal("at_end", sink.ToolCall!.Name);
        Assert.Equal(TextChunkKind.NativeToolCall, chunks[^1].Kind);
    }

    [Fact]
    public void NothingIsForwardedAfterAStop()
    {
        StringBuilder later = new();
        TextFilterSink sink = new(new StopAtOnceFilter(), c => later.Append(c.Text), static () => { });
        sink.Handle(new TextChunk { Kind = TextChunkKind.Chunk, Text = "first" });
        Assert.True(sink.Stopped);
        sink.Handle(new TextChunk { Kind = TextChunkKind.Chunk, Text = "second" });
        sink.Handle(new TextChunk { Kind = TextChunkKind.Reasoning, Text = "third" });
        sink.End();
        Assert.Equal("first", later.ToString());
    }

    /// <summary>Forwards text outside a <c>&lt;tool_call&gt;…&lt;/tool_call&gt;</c> span (holding back a possible partial opening tag), completes the call when the span closes and stops.</summary>
    private sealed class SentinelFilter : ITextStreamFilter
    {
        private const string Open = "<tool_call>";
        private const string Close = "</tool_call>";
        private readonly StringBuilder _buffer = new();
        private bool _inCall;

        public TextFilterResult OnDelta(string delta)
        {
            _buffer.Append(delta);
            string forward = "";
            if (!_inCall)
            {
                string text = _buffer.ToString();
                int at = text.IndexOf(Open, StringComparison.Ordinal);
                if (at < 0)
                {
                    int keep = PartialSuffix(text, Open);
                    forward = text[..^keep];
                    _buffer.Remove(0, forward.Length);
                    return TextFilterResult.Forward(forward);
                }
                forward = text[..at];
                _buffer.Remove(0, at + Open.Length);
                _inCall = true;
            }
            string inner = _buffer.ToString();
            int end = inner.IndexOf(Close, StringComparison.Ordinal);
            if (end < 0) return TextFilterResult.Forward(forward);
            using JsonDocument doc = JsonDocument.Parse(inner[..end]);
            NativeToolCall call = new()
            {
                Id = "call_test_0",
                Name = doc.RootElement.GetProperty("name").GetString()!,
                Arguments = doc.RootElement.GetProperty("arguments").GetRawText(),
            };
            return new TextFilterResult(forward, call, Stop: true);
        }

        public TextFilterResult OnEnd() => TextFilterResult.Forward(_inCall ? "" : _buffer.ToString());

        private static int PartialSuffix(string text, string marker)
        {
            for (int len = Math.Min(marker.Length - 1, text.Length); len > 0; len--)
                if (text.EndsWith(marker[..len], StringComparison.Ordinal)) return len;
            return 0;
        }
    }

    private sealed class HoldAllFilter : ITextStreamFilter
    {
        private readonly StringBuilder _held = new();

        public TextFilterResult OnDelta(string delta)
        {
            _held.Append(delta);
            return TextFilterResult.Empty;
        }

        public TextFilterResult OnEnd() => new(_held.ToString(), new NativeToolCall { Name = "at_end" });
    }

    private sealed class StopAtOnceFilter : ITextStreamFilter
    {
        public TextFilterResult OnDelta(string delta) => new(delta, null, Stop: true);

        public TextFilterResult OnEnd() => TextFilterResult.Forward("never");
    }
}

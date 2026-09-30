using System.Text;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;
using HartsyInference.LLM.OutputParsing;

namespace HartsyInference.Engine.Services;

/// <summary>Turns parser events into the streamed <see cref="TextChunk"/> vocabulary and assembles each complete <see cref="NativeToolCall"/> (ids <c>call_{requestId}_{index}</c>).</summary>
internal sealed class ParsedEventTranslator
{
    private readonly Action<TextChunk> _sink;
    private readonly long _requestId;
    private readonly StringBuilder _args = new();
    private string _name = "";
    private string? _namespace;

    /// <summary>Creates a translator writing to <paramref name="sink"/> for request <paramref name="requestId"/>.</summary>
    public ParsedEventTranslator(Action<TextChunk> sink, long requestId)
    {
        _sink = sink;
        _requestId = requestId;
    }

    /// <summary>Handles one parser event.</summary>
    public void Handle(ParsedEvent e)
    {
        switch (e.Kind)
        {
            case ParsedEventKind.ContentDelta:
                _sink(new TextChunk { Kind = TextChunkKind.Chunk, Text = e.Text });
                break;
            case ParsedEventKind.ReasoningDelta:
                _sink(new TextChunk { Kind = TextChunkKind.Reasoning, Text = e.Text });
                break;
            case ParsedEventKind.ToolCallBegin:
                _name = e.Text ?? "";
                _namespace = e.Namespace;
                _args.Clear();
                _sink(new TextChunk { Kind = TextChunkKind.ToolCallDelta, ToolCallIndex = e.ToolCallIndex, ToolCall = Call(e.ToolCallIndex, "") });
                break;
            case ParsedEventKind.ToolCallArgsDelta:
                _args.Append(e.Text);
                _sink(new TextChunk { Kind = TextChunkKind.ToolCallDelta, ToolCallIndex = e.ToolCallIndex, Text = e.Text });
                break;
            case ParsedEventKind.ToolCallEnd:
                _sink(new TextChunk
                {
                    Kind = TextChunkKind.NativeToolCall,
                    ToolCallIndex = e.ToolCallIndex,
                    ToolCall = Call(e.ToolCallIndex, _args.ToString()),
                });
                break;
            case ParsedEventKind.Malformed:
                Logs.Debug($"Malformed model output: {e.Text}");
                break;
        }
    }

    private NativeToolCall Call(int index, string arguments) => new()
    {
        Id = $"call_{_requestId}_{index}",
        Name = _name,
        Namespace = _namespace,
        Arguments = arguments,
    };
}

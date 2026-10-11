using System.Text;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Routes streamed chunks through an <see cref="ITextStreamFilter"/>: content deltas are replaced by what the filter forwards, a completed tool call becomes a <see cref="TextChunkKind.NativeToolCall"/> chunk, other chunk kinds pass through, and a stop request is relayed to the generation loop.</summary>
internal sealed class TextFilterSink
{
    private readonly ITextStreamFilter _filter;
    private readonly Action<TextChunk>? _downstream;
    private readonly Action _requestStop;
    private readonly StringBuilder _text = new();
    private readonly List<NativeToolCall> _calls = [];
    private int _callIndex;

    /// <summary>Creates a sink over <paramref name="filter"/>; <paramref name="downstream"/> receives the filtered chunks (null on the non-streaming path) and <paramref name="requestStop"/> cancels generation.</summary>
    public TextFilterSink(ITextStreamFilter filter, Action<TextChunk>? downstream, Action requestStop)
    {
        ArgumentNullException.ThrowIfNull(filter);
        ArgumentNullException.ThrowIfNull(requestStop);
        _filter = filter;
        _downstream = downstream;
        _requestStop = requestStop;
    }

    /// <summary>Content forwarded so far: the visible completion once a filter is installed.</summary>
    public string Text => _text.ToString();

    /// <summary>The most recent tool call the filter completed, or null.</summary>
    public NativeToolCall? ToolCall => _calls.Count > 0 ? _calls[^1] : null;

    /// <summary>Every tool call the filter completed, in order.</summary>
    public IReadOnlyList<NativeToolCall> ToolCalls => _calls;

    /// <summary>True once the filter asked generation to stop; every later chunk is dropped.</summary>
    public bool Stopped { get; private set; }

    /// <summary>Handles one chunk from the parser translator.</summary>
    public void Handle(TextChunk chunk)
    {
        if (Stopped) return;
        if (chunk.Kind != TextChunkKind.Chunk)
        {
            _downstream?.Invoke(chunk);
            return;
        }
        Apply(_filter.OnDelta(chunk.Text ?? ""));
    }

    /// <summary>Flushes the filter after a generation that ran to completion; a no-op after a stop.</summary>
    public void End()
    {
        if (Stopped) return;
        Apply(_filter.OnEnd());
    }

    private void Apply(TextFilterResult result)
    {
        if (!string.IsNullOrEmpty(result.ForwardText))
        {
            _text.Append(result.ForwardText);
            _downstream?.Invoke(new TextChunk { Kind = TextChunkKind.Chunk, Text = result.ForwardText });
        }
        if (result.ToolCall is { } call)
        {
            _calls.Add(call);
            _downstream?.Invoke(new TextChunk { Kind = TextChunkKind.NativeToolCall, ToolCall = call, ToolCallIndex = _callIndex++ });
        }
        if (result.Stop)
        {
            Stopped = true;
            _requestStop();
        }
    }
}

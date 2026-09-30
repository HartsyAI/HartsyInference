using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tools.Parsing;

namespace HartsyInference.Tools;

/// <summary>The <see cref="ITextStreamFilter"/> that wraps a <see cref="ToolCallParser"/>: plain text is forwarded, each completed call is emitted as a <see cref="NativeToolCall"/>, and by default generation stops after the first one (<see cref="StopAfterFirstCall"/>). One instance per request; install through <see cref="ToolCalling.Install"/>.</summary>
/// <remarks>The seam carries one call per delta. When a single delta closes several (a Mistral array), the extra calls are emitted on the following deltas and at <see cref="OnEnd"/>; all of them are listed in <see cref="Calls"/>.</remarks>
public sealed class ToolCallStreamFilter : ITextStreamFilter
{
    private readonly ToolCallParser _parser;
    private readonly List<NativeToolCall> _calls = [];
    private int _emitted;

    /// <summary>Creates a filter for <paramref name="format"/>; <paramref name="stopAfterFirstCall"/> ends generation as <see cref="StopReason.ToolCall"/> once a call completes.</summary>
    public ToolCallStreamFilter(ToolCallFormat format = ToolCallFormat.Hermes, bool stopAfterFirstCall = true)
    {
        _parser = new ToolCallParser(format);
        StopAfterFirstCall = stopAfterFirstCall;
    }

    /// <summary>The format being parsed.</summary>
    public ToolCallFormat Format => _parser.Format;

    /// <summary>True when the first completed call stops generation; false keeps decoding so the model can emit several calls, forwarding the text between them.</summary>
    public bool StopAfterFirstCall { get; }

    /// <summary>Every call completed so far, in order, including ones the seam has not surfaced yet.</summary>
    public IReadOnlyList<NativeToolCall> Calls => _calls;

    /// <inheritdoc/>
    public TextFilterResult OnDelta(string delta) => Emit(_parser.Push(delta));

    /// <inheritdoc/>
    public TextFilterResult OnEnd() => Emit(_parser.Flush());

    private TextFilterResult Emit(ToolCallParseResult parsed)
    {
        if (parsed.Calls is { Count: > 0 } calls) _calls.AddRange(calls);
        NativeToolCall? call = _emitted < _calls.Count ? _calls[_emitted++] : null;
        return new TextFilterResult(parsed.ForwardText, call, Stop: call is not null && StopAfterFirstCall);
    }
}

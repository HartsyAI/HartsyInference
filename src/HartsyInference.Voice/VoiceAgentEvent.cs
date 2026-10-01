using HartsyInference.Engine.Requests;

namespace HartsyInference.Voice;

/// <summary>One notification from a <see cref="VoiceAgentSession"/>. <see cref="Kind"/> says which payload fields are set.</summary>
public sealed record VoiceAgentEvent
{
    /// <summary>What happened.</summary>
    public required VoiceAgentEventKind Kind { get; init; }

    /// <summary>The turn it belongs to; 0 outside any turn.</summary>
    public int TurnId { get; init; }

    /// <summary>The new state, for <see cref="VoiceAgentEventKind.StateChanged"/>.</summary>
    public VoiceAgentState State { get; init; }

    /// <summary>Transcript text, tool result, discard reason or error message.</summary>
    public string? Text { get; init; }

    /// <summary>The call, for <see cref="VoiceAgentEventKind.ToolCall"/> and <see cref="VoiceAgentEventKind.ToolResult"/>.</summary>
    public NativeToolCall? ToolCall { get; init; }

    /// <summary>The turn's timings, for <see cref="VoiceAgentEventKind.TurnCompleted"/>.</summary>
    public VoiceTurnMetrics? Metrics { get; init; }

    /// <summary>The failure, for <see cref="VoiceAgentEventKind.Error"/>.</summary>
    public Exception? Error { get; init; }

    /// <summary>When it happened, in <c>MonotonicClock</c> nanoseconds.</summary>
    public long TimestampNs { get; init; }
}

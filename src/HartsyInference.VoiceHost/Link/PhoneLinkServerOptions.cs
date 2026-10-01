using HartsyInference.VoiceHost.Tools;

namespace HartsyInference.VoiceHost.Link;

/// <summary>Socket, pacing, liveness and per-call settings of <see cref="PhoneLinkServer"/>.</summary>
internal sealed record PhoneLinkServerOptions
{
    /// <summary>One outbound frame, and the sender's period.</summary>
    public const long FramePeriodNs = 20_000_000L;

    public required string SocketPath { get; init; }

    /// <summary>Mode of the socket file after bind; <see cref="Config.VoiceHostConfigLoader"/> keeps it within 0660.</summary>
    public UnixFileMode SocketMode { get; init; } = UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.GroupRead | UnixFileMode.GroupWrite;

    /// <summary>The token a gateway's <c>Hello</c> must carry; empty accepts only an empty token.</summary>
    public string Token { get; init; } = "";

    /// <summary>Audio sent ahead of the pace when a burst of reply audio starts.</summary>
    public int PrebufferMs { get; init; } = 40;

    public int LivenessTimeoutMs { get; init; } = 20_000;

    /// <summary>How long a new connection may take to send <c>Hello</c>.</summary>
    public int HandshakeTimeoutMs { get; init; } = 5_000;

    /// <summary>How long a telephony tool waits for the gateway's <c>ToolResult</c>.</summary>
    public int ToolTimeoutMs { get; init; } = 10_000;

    /// <summary>How long ending a call waits for its session to stop.</summary>
    public int CallEndTimeoutMs { get; init; } = 10_000;

    /// <summary>Tools offered on every call, from <see cref="VoiceHostTools.Names"/>.</summary>
    public IReadOnlyList<string> Tools { get; init; } = VoiceHostTools.Names;

    /// <summary>Said when a call starts; null says nothing.</summary>
    public string? Greeting { get; init; }

    /// <summary>Said on a call the gateway re-attaches; null says nothing.</summary>
    public string? ResumeApology { get; init; }

    /// <summary>Periods a late sender makes up at once; beyond that it re-bases its clock.</summary>
    public int MaxCatchUpFrames { get; init; } = 5;

    /// <summary>Sender ticks before its allocation baseline is taken (the first ticks compile the path).</summary>
    public int SenderWarmUpTicks { get; init; } = 50;

    /// <summary>What <c>get_time</c> reads.</summary>
    public TimeProvider Clock { get; init; } = TimeProvider.System;

    /// <summary>Test seam: the sender's period. Production is <see cref="FramePeriodNs"/>; frames stay 20 ms of audio.</summary>
    internal long SenderPeriodNs { get; init; } = FramePeriodNs;
}

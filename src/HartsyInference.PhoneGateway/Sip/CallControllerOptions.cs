using System.Net;
using HartsyInference.PhoneGateway.Media;
using HartsyInference.PhoneGateway.Transport;

namespace HartsyInference.PhoneGateway.Sip;

/// <summary>Call policy and media settings for <see cref="CallController"/>.</summary>
public sealed record CallControllerOptions
{
    public InboundPolicy InboundPolicy { get; init; } = InboundPolicy.AllowAll;

    /// <summary>Caller user parts (numbers or SIP users) answered under <see cref="InboundPolicy.Allowlist"/>.</summary>
    public IReadOnlyList<string> Allowlist { get; init; } = [];

    /// <summary>When not empty, an outbound call or a <c>transfer</c> must dial a number starting with one of these
    /// through the registrar. Empty refuses every outbound destination unless <see cref="AllowAnyDestination"/> is set.</summary>
    public IReadOnlyList<string> DestinationPrefixes { get; init; } = [];

    /// <summary>Dial any destination (LAN and development only: no toll-fraud protection). Exclusive with
    /// <see cref="DestinationPrefixes"/>.</summary>
    public bool AllowAnyDestination { get; init; }

    /// <summary>Raw 8 kHz PCM16 file played to every answered inbound call before the host speaks; null for none.</summary>
    public string? GreetingPromptFile { get; init; }

    public AudioCodecPreference Codec { get; init; } = AudioCodecPreference.Any;

    /// <summary>First RTP port; the range must be forwarded by the router when a provider is used.</summary>
    public int RtpPortStart { get; init; } = 20000;

    public int RtpPortEnd { get; init; } = 20100;

    /// <summary>Address RTP sockets bind to; null for any.</summary>
    public IPAddress? BindAddress { get; init; }

    public int RingTimeoutSeconds { get; init; } = 45;

    /// <summary>Gap between digits for the <c>send_dtmf</c> tool when the request names none.</summary>
    public int DtmfGapMs { get; init; } = 100;

    public int TransferTimeoutSeconds { get; init; } = 10;

    public ClockedAudioSourceOptions Tick { get; init; } = new();

    public LinkOutageGuardOptions Outage { get; init; } = new();

    public RecordingOptions Recording { get; init; } = new();
}

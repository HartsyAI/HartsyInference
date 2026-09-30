using HartsyInference.PhoneGateway.Media;

namespace HartsyInference.PhoneGateway.Config;

/// <summary>The JSON configuration file (<c>phone.json</c>). Secrets are never in the file: the <c>*Env</c> fields name
/// environment variables the loader reads at start-up. Every section has defaults, so <c>{}</c> is a valid file for a
/// LAN softphone test.</summary>
public sealed record GatewayConfig
{
    public SipConfig Sip { get; init; } = new();

    public LinkConfig Link { get; init; } = new();

    public AdminConfig Admin { get; init; } = new();

    public MediaConfig Media { get; init; } = new();

    public RecordingOptions Recording { get; init; } = new();

    public LoggingConfig Logging { get; init; } = new();
}

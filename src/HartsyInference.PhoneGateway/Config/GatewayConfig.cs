namespace HartsyInference.PhoneGateway.Config;

/// <summary>The JSON configuration file (<c>phone.json</c>). Secrets are never in the file: the <c>*Env</c> fields name
/// environment variables the loader reads at start-up. Every section has defaults, so <c>{}</c> is a valid file for a
/// LAN softphone test.</summary>
/// <remarks>These records use settable properties, unlike the options records: the source-generated deserializer
/// writes <c>default</c> into every init-only property the JSON omits, which would erase the defaults.</remarks>
public sealed record GatewayConfig
{
    public SipConfig Sip { get; set; } = new();

    public LinkConfig Link { get; set; } = new();

    public AdminConfig Admin { get; set; } = new();

    public MediaConfig Media { get; set; } = new();

    public RecordingConfig Recording { get; set; } = new();

    public LoggingConfig Logging { get; set; } = new();
}

namespace HartsyInference.VoiceHost.Config;

/// <summary>The JSON configuration file (<c>voice.json</c>). Secrets are never in the file: <c>link.tokenFile</c> names a
/// secret file read at start-up. Every section has defaults, so <c>{}</c> is a valid file on a box whose engine settings
/// already point at the models.</summary>
/// <remarks>These records use settable properties, unlike the options records: the source-generated deserializer writes
/// <c>default</c> into every init-only property the JSON omits, which would erase the defaults.</remarks>
public sealed record VoiceHostConfig
{
    public HostLinkConfig Link { get; set; } = new();

    public HostModelsConfig Models { get; set; } = new();

    public HostAgentConfig Agent { get; set; } = new();

    public HostToolsConfig Tools { get; set; } = new();

    public HostEngineConfig Engine { get; set; } = new();

    public HostLoggingConfig Logging { get; set; } = new();
}

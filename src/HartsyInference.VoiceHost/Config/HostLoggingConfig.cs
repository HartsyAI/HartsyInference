namespace HartsyInference.VoiceHost.Config;

/// <summary>Log level (<c>logging</c> section).</summary>
public sealed record HostLoggingConfig
{
    /// <summary><c>Verbose</c>, <c>Debug</c>, <c>Info</c>, <c>Warning</c> or <c>Error</c>.</summary>
    public string Level { get; set; } = "Info";
}

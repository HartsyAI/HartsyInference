namespace HartsyInference.PhoneGateway.Config;

/// <summary>Log levels (<c>logging</c> section).</summary>
public sealed record LoggingConfig
{
    /// <summary><c>Verbose</c>, <c>Debug</c>, <c>Info</c>, <c>Warning</c> or <c>Error</c>.</summary>
    public string Level { get; init; } = "Info";

    /// <summary>Forward sipsorcery's Debug/Trace output. Off by default: it prints whole SIP messages, credentials included.</summary>
    public bool SipDebug { get; init; }
}

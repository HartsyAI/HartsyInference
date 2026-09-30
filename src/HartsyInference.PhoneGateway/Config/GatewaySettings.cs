namespace HartsyInference.PhoneGateway.Config;

/// <summary>The loaded configuration plus the secrets resolved from the environment variables it names. Secret values
/// are held here and nowhere else; nothing formats them into a log line.</summary>
public sealed record GatewaySettings
{
    public required GatewayConfig Config { get; init; }

    /// <summary>SIP password; empty when no registrar is configured.</summary>
    public required string SipPassword { get; init; }

    /// <summary>PhoneLink token; empty when the variable is unset.</summary>
    public required string LinkToken { get; init; }

    /// <summary>Bearer token for <c>POST /calls</c>; null disables outbound calls through the admin endpoint.</summary>
    public required string? AdminToken { get; init; }
}

using HartsyInference.PhoneGateway.Sip;

namespace HartsyInference.PhoneGateway.Admin;

/// <summary>Body of <c>GET /health</c>.</summary>
public sealed record HealthStatus
{
    /// <summary><c>ok</c> when the host link is up and registration (if configured) holds; otherwise <c>degraded</c>.</summary>
    public required string Status { get; init; }

    public required string Version { get; init; }

    public required bool LinkConnected { get; init; }

    public required uint LinkOutboundRate { get; init; }

    public required bool Registered { get; init; }

    public required string RegistrationState { get; init; }

    public required bool TickFifo { get; init; }

    public CallSummary? Call { get; init; }
}

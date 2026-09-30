namespace HartsyInference.PhoneGateway.Transport;

/// <summary>Timing of the host-outage behaviour in <see cref="LinkOutageGuard"/>.</summary>
public sealed record LinkOutageGuardOptions
{
    /// <summary>How long a live call waits for the host to come back before the gateway says goodbye and hangs up.</summary>
    public int OutageHangupMs { get; init; } = 20_000;

    /// <summary>How often the "one moment" prompt repeats while waiting.</summary>
    public int PromptRepeatMs { get; init; } = 6_000;
}

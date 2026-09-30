namespace HartsyInference.PhoneLink;

/// <summary>JSON payload of <see cref="LinkMessageType.Event"/>. Which fields are set depends on <see cref="Kind"/>; unset
/// fields are omitted from the wire.</summary>
public sealed record LinkEventMessage
{
    public required LinkEventKind Kind { get; init; }

    /// <summary>The turn the event belongs to, when it belongs to one.</summary>
    public uint? TurnId { get; init; }

    /// <summary>Session state name for <see cref="LinkEventKind.State"/>: the host's own vocabulary, shown to operators and
    /// never interpreted by the gateway.</summary>
    public string? State { get; init; }

    /// <summary>Transcript text for the transcript kinds.</summary>
    public string? Text { get; init; }

    /// <summary>Latency breakdown for <see cref="LinkEventKind.TurnLatency"/>.</summary>
    public TurnLatency? Latency { get; init; }
}

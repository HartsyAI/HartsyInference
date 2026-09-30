namespace HartsyInference.PhoneLink;

/// <summary>JSON payload of <see cref="LinkMessageType.Error"/>. A connection-level error (callId 0) is followed by the sender
/// closing the socket.</summary>
public sealed record LinkErrorMessage
{
    public required string Text { get; init; }
}

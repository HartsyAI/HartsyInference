namespace HartsyInference.PhoneLink;

/// <summary>JSON payload of <see cref="LinkMessageType.CallStart"/>. Sent by the gateway once media is flowing.</summary>
public sealed record CallStartMessage
{
    public required LinkCallDirection Direction { get; init; }

    /// <summary>Caller number or SIP identity when known; null for anonymous or withheld callers.</summary>
    public string? CallerId { get; init; }

    /// <summary>The dialed number or SIP identity when known.</summary>
    public string? Called { get; init; }

    /// <summary>SIP <c>Call-ID</c> header, for correlating with the gateway's SIP log.</summary>
    public required string SipCallId { get; init; }

    /// <summary>True when the gateway is re-attaching a call that was live when the link dropped; the host starts a fresh
    /// session and says so.</summary>
    public bool Resume { get; init; }
}

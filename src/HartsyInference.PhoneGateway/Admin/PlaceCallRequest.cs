namespace HartsyInference.PhoneGateway.Admin;

/// <summary>Body of <c>POST /calls</c>.</summary>
public sealed record PlaceCallRequest
{
    /// <summary>A SIP URI, or a number dialled through the configured registrar.</summary>
    public required string Destination { get; init; }
}

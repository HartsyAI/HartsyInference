namespace HartsyInference.PhoneGateway.Admin;

/// <summary>Body of the <c>POST /calls</c> answer.</summary>
public sealed record PlaceCallResponse
{
    public required bool Placed { get; init; }

    public required string Message { get; init; }
}

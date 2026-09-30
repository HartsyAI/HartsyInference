namespace HartsyInference.PhoneGateway.Sip;

/// <summary>Outcome of <see cref="CallController.PlaceCallAsync"/>: the status callers branch on, and a message for people.</summary>
public readonly record struct CallPlacementResult(CallPlacementStatus Status, string Message)
{
    public static CallPlacementResult Ok => new(CallPlacementStatus.Placed, "answered");

    public bool Placed => Status == CallPlacementStatus.Placed;
}

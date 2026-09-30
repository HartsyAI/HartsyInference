namespace HartsyInference.PhoneGateway.Sip;

/// <summary>Outcome of <see cref="CallController.PlaceCallAsync"/>: whether the far end answered, and why not.</summary>
public readonly record struct CallPlacementResult(bool Placed, string Message)
{
    public static CallPlacementResult Ok => new(true, "answered");
}

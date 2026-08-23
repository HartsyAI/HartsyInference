namespace HartsyInference.Engine.Requests;

/// <summary>A reference video clip and, optionally, its own soundtrack. The two travel together because a reference-conditioned model has to know which audio belongs to which clip — pairing them by list index across two separate lists is what makes an off-by-one silently condition on the wrong sound.</summary>
public sealed record ReferenceVideo
{
    /// <summary>Encoded video bytes.</summary>
    public required VideoClip Video { get; init; }

    /// <summary>This clip's soundtrack, or null when the reference is silent.</summary>
    public AudioClip? Audio { get; init; }
}

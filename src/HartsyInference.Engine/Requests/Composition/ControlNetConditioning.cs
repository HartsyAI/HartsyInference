namespace HartsyInference.Engine.Requests;

/// <summary>One ControlNet conditioning layer: a control type, its already-preprocessed hint image, and the strength
/// window over the denoising schedule. Preprocessing (Canny/Depth/OpenPose/…) is done by the caller; this carries the
/// resulting hint pixels.</summary>
public sealed record ControlNetConditioning
{
    /// <summary>Control model id or local path (the ControlNet weights to load).</summary>
    public required string Model { get; init; }

    /// <summary>The preprocessed control hint image.</summary>
    public required ImageData Image { get; init; }

    /// <summary>Conditioning strength.</summary>
    public double Strength { get; init; } = 1.0;

    /// <summary>Fraction of the schedule (0..1) at which this control begins applying.</summary>
    public double Start { get; init; }

    /// <summary>Fraction of the schedule (0..1) at which this control stops applying.</summary>
    public double End { get; init; } = 1.0;
}

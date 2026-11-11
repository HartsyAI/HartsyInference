namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Tuning for <see cref="MeshDecimator"/>.</summary>
public sealed record MeshDecimatorOptions
{
    /// <summary>Weight of the penalty planes that pin boundary edges, relative to face-plane weight (squared edge length).</summary>
    public double BoundaryWeight { get; init; } = 100.0;

    /// <summary>Small squared-edge-length term added to every collapse cost so flat regions collapse short edges first.</summary>
    public double EdgeLengthWeight { get; init; } = 1e-6;

    /// <summary>A collapse is refused if it turns any neighboring face by more than the angle with this cosine.</summary>
    public double MinNormalDot { get; init; } = 0.2;
}

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

    /// <summary>Lets an interior edge between two boundary vertices collapse, which joins two boundary loops at one vertex (a pinch) instead of leaving
    /// the edge in place. A mesh torn into many small pieces has boundary vertices almost everywhere and cannot otherwise reach a small target.</summary>
    public bool AllowBoundaryPinch { get; init; }

    /// <summary>Whether <see cref="MeshDecimator.DecimateStaged"/> may run one more pass with <see cref="AllowBoundaryPinch"/> when its result is still over the
    /// target; false keeps every vertex a single fan (no pinches) at the price of stopping early on a torn mesh.</summary>
    public bool PinchWhenStalled { get; init; } = true;
}

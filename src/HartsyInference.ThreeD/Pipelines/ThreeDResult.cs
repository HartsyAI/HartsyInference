using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>Output of a 3D pipeline. Holds a <see cref="Mesh"/> today (mesh models); the
/// <see cref="Splats"/> slot is reserved for the Gaussian-splat pipelines that arrive with later models.
/// Exactly one representation is populated per result.</summary>
public sealed class ThreeDResult
{
    /// <summary>The generated triangle mesh, or null if this result is a splat cloud.</summary>
    public Mesh? Mesh { get; init; }

    /// <summary>The generated Gaussian splats, or null if this result is a mesh.</summary>
    public GaussianSplatCloud? Splats { get; init; }

    /// <summary>The seed used (resolved if the request seed was null).</summary>
    public required int Seed { get; init; }
}

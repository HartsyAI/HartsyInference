using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>Output of a 3D pipeline: exactly one of <see cref="Mesh"/> or <see cref="Splats"/> is populated, depending on which representation the pipeline produces.</summary>
public sealed class ThreeDResult
{
    /// <summary>The generated triangle mesh, or null if this result is a splat cloud.</summary>
    public Mesh? Mesh { get; init; }

    /// <summary>The generated Gaussian splats, or null if this result is a mesh.</summary>
    public GaussianSplatCloud? Splats { get; init; }

    /// <summary>The seed used (resolved if the request seed was null).</summary>
    public required int Seed { get; init; }
}

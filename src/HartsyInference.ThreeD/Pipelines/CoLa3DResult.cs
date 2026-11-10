using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>Output of <see cref="CoLa3DPipeline"/>: one world-space mesh per decomposed object, in part order.</summary>
public sealed class CoLa3DResult
{
    /// <summary>Object meshes with vertex normals; never empty on a successful run.</summary>
    public required IReadOnlyList<Mesh> Objects { get; init; }
}

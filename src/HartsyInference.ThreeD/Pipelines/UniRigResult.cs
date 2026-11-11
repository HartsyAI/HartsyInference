using HartsyInference.ThreeD.Models.UniRig;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>Output of <see cref="UniRigPipeline"/>: the rigged GLB and the skeleton it holds.</summary>
public sealed class UniRigResult
{
    /// <summary>Binary glTF with the mesh, a bone hierarchy and per-vertex skin weights.</summary>
    public required byte[] Glb { get; init; }

    /// <summary>The generated skeleton, in the input mesh's coordinates.</summary>
    public required UniRigSkeleton Skeleton { get; init; }

    /// <summary>The seed used (resolved if the request seed was null).</summary>
    public required int Seed { get; init; }
}

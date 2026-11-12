using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.Sam3dObjects;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>What <see cref="Sam3dObjectsPipeline"/> produced for one object.</summary>
public sealed class Sam3dObjectsResult
{
    /// <summary>The Gaussian splats in the object's own frame; null when the structure stage found no voxel.</summary>
    public GaussianSplatCloud? Splats { get; init; }

    /// <summary>The object-to-camera pose in the PyTorch3D convention (the splats are not transformed by it).</summary>
    public required SamPose Pose { get; init; }

    /// <summary>Voxels the SLAT stage ran on after pruning and downsampling.</summary>
    public required int Voxels { get; init; }

    /// <summary>The seed used.</summary>
    public required int Seed { get; init; }
}

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>Settings of the voxel geodesic skin prior; defaults are upstream's <c>inference_skin_transform</c> values.</summary>
public sealed record UniRigVoxelSkinOptions
{
    /// <summary>Voxels per axis over [-1, 1]³.</summary>
    public int Grid { get; init; } = 196;

    /// <summary>Blend of linear and quadratic distance in the weight formula.</summary>
    public float Alpha { get; init; } = 0.5f;

    /// <summary>Vertices closer than this are linked by an edge of their distance.</summary>
    public float LinkDistance { get; init; } = 0.00001f;

    /// <summary>Multiplier of the distance of an edge between neighbouring interior voxels.</summary>
    public float GridWeight { get; init; } = 3.0f;
}

using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Turns a triangle mesh into the occupied voxels of a dual grid.</summary>
public interface ISceneVoxelizer
{
    /// <summary>Voxelizes <paramref name="scene"/> on <paramref name="grid"/>; dual vertices come back in voxel units.</summary>
    DualGridVoxels Voxelize(Mesh scene, SceneGrid grid);
}

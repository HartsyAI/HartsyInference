using System.Numerics;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Voxelizes a scene mesh with <see cref="DualGridVoxelizer"/> in the grid's voxel units.</summary>
public sealed class DualGridSceneVoxelizer : ISceneVoxelizer
{
    // TRELLIS.2 mesh_to_flexible_dual_grid weights: face QEF, open-boundary QEF and mean-position regularizer.
    private const float FaceWeight = 1f;
    private const float BoundaryWeight = 0.2f;
    private const float RegularizationWeight = 0.01f;

    /// <inheritdoc/>
    public DualGridVoxels Voxelize(Mesh scene, SceneGrid grid)
    {
        ArgumentNullException.ThrowIfNull(scene);
        ArgumentNullException.ThrowIfNull(grid);
        float[] source = scene.Vertices;
        float[] vertices = new float[source.Length];
        for (int i = 0; i < source.Length; i += 3)
        {
            Vector3 voxel = grid.ToVoxelUnits(new Vector3(source[i], source[i + 1], source[i + 2]));
            vertices[i] = voxel.X;
            vertices[i + 1] = voxel.Y;
            vertices[i + 2] = voxel.Z;
        }

        int r = grid.Resolution;
        return DualGridVoxelizer.Voxelize(
            vertices, scene.Indices, Vector3.One, [0, 0, 0], [r, r, r], FaceWeight, BoundaryWeight, RegularizationWeight);
    }
}

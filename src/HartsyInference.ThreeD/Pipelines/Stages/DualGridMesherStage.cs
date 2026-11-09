using System.Numerics;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Dual-grid field to a world-space mesh with vertex normals, via <see cref="DualGridMesher"/>.</summary>
public sealed class DualGridMesherStage : IDualGridMesher
{
    /// <inheritdoc/>
    public Mesh Extract(DualGridField field, SceneGrid grid)
    {
        ArgumentNullException.ThrowIfNull(field);
        ArgumentNullException.ThrowIfNull(grid);
        DualGridVoxels voxels = field.Voxels;
        float[] offsets = new float[voxels.DualVertices.Length];
        for (int i = 0; i < offsets.Length; i++)
            offsets[i] = voxels.DualVertices[i] - voxels.Coords[i];

        DualGridMesh extracted = DualGridMesher.Extract(
            voxels.Coords, offsets, voxels.Intersected, field.SplitWeight, grid.Origin, new Vector3(grid.VoxelSize));
        return MeshOps.ComputeVertexNormals(new Mesh { Vertices = extracted.Vertices, Indices = extracted.Triangles });
    }
}

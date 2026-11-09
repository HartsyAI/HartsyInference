using System.Numerics;

namespace HartsyInference.ThreeD.Geometry;

/// <summary>
/// Sparse dual grid: occupied voxel coordinates <c>[N·3]</c>, absolute dual-vertex positions <c>[N·3]</c> (normally
/// inside their voxel), and per-axis edge-intersection flags <c>[N·3]</c>, in the reference insertion order.
/// </summary>
public sealed record DualGridVoxels(int[] Coords, float[] DualVertices, bool[] Intersected)
{
    /// <summary>
    /// Converts the absolute dual vertices to offsets within each voxel, in voxel units, as
    /// <see cref="DualGridMesher.Extract"/> expects. Pass <see cref="Vector3.Zero"/> as the origin to that call.
    /// </summary>
    public float[] ToVoxelOffsets(Vector3 voxelSize)
    {
        if (!(voxelSize.X > 0f && voxelSize.Y > 0f && voxelSize.Z > 0f))
        {
            throw new ArgumentException("voxelSize must be positive on every axis.", nameof(voxelSize));
        }

        if (DualVertices.Length != Coords.Length)
        {
            throw new ArgumentException("DualVertices must be [N*3] to match Coords.", nameof(DualVertices));
        }

        int n = Coords.Length / 3;
        float[] offsets = new float[n * 3];
        for (int i = 0; i < n; i++)
        {
            for (int axis = 0; axis < 3; axis++)
            {
                float size = axis switch { 0 => voxelSize.X, 1 => voxelSize.Y, _ => voxelSize.Z };
                offsets[i * 3 + axis] = DualVertices[i * 3 + axis] / size - Coords[i * 3 + axis];
            }
        }

        return offsets;
    }
}

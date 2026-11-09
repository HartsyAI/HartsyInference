namespace HartsyInference.ThreeD.Geometry;

/// <summary>
/// Sparse dual grid: occupied voxel coordinates <c>[N·3]</c>, absolute dual-vertex positions <c>[N·3]</c> (each inside
/// its voxel), and per-axis edge-intersection flags <c>[N·3]</c>, in the reference insertion order.
/// </summary>
public sealed record DualGridVoxels(int[] Coords, float[] DualVertices, bool[] Intersected);

namespace HartsyInference.ThreeD.Geometry;

/// <summary>
/// Dual-grid field ready for mesh extraction: occupied voxels with absolute dual-vertex positions and edge flags
/// (<see cref="DualGridVoxels"/>), plus the per-voxel split weight that picks each quad's diagonal. Decoder fields also
/// carry the pre-threshold edge logits; the flags are their sign.
/// </summary>
public sealed record DualGridField(DualGridVoxels Voxels, float[] SplitWeight, float[]? IntersectionLogits = null);

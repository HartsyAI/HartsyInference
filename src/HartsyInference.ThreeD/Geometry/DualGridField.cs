namespace HartsyInference.ThreeD.Geometry;

/// <summary>
/// Dual-grid field ready for mesh extraction: occupied voxels with edge flags (<see cref="DualGridVoxels"/>) and the
/// per-voxel split weight that picks each quad's diagonal. Decoder fields hold dual vertices as offsets from each voxel
/// corner in voxel units, not absolute positions, and carry the pre-threshold edge logits; the flags are their sign.
/// </summary>
public sealed record DualGridField(DualGridVoxels Voxels, float[] SplitWeight, float[]? IntersectionLogits = null);

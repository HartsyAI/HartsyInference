namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>How the upsampled coarse voxels are mapped onto the high-resolution flow's grid.</summary>
public enum Trellis2CascadeQuantizer
{
    /// <summary>TRELLIS.2: <c>int((c + 0.5) / coarse · grid)</c>, the floor of the voxel centre scaled to <c>grid</c> cells.</summary>
    Floor,

    /// <summary>Pixal3D: <c>round((c + 0.5) / coarse · (grid − 1))</c>, half to even, matching its corner-inclusive projection grid.</summary>
    RoundToCorners,
}

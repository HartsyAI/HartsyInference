namespace HartsyInference.ThreeD.Geometry;

/// <summary>A triplane feature volume: three orthogonal feature planes (XY, XZ, YZ), each
/// <c>[Channels, Height, Width]</c>, packed contiguously as <c>[3, C, H, W]</c>. The implicit representation
/// produced by LRM-style image→3D models (TripoSR); a NeRF/occupancy MLP decodes density+color at any 3D
/// point by bilinearly sampling each plane (<see cref="Ops.GridSampler.BilinearPlane"/>) and combining.</summary>
public sealed class Triplane
{
    /// <summary>Packed plane features, length <c>3 * Channels * Height * Width</c> (plane-major: XY, XZ, YZ).</summary>
    public required float[] Features { get; init; }

    /// <summary>Channels per plane.</summary>
    public required int Channels { get; init; }

    /// <summary>Plane height.</summary>
    public required int Height { get; init; }

    /// <summary>Plane width.</summary>
    public required int Width { get; init; }

    /// <summary>Element offset of plane <paramref name="p"/> (0=XY, 1=XZ, 2=YZ) within <see cref="Features"/>.</summary>
    public int PlaneOffset(int p) => p * Channels * Height * Width;
}

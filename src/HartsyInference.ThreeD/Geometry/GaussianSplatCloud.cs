namespace HartsyInference.ThreeD.Geometry;

/// <summary>A 3D Gaussian-splat set (3DGS): per-splat position, scale, rotation, opacity, and SH color, compatible with the INRIA splat PLY layout written by <see cref="Io.PlyWriter"/>.</summary>
public sealed class GaussianSplatCloud
{
    /// <summary>Centers, length <c>3 * Count</c>.</summary>
    public required float[] Positions { get; init; }

    /// <summary>Per-axis log-scales (pre-exp), length <c>3 * Count</c>.</summary>
    public required float[] Scales { get; init; }

    /// <summary>Orientation quaternions (w,x,y,z), length <c>4 * Count</c>.</summary>
    public required float[] Rotations { get; init; }

    /// <summary>Per-splat logit opacity (pre-sigmoid), length <c>Count</c>.</summary>
    public required float[] Opacities { get; init; }

    /// <summary>SH color coefficients, length <c>Count * ShCoeffsPerSplat * 3</c> (RGB per coeff).</summary>
    public required float[] ShCoefficients { get; init; }

    /// <summary>Number of SH coefficients per channel (1 for DC-only, 16 for degree-3).</summary>
    public required int ShCoeffsPerSplat { get; init; }

    /// <summary>Number of splats.</summary>
    public int Count => Opacities.Length;
}

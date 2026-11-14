namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Settings for <see cref="PbrMeshPostProcessor"/>. Lengths are mesh units; upstream defaults assume a mesh about one unit across.</summary>
public sealed record PbrPostProcessOptions
{
    /// <summary>Face count the decimated mesh aims for (upstream default 1,000,000). A mesh at or below it is only cleaned.</summary>
    public int DecimationTarget { get; init; } = 1_000_000;

    /// <summary>Intermediate decimation factor of the two-pass schedule.</summary>
    public int IntermediateFactor { get; init; } = 3;

    /// <summary>Texture width and height in texels.</summary>
    public int TextureSize { get; init; } = 2048;

    /// <summary>Hole perimeter filled before the surface is snapshotted for baking, and again between decimation passes.</summary>
    public double HoleFillMaxPerimeter { get; init; } = 3e-2;

    /// <summary>Widest opening, as a share of the mesh's bounding-box diagonal, the cleaning passes close by zipping a boundary loop shut (<see cref="MeshCleanOptions.HoleFillMaxWidth"/>); 0 leaves them open. The raw dual-grid surface has pinholes a perimeter limit cannot close, which render as dark specks.</summary>
    public double OpeningWidthRatio { get; init; } = 0.005;

    /// <summary>Component area fraction below which fragments are dropped.</summary>
    public double SmallComponentAreaRatio { get; init; } = 1e-5;

    /// <summary>Maximum angle in degrees between a UV chart's seed normal and its faces.</summary>
    public double ChartConeHalfAngleDegrees { get; init; } = 50.0;

    /// <summary>Gutter around each chart in texels.</summary>
    public int PaddingTexels { get; init; } = 4;

    /// <summary>Telea radius for the base-color channels.</summary>
    public int BaseColorInpaintRadius { get; init; } = 3;

    /// <summary>Telea radius for metallic, roughness and alpha.</summary>
    public int OtherInpaintRadius { get; init; } = 1;

    /// <summary>Snap each texel's position to the pre-decimation surface before sampling (upstream behavior).</summary>
    public bool SnapToSourceSurface { get; init; } = true;

    /// <summary>Material name carried into the result.</summary>
    public string MaterialName { get; init; } = "pbr";
}

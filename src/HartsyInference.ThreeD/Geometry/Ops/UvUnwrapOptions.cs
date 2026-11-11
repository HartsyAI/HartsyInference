namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Settings for <see cref="UvUnwrapper"/>.</summary>
public sealed record UvUnwrapOptions
{
    /// <summary>Largest angle between a chart's seed face normal and any of its faces. Smaller values give flatter charts (less stretch,
    /// more seams); charts that fail the overlap or fold-over check are re-split at half the angle.</summary>
    public double ConeHalfAngleDegrees { get; init; } = 50.0;

    /// <summary>Splitting stops halving the angle below this value and falls back to single-triangle charts.</summary>
    public double MinConeHalfAngleDegrees { get; init; } = 3.0;

    /// <summary>Texture edge length the atlas is padded for.</summary>
    public int TextureSize { get; init; } = 2048;

    /// <summary>Gutter around every chart in texels of <see cref="TextureSize"/>; reduced automatically when charts cannot fit.</summary>
    public int PaddingTexels { get; init; } = 4;
}

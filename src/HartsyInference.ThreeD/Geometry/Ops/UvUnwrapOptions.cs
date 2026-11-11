namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Settings for <see cref="UvUnwrapper"/>.</summary>
public sealed record UvUnwrapOptions
{
    /// <summary>Largest angle between a chart's projection normal and any of its faces while the chart grows. Smaller values give flatter
    /// charts (less stretch, more seams). A face that would overlap the chart in the projection is never taken, whatever its angle.</summary>
    public double ConeHalfAngleDegrees { get; init; } = 50.0;

    /// <summary>The wider angle at which a small chart's faces may be absorbed into a neighbouring large chart. Such faces are noise
    /// or tiny facets that would otherwise cost a chart and its gutter each; they keep fewer texels, not a wrong parametrization.</summary>
    public double AbsorbConeHalfAngleDegrees { get; init; } = 80.0;

    /// <summary>The cone of the last absorb round, for leftover faces no larger than <see cref="SteepAbsorbMaxAreaRatio"/> times the mean face
    /// area; must stay under 90 degrees so no projected face is mirrored.</summary>
    public double SteepAbsorbConeHalfAngleDegrees { get; init; } = 87.0;

    /// <summary>Largest face, as a multiple of the mean face area, the steep round may fold into a chart.</summary>
    public double SteepAbsorbMaxAreaRatio { get; init; } = 3.0;

    /// <summary>Charts with fewer faces than this are dissolved into their large neighbours where the projection stays valid.</summary>
    public int AbsorbBelowFaces { get; init; } = 32;

    /// <summary>Texture edge length the atlas is padded for.</summary>
    public int TextureSize { get; init; } = 2048;

    /// <summary>Gutter around every chart in texels of <see cref="TextureSize"/>. When the charts do not fit it is halved, but never below
    /// <see cref="MinPaddingTexels"/> (the atlas scale shrinks instead), so a bilinear fetch or a dilation never reads a neighbouring chart.</summary>
    public int PaddingTexels { get; init; } = 4;

    /// <summary>Floor of the gutter in texels; at least 1 keeps the charts apart by one texel on each side.</summary>
    public int MinPaddingTexels { get; init; } = 1;
}

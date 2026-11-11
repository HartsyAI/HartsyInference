namespace HartsyInference.ThreeD.Paint;

/// <summary>Constants of the Paint back-projection bake (<c>MeshRender</c> and <c>Hunyuan3DPaintConfig</c>).</summary>
public sealed record PaintBakeOptions
{
    /// <summary>Surfaces turned more than this many degrees from the view direction are not trusted.</summary>
    public double AngleThresholdDegrees { get; init; } = 75.0;

    /// <summary>Border width of the unreliable silhouette and depth-edge region, in texels per 512 of view resolution.</summary>
    public double UnreliableBoundaryScale { get; init; } = 2.0;

    /// <summary>Largest camera-space depth difference between a texel's point and the view's surface for the point to count as visible.</summary>
    public double DepthThreshold { get; init; } = 3e-3;

    /// <summary>Exponent applied to the viewing-angle cosine when views are merged.</summary>
    public double CosineExponent { get; init; } = 4.0;

    /// <summary>Lower and upper hysteresis threshold of the depth edge detector, on the 8-bit normalised depth image.</summary>
    public (int Low, int High) DepthEdgeThresholds { get; init; } = (30, 80);

    /// <summary>A view is skipped when more than this fraction of its trusted texels are already painted.</summary>
    public double SkipOverlap { get; init; } = 0.99;
}

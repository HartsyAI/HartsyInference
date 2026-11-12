namespace HartsyInference.ThreeD.Render;

/// <summary>How a model is drawn into a picture.</summary>
public sealed record MeshRenderOptions
{
    /// <summary>The largest edge a render may have.</summary>
    public const int MaxSize = 2048;

    /// <summary>The most samples a render may hold (4096 x 4096), which bounds its memory at about 117 MB.</summary>
    public const long MaxSampleCount = 4096L * 4096;

    /// <summary>Picture width in pixels, 16 to <see cref="MaxSize"/>.</summary>
    public int Width { get; init; } = 256;

    /// <summary>Picture height in pixels, 16 to <see cref="MaxSize"/>.</summary>
    public int Height { get; init; } = 256;

    /// <summary>Where the camera is; the default is a three-quarter view.</summary>
    public MeshViewpoint Viewpoint { get; init; } = MeshViewpoint.Named("iso");

    /// <summary>Vertical field of view in degrees, 10 to 90.</summary>
    public float FovDegrees { get; init; } = 30f;

    /// <summary>Which axis of the geometry is up.</summary>
    public MeshUpAxis Up { get; init; } = MeshUpAxis.Y;

    /// <summary>Background colour as RGB bytes, or null for a transparent background (an RGBA picture).</summary>
    public byte[]? Background { get; init; }

    /// <summary>Samples per pixel along each axis (1 to 4); more smooths edges and costs the square.</summary>
    public int Samples { get; init; } = 2;

    /// <summary>Whether base-colour textures are sampled; false draws the material colour and the vertex colours only.</summary>
    public bool Textured { get; init; } = true;

    /// <summary>The most texture pixels the render decodes in all, 1 to <see cref="MeshRenderer.TextureBudgetPixels"/>; textures past it are drawn with their material colour and a note.</summary>
    public long TexturePixelBudget { get; init; } = MeshRenderer.TextureBudgetPixels;

    /// <summary>Point size in output pixels for a point or splat cloud.</summary>
    public float PointSize { get; init; } = 2f;

    /// <summary>Checks the ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A value is out of range.</exception>
    public void Validate()
    {
        if (Width is < 16 or > MaxSize)
            throw new ArgumentOutOfRangeException(nameof(Width), Width, $"Width must be between 16 and {MaxSize}.");
        if (Height is < 16 or > MaxSize)
            throw new ArgumentOutOfRangeException(nameof(Height), Height, $"Height must be between 16 and {MaxSize}.");
        if (Samples is < 1 or > 4)
            throw new ArgumentOutOfRangeException(nameof(Samples), Samples, "Samples must be between 1 and 4.");
        if ((long)Width * Height * Samples * Samples > MaxSampleCount)
            throw new ArgumentOutOfRangeException(nameof(Samples), Samples, $"Width x Height x Samples squared must be at most {MaxSampleCount} (a 2048 x 2048 picture takes 2 samples per axis at most 4096 x 4096 in all).");
        if (!(FovDegrees is >= 10f and <= 90f))
            throw new ArgumentOutOfRangeException(nameof(FovDegrees), FovDegrees, "FovDegrees must be between 10 and 90.");
        if (!(Viewpoint.ElevationDegrees is >= -90f and <= 90f) || !float.IsFinite(Viewpoint.AzimuthDegrees))
            throw new ArgumentOutOfRangeException(nameof(Viewpoint), Viewpoint, "Elevation must be between -90 and 90 degrees.");
        if (Background is { Length: not 3 })
            throw new ArgumentOutOfRangeException(nameof(Background), "Background must be three RGB bytes.");
        if (TexturePixelBudget is < 1 or > MeshRenderer.TextureBudgetPixels)
            throw new ArgumentOutOfRangeException(nameof(TexturePixelBudget), TexturePixelBudget, $"TexturePixelBudget must be between 1 and {MeshRenderer.TextureBudgetPixels}.");
        if (!(PointSize is >= 0.5f and <= 32f))
            throw new ArgumentOutOfRangeException(nameof(PointSize), PointSize, "PointSize must be between 0.5 and 32 pixels.");
    }
}

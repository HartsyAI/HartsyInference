namespace HartsyInference.Engine.Requests;

/// <summary>How a thumbnail of a model file is drawn. Every field has a default, so <c>new MeshThumbnailRequest()</c> is a 256 px three-quarter view on a transparent background.</summary>
public sealed record MeshThumbnailRequest
{
    /// <summary>Picture width in pixels, 16 to 2048.</summary>
    public int Width { get; init; } = 256;

    /// <summary>Picture height in pixels, 16 to 2048.</summary>
    public int Height { get; init; } = 256;

    /// <summary>A named viewpoint: <c>front</c>, <c>back</c>, <c>left</c>, <c>right</c>, <c>top</c>, <c>bottom</c> or <c>iso</c>. <see cref="Azimuth"/> and <see cref="Elevation"/> override it.</summary>
    public string View { get; init; } = "iso";

    /// <summary>Camera azimuth in degrees (0 front, 90 from the right, 180 back); null takes <see cref="View"/>'s.</summary>
    public float? Azimuth { get; init; }

    /// <summary>Camera elevation in degrees, -90 to 90 (90 looks straight down); null takes <see cref="View"/>'s.</summary>
    public float? Elevation { get; init; }

    /// <summary>Vertical field of view in degrees, 10 to 90.</summary>
    public float FovDegrees { get; init; } = 30f;

    /// <summary>The axis of the file that points up: <c>y</c> (glTF) or <c>z</c>. Models return their own frames, so say which when the picture comes out lying down.</summary>
    public string Up { get; init; } = "y";

    /// <summary>Background colour as <c>#rrggbb</c>, or null for a transparent one.</summary>
    public string? Background { get; init; }

    /// <summary>Samples per pixel along each axis, 1 to 4.</summary>
    public int Samples { get; init; } = 2;

    /// <summary>Whether base-colour textures are drawn; false draws material and vertex colours only.</summary>
    public bool Textured { get; init; } = true;

    /// <summary>Dot size in pixels for a point or splat file.</summary>
    public float PointSize { get; init; } = 2f;
}

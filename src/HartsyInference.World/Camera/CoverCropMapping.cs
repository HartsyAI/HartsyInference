namespace HartsyInference.World.Camera;

/// <summary>Aspect-preserving cover-fit then center-crop from a source image to a target size, with the matching mapping of
/// pixel coordinates and pinhole intrinsics into the cropped image. Positions scale as continuous (pixel-edge) coordinates,
/// so a principal point maps exactly like any other point.</summary>
public readonly record struct CoverCropMapping
{
    private CoverCropMapping(int sourceWidth, int sourceHeight, int resizedWidth, int resizedHeight, int left, int top)
    {
        SourceWidth = sourceWidth;
        SourceHeight = sourceHeight;
        ResizedWidth = resizedWidth;
        ResizedHeight = resizedHeight;
        Left = left;
        Top = top;
    }

    /// <summary>Source width in pixels.</summary>
    public int SourceWidth { get; }

    /// <summary>Source height in pixels.</summary>
    public int SourceHeight { get; }

    /// <summary>Width after the cover-fit resize, before the crop.</summary>
    public int ResizedWidth { get; }

    /// <summary>Height after the cover-fit resize, before the crop.</summary>
    public int ResizedHeight { get; }

    /// <summary>Horizontal crop offset, in resized pixels.</summary>
    public int Left { get; }

    /// <summary>Vertical crop offset, in resized pixels.</summary>
    public int Top { get; }

    /// <summary>Horizontal resize factor (<c>ResizedWidth / SourceWidth</c>).</summary>
    public double ScaleX => ResizedWidth / (double)SourceWidth;

    /// <summary>Vertical resize factor (<c>ResizedHeight / SourceHeight</c>).</summary>
    public double ScaleY => ResizedHeight / (double)SourceHeight;

    /// <summary>Computes the cover-fit and center-crop geometry. Sizes round half-to-even, as Python's <c>round</c> does in the
    /// reference preprocessing.</summary>
    public static CoverCropMapping Create(int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
    {
        if (sourceWidth <= 0 || sourceHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(sourceWidth), $"Source size must be positive; got {sourceWidth}x{sourceHeight}.");
        if (targetWidth <= 0 || targetHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetWidth), $"Target size must be positive; got {targetWidth}x{targetHeight}.");

        double scale = Math.Max((double)targetHeight / sourceHeight, (double)targetWidth / sourceWidth);
        int resizedWidth = Math.Max(targetWidth, (int)Math.Round(sourceWidth * scale));
        int resizedHeight = Math.Max(targetHeight, (int)Math.Round(sourceHeight * scale));
        return new CoverCropMapping(sourceWidth, sourceHeight, resizedWidth, resizedHeight,
            (resizedWidth - targetWidth) / 2, (resizedHeight - targetHeight) / 2);
    }

    /// <summary>Maps a source-image point into the cropped image.</summary>
    public (float X, float Y) MapPoint(float x, float y) =>
        ((float)(x * ScaleX - Left), (float)(y * ScaleY - Top));

    /// <summary>Maps a cropped-image point back to the source image. Inverse of <see cref="MapPoint"/>.</summary>
    public (float X, float Y) UnmapPoint(float x, float y) =>
        ((float)((x + Left) / ScaleX), (float)((y + Top) / ScaleY));

    /// <summary>Maps source <c>[fx, fy, cx, cy]</c> into the cropped image.</summary>
    public (float Fx, float Fy, float Cx, float Cy) MapIntrinsics(float fx, float fy, float cx, float cy) =>
        ((float)(fx * ScaleX), (float)(fy * ScaleY), (float)(cx * ScaleX - Left), (float)(cy * ScaleY - Top));

    /// <summary>Maps cropped-image <c>[fx, fy, cx, cy]</c> back to the source image. Inverse of <see cref="MapIntrinsics"/>.</summary>
    public (float Fx, float Fy, float Cx, float Cy) UnmapIntrinsics(float fx, float fy, float cx, float cy) =>
        ((float)(fx / ScaleX), (float)(fy / ScaleY), (float)((cx + Left) / ScaleX), (float)((cy + Top) / ScaleY));
}

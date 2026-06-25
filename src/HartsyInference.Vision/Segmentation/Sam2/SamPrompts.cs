namespace HartsyInference.Vision.Segmentation.Sam2;

/// <summary>A point prompt in image pixel coordinates with a label: foreground (include) or background (exclude).</summary>
public readonly record struct SamPointPrompt(float X, float Y, bool Foreground);

/// <summary>A box prompt in image pixel coordinates (top-left / bottom-right).</summary>
public readonly record struct SamBoxPrompt(float X0, float Y0, float X1, float Y1);

/// <summary>A complete prompt set for one segmentation request. Any combination of points and an optional
/// box is allowed (matching the SAM API). An empty point list with a box is the common "box prompt" case.</summary>
public sealed class SamPrompt
{
    /// <summary>Foreground/background point clicks.</summary>
    public IReadOnlyList<SamPointPrompt> Points { get; init; } = [];

    /// <summary>Optional bounding-box prompt.</summary>
    public SamBoxPrompt? Box { get; init; }
}

/// <summary>One predicted mask with its estimated quality (IoU) score.</summary>
public sealed class SamMaskResult
{
    /// <summary>Row-major binary mask <c>[Height*Width]</c> (1 = object, 0 = background).</summary>
    public required byte[] Mask { get; init; }

    /// <summary>Mask width in pixels.</summary>
    public required int Width { get; init; }

    /// <summary>Mask height in pixels.</summary>
    public required int Height { get; init; }

    /// <summary>The decoder's predicted IoU (mask-quality) score in [0,1].</summary>
    public required float IoUScore { get; init; }
}

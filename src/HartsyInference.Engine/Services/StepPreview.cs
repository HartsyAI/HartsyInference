namespace HartsyInference.Engine.Services;

/// <summary>A progress tick from a generation: the current step, the total, and an optional low-resolution
/// RGB preview of the in-progress result. Image generations populate <see cref="PreviewRgb"/>. Video
/// generations additionally populate <see cref="PreviewFramesRgb"/> with the temporal latent frames.</summary>
public readonly record struct StepPreview
{
    /// <summary>The current step (1-based).</summary>
    public int Step { get; init; }

    /// <summary>Total steps in the schedule; 0 when indeterminate.</summary>
    public int TotalSteps { get; init; }

    /// <summary>Optional RGB24 preview pixels; null when no preview was produced.</summary>
    public byte[]? PreviewRgb { get; init; }

    /// <summary>Optional RGB24 video-preview frames in temporal order. Each entry has
    /// <c>PreviewWidth * PreviewHeight * 3</c> bytes. Null for static previews.</summary>
    public byte[][]? PreviewFramesRgb { get; init; }

    /// <summary>Preview width in pixels.</summary>
    public int PreviewWidth { get; init; }

    /// <summary>Preview height in pixels.</summary>
    public int PreviewHeight { get; init; }
}

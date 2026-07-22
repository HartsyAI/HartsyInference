namespace HartsyInference.Engine.Services;

/// <summary>A progress tick from a diffusion/video generation: the current step, the total, and an optional
/// low-res RGB preview of the in-progress latent.</summary>
public readonly record struct StepPreview
{
    /// <summary>The current step (1-based).</summary>
    public int Step { get; init; }

    /// <summary>Total steps in the schedule; 0 when indeterminate.</summary>
    public int TotalSteps { get; init; }

    /// <summary>Optional RGB24 preview pixels; null when no preview was produced.</summary>
    public byte[]? PreviewRgb { get; init; }

    /// <summary>Preview width in pixels.</summary>
    public int PreviewWidth { get; init; }

    /// <summary>Preview height in pixels.</summary>
    public int PreviewHeight { get; init; }
}

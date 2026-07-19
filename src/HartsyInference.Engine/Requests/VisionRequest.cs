namespace HartsyInference.Engine.Requests;

/// <summary>Native vision request: embed, detect, or segment an image, optionally conditioned on a text prompt
/// (for open-vocabulary detectors/segmenters like GroundingDINO and ClipSeg).</summary>
public sealed record VisionRequest
{
    /// <summary>The input image.</summary>
    public required ImageData Image { get; init; }

    /// <summary>The operation to perform.</summary>
    public required VisionMode Mode { get; init; }

    /// <summary>Text query for open-vocabulary detect/segment; null/empty for class-agnostic or embed.</summary>
    public string? Prompt { get; init; }

    /// <summary>Score threshold for detections/masks.</summary>
    public double Threshold { get; init; } = 0.25;
}

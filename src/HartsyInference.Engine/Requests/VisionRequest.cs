using HartsyInference.Core.MemoryManagement;

namespace HartsyInference.Engine.Requests;

/// <summary>Native vision request: embed, detect, or segment an image, optionally conditioned on a text prompt (for open-vocabulary detectors/segmenters like GroundingDINO and ClipSeg).</summary>
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

    /// <summary><see cref="VisionMode.Upscale"/> only: requested output width. Null enlarges by the model factor and
    /// stops. When set with <see cref="TargetHeight"/>, the upscaler runs as many model passes as the target needs
    /// (at most two) and then resizes down to the target; it never stretches past what the last pass produced.</summary>
    public int? TargetWidth { get; init; }

    /// <summary><see cref="VisionMode.Upscale"/> only: requested output height; see <see cref="TargetWidth"/>.</summary>
    public int? TargetHeight { get; init; }

    /// <summary>Per-request VRAM lever overrides; null follows the backend's policy.</summary>
    public VramOverrides? Vram { get; init; }
}

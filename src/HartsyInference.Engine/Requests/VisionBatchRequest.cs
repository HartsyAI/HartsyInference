using HartsyInference.Core.Memory.Vram;

namespace HartsyInference.Engine.Requests;

/// <summary>What an ordered list of images in a <see cref="VisionBatchRequest"/> is.</summary>
public enum VisionSequenceKind
{
    /// <summary>Several views of one scene with no time order (multi-view depth, geometry and camera estimation).</summary>
    Views,

    /// <summary>Frames of a video in time order (tracking, interpolation).</summary>
    Frames,
}

/// <summary>A vision request over several images: the views of one scene or the frames of a video. Single-image modes
/// keep using <see cref="VisionRequest"/>; <see cref="VisionMode.Track"/> and <see cref="VisionMode.Interpolate"/> only
/// accept this record. <see cref="Services.VisionRequestValidator"/> enforces the per-mode rules.</summary>
public sealed record VisionBatchRequest
{
    /// <summary>The ordered images. All must share one size.</summary>
    public required IReadOnlyList<ImageData> Images { get; init; }

    /// <summary>The operation to perform.</summary>
    public required VisionMode Mode { get; init; }

    /// <summary>Whether <see cref="Images"/> are views or frames.</summary>
    public required VisionSequenceKind Kind { get; init; }

    /// <summary><see cref="VisionSequenceKind.Frames"/> only: each frame's presentation time in seconds, strictly
    /// increasing, one per image. Exactly one of this and <see cref="Fps"/> is required for frames.</summary>
    public IReadOnlyList<double>? TimestampsSeconds { get; init; }

    /// <summary><see cref="VisionSequenceKind.Frames"/> only: a constant frame rate, frame <c>i</c> at <c>i / Fps</c>
    /// seconds. Exactly one of this and <see cref="TimestampsSeconds"/> is required for frames.</summary>
    public double? Fps { get; init; }

    /// <summary>Per-frame prompts (text, points, boxes). <see cref="VisionMode.Track"/> needs at least one.</summary>
    public IReadOnlyList<VisionFramePrompt>? Prompts { get; init; }

    /// <summary><see cref="VisionMode.Interpolate"/> only: output frames per input interval (2 doubles the rate).
    /// Exactly one of this and <see cref="TargetFps"/> is required.</summary>
    public int? InterpolationFactor { get; init; }

    /// <summary><see cref="VisionMode.Interpolate"/> only: the output frame rate, above the source rate. Exactly one of
    /// this and <see cref="InterpolationFactor"/> is required.</summary>
    public double? TargetFps { get; init; }

    /// <summary>Score threshold for detections, masks and tracks.</summary>
    public double Threshold { get; init; } = 0.25;

    /// <summary>Per-request VRAM lever overrides; null follows the backend's policy.</summary>
    public VramOverrides? Vram { get; init; }

    /// <summary>The time of frame <paramref name="index"/> in seconds, from <see cref="TimestampsSeconds"/> or
    /// <see cref="Fps"/>; null for views.</summary>
    public double? TimeOf(int index) =>
        Kind != VisionSequenceKind.Frames ? null
            : TimestampsSeconds is { } times ? times[index]
            : Fps is > 0 and double fps ? index / fps
            : null;
}

/// <summary>A prompt attached to one frame of a <see cref="VisionBatchRequest"/>. Points and boxes are in that frame's
/// pixel coordinates (pixel centres at +0.5, see <c>docs/Agents/ENGINE_PATTERNS.md</c>).</summary>
public sealed record VisionFramePrompt
{
    /// <summary>Zero-based index into <see cref="VisionBatchRequest.Images"/>.</summary>
    public required int FrameIndex { get; init; }

    /// <summary>Object id the prompt belongs to; prompts sharing an id refine one object.</summary>
    public int ObjectId { get; init; }

    /// <summary>Text naming what to find, for text-prompted models.</summary>
    public string? Text { get; init; }

    /// <summary>Click points with a positive or negative label.</summary>
    public IReadOnlyList<VisionPoint>? Points { get; init; }

    /// <summary>Boxes around the object.</summary>
    public IReadOnlyList<VisionBox>? Boxes { get; init; }

    /// <summary>True when the prompt carries any text, point or box.</summary>
    public bool IsEmpty => string.IsNullOrWhiteSpace(Text) && Points is not { Count: > 0 } && Boxes is not { Count: > 0 };
}

/// <summary>A click point in pixels; <see cref="Positive"/> false marks background.</summary>
public readonly record struct VisionPoint(float X, float Y, bool Positive = true);

/// <summary>An axis-aligned box in pixels, <c>[X0, Y0)</c> to <c>[X1, Y1)</c>.</summary>
public readonly record struct VisionBox(float X0, float Y0, float X1, float Y1);

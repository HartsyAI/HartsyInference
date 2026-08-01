namespace HartsyInference.Engine.Requests;

/// <summary>Video/image restoration request (SeedVR2). Exactly one of <see cref="Video"/>,
/// <see cref="Frames"/>, or <see cref="Image"/> must be set. The target is an AREA
/// (<see cref="TargetWidth"/>×<see cref="TargetHeight"/>) with aspect preserved — SeedVR2 has no scale
/// factor; the bicubic area-resize is part of the model contract and the DiT restores detail at the
/// target resolution.</summary>
public sealed record RestoreRequest
{
    /// <summary>Encoded input video (decoded via the ffmpeg subprocess decoder).</summary>
    public VideoClip? Video { get; init; }

    /// <summary>Pre-decoded RGB24 frames (used by in-process callers like the SwarmUI extension to skip
    /// the container round-trip). All frames must share one size.</summary>
    public IReadOnlyList<ImageData>? Frames { get; init; }

    /// <summary>Single still image — exercises the model's t==1 branch.</summary>
    public ImageData? Image { get; init; }

    /// <summary>Target width component of the area budget; default 1280.</summary>
    public int? TargetWidth { get; init; }

    /// <summary>Target height component of the area budget; default 720.</summary>
    public int? TargetHeight { get; init; }

    /// <summary>Frames per processing chunk; rounded up to the causal-VAE contract ((n−1) % 4 == 0).</summary>
    public int? ClipFrames { get; init; }

    /// <summary>Frames of overlap between chunks, cross-faded.</summary>
    public int? Overlap { get; init; }

    /// <summary>Wavelet transplant weight in [0,1]; 1.0 (default) = pure model output. Lower values pull
    /// the low-frequency band toward the resized input — the oversharpening guard.</summary>
    public float? Strength { get; init; }

    /// <summary>Output frame rate for encoded results; null keeps the source rate.</summary>
    public double? Fps { get; init; }

    /// <summary>RNG seed; null uses the reference default (666).</summary>
    public int? Seed { get; init; }
}

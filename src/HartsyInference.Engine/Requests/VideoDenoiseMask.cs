namespace HartsyInference.Engine.Requests;

/// <summary>Continuous video-latent denoise mask and the source pixels preserved by its black regions.</summary>
public sealed record VideoDenoiseMask
{
    /// <summary>Single-frame mask; mutually exclusive with <see cref="MaskVideo"/>. White generates and black preserves.</summary>
    public ImageData? MaskImage { get; init; }

    /// <summary>Per-frame mask clip; mutually exclusive with <see cref="MaskImage"/>.</summary>
    public VideoClip? MaskVideo { get; init; }

    /// <summary>One mask value per latent frame, the video mirror of <see cref="AudioDenoiseMask.Values"/>; mutually
    /// exclusive with the other two. A binary boundary cannot survive <see cref="MaskVideo"/>'s lossy codec — mask
    /// values quantize upward, so a smeared black partly denoises rows meant to be preserved.</summary>
    public IReadOnlyList<float>? MaskFrameValues { get; init; }

    /// <summary>Single source frame to preserve; mutually exclusive with <see cref="SourceVideo"/>.</summary>
    public ImageData? SourceImage { get; init; }

    /// <summary>Source clip to preserve; mutually exclusive with <see cref="SourceImage"/>.</summary>
    public VideoClip? SourceVideo { get; init; }

    /// <summary>Raw source frames to preserve, in order; mutually exclusive with the other two. Skips the
    /// encode/decode round trip a caller already holding decoded frames would otherwise pay.</summary>
    public IReadOnlyList<ImageData>? SourceFrames { get; init; }
}

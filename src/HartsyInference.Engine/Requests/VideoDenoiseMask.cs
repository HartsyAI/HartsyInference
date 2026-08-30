namespace HartsyInference.Engine.Requests;

/// <summary>Continuous video-latent denoise mask and the source pixels preserved by its black regions.</summary>
public sealed record VideoDenoiseMask
{
    /// <summary>Single-frame mask; mutually exclusive with <see cref="MaskVideo"/>. White generates and black preserves.</summary>
    public ImageData? MaskImage { get; init; }

    /// <summary>Per-frame mask clip; mutually exclusive with <see cref="MaskImage"/>.</summary>
    public VideoClip? MaskVideo { get; init; }

    /// <summary>Single source frame to preserve; mutually exclusive with <see cref="SourceVideo"/>.</summary>
    public ImageData? SourceImage { get; init; }

    /// <summary>Source clip to preserve; mutually exclusive with <see cref="SourceImage"/>.</summary>
    public VideoClip? SourceVideo { get; init; }
}

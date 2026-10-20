namespace HartsyInference.Engine.Features;

/// <summary>How an edit model wants its reference images prepared: how many it addresses, the pixel budget and snap of
/// the copy its VAE encodes, and the budget of the copy its text encoder's vision tower reads. Consumed by
/// <see cref="ReferenceImageResolver"/>.</summary>
public sealed record ReferenceImagePolicy
{
    /// <summary>Reference slots the model was trained with.</summary>
    public required int MaxReferences { get; init; }

    /// <summary>Pixel budget of the VAE copy; 0 keeps each image's own size (still snapped).</summary>
    public int LatentTargetArea { get; init; }

    /// <summary>Whether a smaller image is scaled up to <see cref="LatentTargetArea"/>; false only scales down.</summary>
    public bool LatentUpscale { get; init; }

    /// <summary>The VAE copy's sides are rounded to this multiple (the latent token stride).</summary>
    public required int LatentMultiple { get; init; }

    /// <summary>When set, each reference is cover-fit and centre-cropped to this size first, and both copies come from that
    /// page, so a reference lines up with the target grid it is placed on; null keeps each image's own aspect.</summary>
    public (int Width, int Height)? FitTo { get; init; }

    /// <summary>When above zero, the VAE copy's sides are the fitted size divided by this factor (rounded to
    /// <see cref="LatentMultiple"/>) instead of an area budget.</summary>
    public int LatentSideDivisor { get; init; }

    /// <summary>Pixel budget of the vision-tower copy; 0 makes no vision copy.</summary>
    public int VisionTargetArea { get; init; }

    /// <summary>Whether a smaller image is scaled up to <see cref="VisionTargetArea"/>; false only scales down.</summary>
    public bool VisionUpscale { get; init; }

    /// <summary>What happens to images past <see cref="MaxReferences"/>.</summary>
    public ReferenceOverflow Overflow { get; init; } = ReferenceOverflow.Refuse;
}

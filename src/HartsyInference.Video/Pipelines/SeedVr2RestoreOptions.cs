namespace HartsyInference.Video.Pipelines;

/// <summary>Options for <see cref="SeedVr2RestorePipeline"/>. Reference defaults throughout; the target is
/// an AREA (aspect preserved by AreaResize), never an output size or scale factor.</summary>
public sealed record SeedVr2RestoreOptions
{
    /// <summary>Target pixel area (reference res_h·res_w = 1280·720).</summary>
    public long TargetArea { get; init; } = 1280L * 720L;

    /// <summary>RNG seed for the posterior sample and the init noise (reference default 666).</summary>
    public int Seed { get; init; } = 666;

    /// <summary>Wavelet transplant weight: 1.0 = pure model output (reference default — color_fix.py is an
    /// optional drop-in upstream); &lt;1 lerps the low-frequency band toward the resized input.</summary>
    public float Strength { get; init; } = 1.0f;

    /// <summary>Frames per chunk; rounded up to the causal-VAE contract (n−1) % 4 == 0. Whole-clip when the
    /// input is shorter.</summary>
    public int ClipFrames { get; init; } = 25;

    /// <summary>Frames of overlap between consecutive chunks, linearly cross-faded.</summary>
    public int Overlap { get; init; } = 4;
}

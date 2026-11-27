namespace HartsyInference.Video.Pipelines;

/// <summary>How a SeedVR2 restore's colours are matched back to the resized input. The same modes as ComfyUI's
/// <c>SeedVR2PostProcessing</c>: <see cref="Lab"/> is its default, <see cref="AdaIn"/> its global-tint mode. The wavelet
/// mode is <see cref="SeedVr2RestoreOptions.Strength"/> at 0.</summary>
public enum SeedVr2ColorTransfer
{
    /// <summary>No colour transfer: the model's output as is.</summary>
    None,

    /// <summary>CIELAB transfer: wavelet recombination, then histogram matching of a/b and 80% of L.</summary>
    Lab,

    /// <summary>Per-channel mean and standard deviation matching.</summary>
    AdaIn,
}

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

    /// <summary>Colour transfer applied to each restored frame before <see cref="Strength"/>'s wavelet lerp.
    /// Defaults to <see cref="SeedVr2ColorTransfer.None"/>, the reference's no-transfer path.</summary>
    public SeedVr2ColorTransfer ColorTransfer { get; init; } = SeedVr2ColorTransfer.None;

    /// <summary>Frames per chunk; rounded up to the causal-VAE contract (n−1) % 4 == 0. Whole-clip when the
    /// input is shorter.</summary>
    public int ClipFrames { get; init; } = 25;

    /// <summary>Frames of overlap between consecutive chunks, linearly cross-faded.</summary>
    public int Overlap { get; init; } = 4;
}

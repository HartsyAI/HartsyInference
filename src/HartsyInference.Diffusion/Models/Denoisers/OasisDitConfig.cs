namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for the Oasis-500m DiT-S/2 (Decart/Etched, MIT). No config.json ships with the checkpoint —
/// these constants are hard-coded in <c>open-oasis/dit.py</c>. See <c>docs/Research/OASIS_ARCHITECTURE.md</c> § 3.</summary>
public sealed record OasisDitConfig
{
    /// <summary>Latent grid height (360 / 20).</summary>
    public int InputH { get; init; } = 18;

    /// <summary>Latent grid width (640 / 20).</summary>
    public int InputW { get; init; } = 32;

    /// <summary>DiT-internal latent patch size (→ 9×16 = 144 tokens per frame).</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>VAE latent channels.</summary>
    public int InChannels { get; init; } = 16;

    /// <summary>Model dim.</summary>
    public int HiddenSize { get; init; } = 1024;

    /// <summary>Number of spatio-temporal blocks.</summary>
    public int Depth { get; init; } = 16;

    /// <summary>Attention heads (head dim = HiddenSize / NumHeads = 64).</summary>
    public int NumHeads { get; init; } = 16;

    /// <summary>MLP expansion.</summary>
    public float MlpRatio { get; init; } = 4.0f;

    /// <summary>Action vector width (Minecraft VPT: 23 keys + 2 camera floats).</summary>
    public int ExternalCondDim { get; init; } = 25;

    /// <summary>Sliding-window cap on the time axis.</summary>
    public int MaxFrames { get; init; } = 32;

    /// <summary>Sinusoidal timestep embedding width.</summary>
    public int FreqDim { get; init; } = 256;

    /// <summary>Spatial axial RoPE: dims rotated per axis (head_dim / 2).</summary>
    public int SpatialRopeDimPerAxis { get; init; } = 32;

    /// <summary>Spatial axial RoPE max frequency ("pixel" mode).</summary>
    public double SpatialRopeMaxFreq { get; init; } = 256;

    /// <summary>The released DiT-S/2 checkpoint.</summary>
    public static OasisDitConfig S2 => new();
}

namespace HartsyInference.Diffusion.Models.Denoisers.Diamond;

/// <summary>Configuration for the DIAMOND (Eloi Alonso et al., MIT) diffusion world model — a CNN U-Net
/// denoiser with EDM (Karras) preconditioning, conditioned on the last <see cref="NumStepsConditioning"/>
/// frames + actions. The Atari-100k preset is a 4-level 64-channel U-Net (no attention except the mid block);
/// the CS:GO preset scales up. Matches <c>eloialonso/diamond</c>'s <c>InnerModelConfig</c>/<c>DenoiserConfig</c>.</summary>
public sealed record DiamondConfig
{
    /// <summary>Image channels per frame (RGB = 3).</summary>
    public int ImgChannels { get; init; } = 3;

    /// <summary>Number of past frames the denoiser conditions on.</summary>
    public int NumStepsConditioning { get; init; } = 4;

    /// <summary>Conditioning embedding width (Fourier noise + action embedding, projected).</summary>
    public int CondChannels { get; init; } = 256;

    /// <summary>Discrete action count (per game; Breakout = 4).</summary>
    public int NumActions { get; init; } = 4;

    /// <summary>Residual blocks per U-Net level.</summary>
    public int[] Depths { get; init; } = [2, 2, 2, 2];

    /// <summary>Channels per U-Net level.</summary>
    public int[] Channels { get; init; } = [64, 64, 64, 64];

    /// <summary>Whether each level's residual blocks carry self-attention (Atari: none).</summary>
    public bool[] AttnDepths { get; init; } = [false, false, false, false];

    /// <summary>Square frame size (Atari = 64).</summary>
    public int ImgSize { get; init; } = 64;

    // --- EDM preconditioning (DenoiserConfig) ---
    /// <summary>Karras <c>sigma_data</c>.</summary>
    public float SigmaData { get; init; } = 0.5f;

    /// <summary>Per-channel offset-noise std folded into the effective sigma.</summary>
    public float SigmaOffsetNoise { get; init; } = 0.3f;

    // --- Sampler (DiffusionSamplerConfig) ---
    /// <summary>Denoising steps at inference (DIAMOND Atari uses a handful).</summary>
    public int NumStepsDenoising { get; init; } = 3;

    /// <summary>Karras sigma range + rho.</summary>
    public float SigmaMin { get; init; } = 2e-3f;
    public float SigmaMax { get; init; } = 5f;
    public int Rho { get; init; } = 7;

    /// <summary>GroupNorm group size (channels / 32, min 1) and epsilon — DIAMOND's <c>blocks.py</c> constants.</summary>
    public int GnGroupSize { get; init; } = 32;
    public float GnEps { get; init; } = 1e-5f;

    /// <summary>Attention head dim for the mid-block self-attention.</summary>
    public int AttnHeadDim { get; init; } = 8;

    /// <summary>The <c>eloialonso/diamond</c> Atari-100k preset (per-game <paramref name="numActions"/>).</summary>
    public static DiamondConfig Atari(int numActions) => new() { NumActions = numActions };
}

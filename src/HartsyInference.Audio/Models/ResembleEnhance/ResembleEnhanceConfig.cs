namespace HartsyInference.Audio.Models.ResembleEnhance;

/// <summary>Configuration for Resemble AI's resemble-enhance speech enhancer. The LCFM enhancer = a latent
/// CFM (a WaveNet velocity net solved by OT-CFM) + an IRMAE latent autoencoder, conditioned on a (lambda-
/// blended, optionally denoised) mel, then a UnivNet vocoder. See <c>docs/Research/RESEMBLE_ENHANCE_ARCHITECTURE.md</c>.
///
/// <para><b>Reuse:</b> the CFM solver reuses the CosyVoice <c>ConditionalCfm</c> (OT-CFM Euler, CFG off); the
/// velocity net is a new <c>ICfmEstimator</c> (WN WaveNet). Mel via <c>MelSpectrogramExtractor</c>.</para></summary>
public sealed record ResembleEnhanceConfig
{
    public int SampleRate { get; init; } = 44_100;
    public int NMels { get; init; } = 128;
    public int NFft { get; init; } = 2_048;
    public int HopLength { get; init; } = 420;

    // ── Latent CFM ──
    public int LatentDim { get; init; } = 64;
    public float LatentScale { get; init; } = 5f;        // lcfm_z_scale
    public int WnLayers { get; init; } = 30;
    public int WnHidden { get; init; } = 512;
    public int WnKernel { get; init; } = 3;
    public int WnDilationCycle { get; init; } = 5;
    public int TimeEmbDim { get; init; } = 128;

    // ── IRMAE autoencoder ──
    public int AeHidden { get; init; } = 1_024;
    public int AeResBlocks { get; init; } = 4;
    public IReadOnlyList<int> AeDilations { get; init; } = [1, 2, 4, 8];
    public float NormEps { get; init; } = 1e-5f;

    // ── Solver / inference ──
    public int Nfe { get; init; } = 64;                  // function evals
    public string Solver { get; init; } = "midpoint";    // euler / midpoint / rk4
    public float Lambd { get; init; } = 1.0f;            // denoiser blend (CLI default value)
    public float Tau { get; init; } = 0.5f;              // prior temperature
    public int TimeMappingDivisor { get; init; } = 4;

    // ── Denoiser (UNet) ──
    /// <summary>Base hidden channel width of the denoiser UNet.</summary>
    public int DenoiserHiddenDim { get; init; } = 16;
    /// <summary>Number of down/up sampling blocks in the denoiser UNet.</summary>
    public int DenoiserNumBlocks { get; init; } = 4;
    /// <summary>Number of bottleneck (middle) blocks in the denoiser UNet.</summary>
    public int DenoiserNumMiddleBlocks { get; init; } = 2;
    /// <summary>Convolution kernel size used throughout the denoiser UNet.</summary>
    public int DenoiserKernel { get; init; } = 3;

    // ── UnivNet vocoder ──
    /// <summary>UnivNet base channel count (nc).</summary>
    public int UnivNetNc { get; init; } = 96;
    /// <summary>Extra latent dimension fed into the vocoder.</summary>
    public int VocoderExtraDim { get; init; } = 32;

    // ── IRMAE / latent ──
    /// <summary>Number of intermediate residual MLP (IRM) layers in the IRMAE.</summary>
    public int NumIrms { get; init; } = 4;
    /// <summary>Group count for IRMAE GroupNorm layers.</summary>
    public int IrmaeGroupNorm { get; init; } = 32;
    /// <summary>Force a Gaussian prior on the IRMAE latent.</summary>
    public bool ForceGaussianPrior { get; init; } = false;

    // ── CFM / STFT ──
    /// <summary>Minimum sigma for the OT-CFM probability path.</summary>
    public float CfmSigma { get; init; } = 1e-4f;
    /// <summary>STFT window size (samples).</summary>
    public int WinSize { get; init; } = 2_048;
    /// <summary>Floor applied to STFT magnitudes before log scaling.</summary>
    public float StftMagnitudeMin { get; init; } = 1e-4f;
    /// <summary>Pre-emphasis coefficient applied to the input waveform.</summary>
    public float Preemphasis { get; init; } = 0.97f;

    public static ResembleEnhanceConfig Default => new();
}

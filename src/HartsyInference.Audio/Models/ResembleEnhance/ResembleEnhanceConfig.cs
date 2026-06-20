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
    public float Lambd { get; init; } = 0.5f;            // denoiser blend
    public float Tau { get; init; } = 0.5f;              // prior temperature
    public int TimeMappingDivisor { get; init; } = 4;

    public static ResembleEnhanceConfig Default => new();
}

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for ACE-Step v1 (3.5B, Apache-2.0) — the flow-matching music DiT
/// (<c>ACEStepTransformer2DModel</c>): 24 layers × 20 heads × 128 head-dim (inner 2560), Sana-style LiteLA linear
/// self-attention + softmax cross-attention over [speaker ‖ UMT5 text ‖ Conformer lyrics], GLUMBConv FFN
/// (mlp_ratio 2.5), patch [16,1] collapsing the 16-tall mel-bin axis so tokens run 1-D along time (~10.77 Hz).
/// Defaults verbatim from <c>ace_step_transformer/config.json</c>. See <c>docs/Research/ACE_STEP_ARCHITECTURE.md</c>.</summary>
public sealed record AceStepConfig
{
    /// <summary>Latent channels in/out (Music-DCAE).</summary>
    public int InChannels { get; init; } = 8;

    /// <summary>Attention heads.</summary>
    public int NumHeads { get; init; } = 20;

    /// <summary>Per-head dim.</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Inner model dim (= heads · head dim).</summary>
    public int InnerDim => NumHeads * HeadDim;

    /// <summary>DiT blocks.</summary>
    public int NumLayers { get; init; } = 24;

    /// <summary>GLUMBConv expansion ratio (hidden = dim · ratio).</summary>
    public float MlpRatio { get; init; } = 2.5f;

    /// <summary>Patch size (mel-bin height, time width).</summary>
    public (int H, int W) PatchSize { get; init; } = (16, 1);

    /// <summary>Latent grid height (128 mel bins / 8).</summary>
    public int LatentHeight { get; init; } = 16;

    /// <summary>RoPE base.</summary>
    public double RopeTheta { get; init; } = 1_000_000.0;

    /// <summary>UMT5 text feature width.</summary>
    public int TextDim { get; init; } = 768;

    /// <summary>Speaker/timbre vector width.</summary>
    public int SpeakerDim { get; init; } = 512;

    /// <summary>Conformer lyric hidden width.</summary>
    public int LyricHiddenDim { get; init; } = 1024;

    /// <summary>Lyric BPE vocab (safetensors embedding rows; config says 6693).</summary>
    public int LyricVocabSize { get; init; } = 6693;

    /// <summary>Lyric encoder layers (6 verified from the 3.5B key dump — the research doc's "8" was wrong).</summary>
    public int LyricEncoderLayers { get; init; } = 6;

    /// <summary>Patch-embed intermediate channels (in·256 = 2048 in the real checkpoint; read from weights at load).</summary>
    public int PatchEmbedHidden { get; init; } = 2048;

    /// <summary>Timestep sinusoidal frequency dim.</summary>
    public int FreqDim { get; init; } = 256;

    /// <summary>Flow-match shift.</summary>
    public float FlowShift { get; init; } = 3.0f;

    /// <summary>Default inference steps (upstream <c>pipeline_ace_step.py</c> default).</summary>
    public int NumInferenceSteps { get; init; } = 60;

    /// <summary>Default guidance scale (upstream default 15.0).</summary>
    public float GuidanceScale { get; init; } = 15.0f;

    /// <summary>Fraction of the trajectory (centered) over which CFG is applied; outside it the plain conditional
    /// velocity is used (upstream <c>guidance_interval</c> = 0.5).</summary>
    public float GuidanceInterval { get; init; } = 0.5f;

    /// <summary>Linear decay of the guidance scale across the guidance interval toward <see cref="MinGuidanceScale"/>
    /// (upstream <c>guidance_interval_decay</c> = 0.0 = no decay).</summary>
    public float GuidanceIntervalDecay { get; init; } = 0.0f;

    /// <summary>Floor the guidance scale decays toward (upstream <c>min_guidance_scale</c> = 3.0).</summary>
    public float MinGuidanceScale { get; init; } = 3.0f;

    /// <summary>Granularity / "ERG-diffusion" rescale applied in the scheduler step (upstream <c>omega_scale</c> = 10.0).
    /// The per-step update rescales the non-mean component of the Euler delta by
    /// <c>0.9 + 0.2/(1 + e^(−0.1·omega))</c>, mean-preserving.</summary>
    public float OmegaScale { get; init; } = 10.0f;

    /// <summary>Double-condition CFG text-guidance scale (upstream <c>guidance_scale_text</c> = 0.0 = disabled).
    /// Enabled only when both this and <see cref="GuidanceScaleLyric"/> exceed 1.0.</summary>
    public float GuidanceScaleText { get; init; } = 0.0f;

    /// <summary>Double-condition CFG lyric-guidance scale (upstream <c>guidance_scale_lyric</c> = 0.0 = disabled).</summary>
    public float GuidanceScaleLyric { get; init; } = 0.0f;

    /// <summary>DCAE pipeline latent scale (NOT the diffusers 0.41407 — see research § 7).</summary>
    public float LatentScaleFactor { get; init; } = 0.1786f;

    /// <summary>DCAE pipeline latent shift.</summary>
    public float LatentShiftFactor { get; init; } = -1.9091f;

    /// <summary>Output sample rate.</summary>
    public int SampleRate { get; init; } = 44100;

    /// <summary>Mel hop (samples per mel frame).</summary>
    public int HopLength { get; init; } = 512;

    /// <summary>DCAE time downsample (mel frames per latent frame).</summary>
    public int DcaeTimeDownsample { get; init; } = 8;

    /// <summary>Latent time frames for a duration: <c>F_lat = duration · 44100 / 512 / 8</c> (~10.77 Hz).</summary>
    public int LatentFrames(double durationSeconds) =>
        (int)(durationSeconds * SampleRate / HopLength / DcaeTimeDownsample);

    /// <summary>The published v1 3.5B model.</summary>
    public static AceStepConfig V1 => new();
}

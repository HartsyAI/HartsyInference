using SharpInference.Audio.Models.Kokoro;

namespace SharpInference.Audio.Models.StyleTts2;

/// <summary>Configuration for StyleTTS 2 (`yl4579/StyleTTS2-LibriTTS` multi-speaker /
/// `StyleTTS2-LJSpeech` single-speaker). StyleTTS 2 is the parent architecture of Kokoro: the PLBERT,
/// text encoder, prosody predictor, and iSTFTNet decoder are <b>bit-identical in topology</b> and are
/// reused from the Kokoro modules. What StyleTTS 2 adds (and Kokoro removed) is the runtime style path:
/// two <see cref="StyleEncoder"/>s (acoustic + prosodic, 128-d each) that extract a 256-d style vector
/// from a reference mel, and a <see cref="StyleDiffusionSampler"/> (Karras KDiffusion + ADPM2) that
/// denoises a 256-d Gaussian latent into a style vector conditioned on PLBERT text features.
/// See <c>docs/Research/STYLETTS2_ARCHITECTURE.md</c>.</summary>
public sealed record StyleTts2Config
{
    /// <summary>The shared Kokoro/StyleTTS2 backbone config (hidden 512, 3 layers, style_dim 128,
    /// 178 tokens, max_dur 50, iSTFTNet decoder). Reused verbatim.</summary>
    public KokoroConfig Backbone { get; init; } = KokoroConfig.V1;

    /// <summary>Full style-vector dim — <c>2 × style_dim = 256</c> (acoustic ⊕ prosodic).</summary>
    public int StyleDim => Backbone.StyleDim * 2;

    /// <summary>Diffusion style sampler — Karras schedule + ADPM2 second-order sampler.</summary>
    public int DiffusionSteps { get; init; } = 5;
    public float SigmaMin { get; init; } = 1e-4f;
    public float SigmaMax { get; init; } = 3.0f;
    public float SigmaRho { get; init; } = 9.0f;
    public float SigmaData { get; init; } = 0.2f;

    /// <summary>Classifier-free-guidance scale for the style sampler: <c>out = (1−s)·uncond + s·cond</c>.
    /// 1.0 = use the conditional output directly; higher = more text-faithful, less diverse.</summary>
    public float EmbeddingScale { get; init; } = 1.0f;

    /// <summary>True for the multi-speaker LibriTTS checkpoint (has <c>style_encoder</c> +
    /// <c>predictor_encoder</c> for zero-shot cloning, and the style-modulated <c>StyleTransformer1d</c>
    /// diffusion variant). False for single-speaker LJSpeech (diffusion-only, no speech encoders).</summary>
    public bool MultiSpeaker { get; init; } = true;

    public static StyleTts2Config LibriTts => new() { MultiSpeaker = true };
    public static StyleTts2Config LjSpeech => new() { MultiSpeaker = false, DiffusionSteps = 5, EmbeddingScale = 1.5f };
}

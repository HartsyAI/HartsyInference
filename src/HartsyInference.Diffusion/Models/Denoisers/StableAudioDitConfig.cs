namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for the Stable Audio Open DiT (Stability AI's timing-conditioned text-to-audio /
/// sound-effect transformer over Oobleck-VAE latents: 64-ch, ~21.5 Hz, 44.1 kHz stereo). Single-stream 1-D DiT
/// with self-attention over the latent tokens, cross-attention to the T5-base prompt + Fourier timing tokens,
/// and a prepended global (timestep + timing) token — NO per-block AdaLN in the shipped checkpoints (this is
/// the <c>stable_audio_tools</c> <c>global_cond_type="prepend"</c> path, not diffusers' AdaLN wrapper).
///
/// <para>Small dims are VERIFIED against the real <c>FastVideo/stable-audio-open-small-Diffusers</c> safetensors
/// (key dump + cross-checked against the upstream <c>stable_audio_tools</c> <c>DiffusionTransformer</c> /
/// <c>ContinuousTransformer</c> / <c>Attention</c> source, not just the diffusers port): 1024/16L/8h (head_dim
/// 128), symmetric 8/8 cross-attn KV, true per-head QK-LayerNorm (eps 1e-6, weight+bias — <c>qk_norm: "ln"</c>),
/// RoPE dim 64 (verified <c>rotary_pos_emb.inv_freq</c>, θ=10000, split-half/NeoX pairing — NOT interleaved
/// pairs), <c>seconds_total</c>-only timing (max 256s, verified from <c>conditioner/config.json</c>), rectified
/// flow, pingpong 8-step no-CFG. Open 1.0 dims (1536/24L/24h, head_dim 64, 12 cross-attn KV heads,
/// seconds_start+total max 512s, dpmpp-3m-sde 100-step CFG 7) are NOT weight-verified — carried from
/// <c>docs/Research/STABLE_AUDIO_ARCHITECTURE.md</c> pending real-checkpoint validation.</para></summary>
public sealed record StableAudioDitConfig
{
    /// <summary>Model width (embed_dim = heads · head_dim).</summary>
    public int HiddenSize { get; init; } = 1024;

    /// <summary>DiT layers (depth).</summary>
    public int NumLayers { get; init; } = 16;

    /// <summary>Attention heads (self-attention Q/K/V and cross-attention Q).</summary>
    public int NumHeads { get; init; } = 8;

    /// <summary>Cross-attention key/value heads. Symmetric with <see cref="NumHeads"/> on Open Small; Open 1.0
    /// uses 12 (MQA-style, repeat-interleaved up to <see cref="NumHeads"/>=24 at attention time).</summary>
    public int CrossKvHeads { get; init; } = 8;

    /// <summary>Per-head dim (HiddenSize / NumHeads).</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>SwiGLU FFN inner dim (the packed <c>ff.0.proj</c> is <c>2 · FfInner</c> = gate ‖ up).</summary>
    public int FfInner { get; init; } = 4096;

    /// <summary>RoPE rotary dim per head (= <c>2 · inv_freq.length</c>); applied to the leading channels of each
    /// head over the latent-token axis only (never the prepended global token or the cross-attn tokens).</summary>
    public int RopeDim { get; init; } = 64;

    /// <summary>RoPE base frequency.</summary>
    public double RopeTheta { get; init; } = 10_000.0;

    /// <summary>Per-head QK LayerNorm (weight+bias over <see cref="HeadDim"/>). On for Small; off for Open 1.0.</summary>
    public bool QkNorm { get; init; } = true;

    /// <summary>QK-LayerNorm epsilon — fixed at 1e-6 in <c>stable_audio_tools.models.transformer.Attention</c>
    /// regardless of <see cref="LayerNormEps"/> (which governs the block-level pre/cross-attend/ff norms).</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    /// <summary>Oobleck VAE latent channels = DiT I/O channels (<c>preprocess_conv</c>/<c>postprocess_conv</c> +
    /// <c>project_in</c>/<c>project_out</c>).</summary>
    public int IoChannels { get; init; } = 64;

    /// <summary>Cross-attention conditioning token dim (T5-base hidden = 768; projected to HiddenSize by
    /// <c>to_cond_embed</c>).</summary>
    public int CondTokenDim { get; init; } = 768;

    /// <summary>Global conditioning dim (the concatenated timing Fourier tokens; projected to HiddenSize by
    /// <c>to_global_embed</c> and summed with the timestep embedding into the prepended global token).</summary>
    public int GlobalCondDim { get; init; } = 768;

    /// <summary>Timestep Fourier-feature dim (<c>timestep_features</c> has this/2 frequencies → sin‖cos; fed to
    /// <c>to_timestep_embed</c> 256→HiddenSize→HiddenSize).</summary>
    public int TimestepFeaturesDim { get; init; } = 256;

    /// <summary>LayerNorm epsilon (bias-ful LN throughout — pre_norm / cross_attend_norm / ff_norm).</summary>
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>Whether the model conditions on <c>seconds_start</c> in addition to <c>seconds_total</c>. Off on
    /// Open Small (always starts at 0 → one timing token); on for Open 1.0 (two timing tokens).</summary>
    public bool HasSecondsStart { get; init; } = false;

    /// <summary>Timing scalar clamp/normalization range (seconds): value → clamp[0,max] → /max → Fourier.
    /// VERIFIED 256 for Open Small (<c>conditioner/config.json</c> <c>seconds_total.max_val</c>); Open 1.0's
    /// 512 is the unverified <c>stable_audio_tools</c> default.</summary>
    public double TimingMaxSeconds { get; init; } = 256.0;

    // ── VAE / audio geometry (shared across variants) ──
    public int SampleRate { get; init; } = 44_100;
    public int AudioChannels { get; init; } = 2;
    /// <summary>Audio samples per latent step (2·4·4·8·8) → latent rate 44100/2048 ≈ 21.533 Hz.</summary>
    public int VaeDownsample { get; init; } = 2048;

    /// <summary>Max latent tokens (trained length): Small 256 (~11.89 s), Open 1.0 1024 (~47.55 s).</summary>
    public int MaxLatentTokens { get; init; } = 256;

    /// <summary>Stable Audio Open Small — 341M DiT, rectified-flow + ARC, pingpong 8 steps, CFG off, ~11.89 s.</summary>
    public static StableAudioDitConfig OpenSmall => new();

    /// <summary>Stable Audio Open 1.0 — 1.06B DiT, EDM v-prediction, dpmpp-3m-sde 100 steps, CFG 7, ~47.55 s.</summary>
    public static StableAudioDitConfig Open1_0 => new()
    {
        HiddenSize = 1536, NumLayers = 24, NumHeads = 24, CrossKvHeads = 12, HeadDim = 64,
        FfInner = 6144, RopeDim = 32, QkNorm = false, GlobalCondDim = 1536,
        HasSecondsStart = true, MaxLatentTokens = 1024, TimingMaxSeconds = 512.0,
    };
}

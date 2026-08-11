namespace HartsyInference.Audio.Models.VibeVoice;

/// <summary>Configuration for VibeVoice (Microsoft Research, 2025) — long-form multi-speaker
/// TTS built on Qwen2.5 + next-token-diffusion. Fields verified against the public
/// safetensors layout of <c>microsoft/VibeVoice-1.5B</c>, <c>microsoft/VibeVoice-Large</c>,
/// and <c>microsoft/VibeVoice-Realtime-0.5B</c>.
///
/// <para>VibeVoice is a composition: a Qwen2.5 LM (1.5B / 7B / 0.5B-streaming) drives an
/// autoregressive loop over a 5-token constrained vocabulary
/// {<c>speech_start</c>, <c>speech_end</c>, <c>speech_diffusion</c>, <c>eos</c>, <c>bos</c>}.
/// Each <c>speech_diffusion</c> token triggers a 20-step DDPM denoise (cosine schedule,
/// v-prediction, CFG) of a single 64-d continuous latent. The latent is decoded by a
/// causal 1D-ConvNeXt acoustic VAE at 7.5 Hz (3200× downsample of 24 kHz audio), the
/// decoded chunk is re-encoded through a semantic VAE, and both connectors feed back into
/// the LM as the next input embedding.</para>
///
/// <para>Three official variants share the same VAE / diffusion-head topology; they differ
/// in the Qwen LM size, context window, and (for the streaming variant) whether the
/// semantic VAE is present and whether the LM is split.</para></summary>
public sealed record VibeVoiceConfig
{
    /// <summary>Master model_type discriminator. <c>"vibevoice"</c> for multi-speaker
    /// (1.5B / 7B), <c>"vibevoice_streaming"</c> for the 0.5B real-time variant.</summary>
    public required string ModelType { get; init; }

    /// <summary>Acoustic VAE (encoder + decoder, 64-d latent, 7.5 Hz frame rate).</summary>
    public required VibeVoiceTokenizerConfig AcousticTokenizer { get; init; }

    /// <summary>Semantic VAE (encoder only, 128-d, deterministic mean). <c>null</c> on
    /// the streaming variant which has no semantic feedback.</summary>
    public VibeVoiceTokenizerConfig? SemanticTokenizer { get; init; }

    /// <summary>Qwen2.5 LM backbone shape. The actual forward is provided by the dotLLM
    /// dependency at pipeline-construction time; this record only stores the dimensions
    /// needed to size connectors and the diffusion head.</summary>
    public required VibeVoiceDecoderConfig Decoder { get; init; }

    /// <summary>4-layer DDPM denoiser conditioned on the LM's per-step hidden state.</summary>
    public required VibeVoiceDiffusionHeadConfig DiffusionHead { get; init; }

    /// <summary>Acoustic latent channels. Always 64 for the official checkpoints.</summary>
    public int AcousticVaeDim { get; init; } = 64;

    /// <summary>Semantic latent channels. Always 128 for the multi-speaker checkpoints.
    /// Unused on the streaming variant.</summary>
    public int SemanticVaeDim { get; init; } = 128;

    /// <summary>Streaming-only: number of upper Qwen layers used for TTS (the lower
    /// <c>num_hidden_layers - tts_backbone_num_hidden_layers</c> are text-only). Ignored
    /// on the multi-speaker variant. Default 20 per <c>VibeVoice-Streaming-0.5B</c>.</summary>
    public int TtsBackboneNumHiddenLayers { get; init; } = 20;

    /// <summary>Classifier-free guidance scale for the diffusion head. The denoise step
    /// forms <c>eps = uncond + CfgScale*(cond - uncond)</c> from a positive (text+voice
    /// conditioned) and a negative (voice/acoustic-only, text-masked) LM stream. 1.3 is the
    /// upstream demo default; 1.0 disables guidance (single-stream).</summary>
    public float CfgScale { get; init; } = 1.3f;

    /// <summary>AR token selection. Upstream uses greedy (<c>do_sample=False</c>) once CFG
    /// is enabled — the diffusion head stops cleanly. Set true to restore temperature/top-p
    /// stochastic sampling.</summary>
    public bool DoSample { get; init; } = false;

    /// <summary>Convenience accessor — true for the streaming variant.</summary>
    public bool IsStreaming => ModelType == "vibevoice_streaming";

    /// <summary><c>VibeVoice-1.5B</c> preset — Qwen2.5-1.5B backbone, 65k context, up to 4
    /// speakers, ~90 min output.</summary>
    public static VibeVoiceConfig V15B => new()
    {
        ModelType = "vibevoice",
        AcousticTokenizer = VibeVoiceTokenizerConfig.AcousticDefault,
        SemanticTokenizer = VibeVoiceTokenizerConfig.SemanticDefault,
        Decoder = VibeVoiceDecoderConfig.Qwen25_1_5B,
        DiffusionHead = VibeVoiceDiffusionHeadConfig.Default with { HiddenSize = 1_536 },
    };

    /// <summary><c>VibeVoice-Large</c> (7B) preset — Qwen2.5-7B backbone, 32k context, up to
    /// 4 speakers, ~45 min output.</summary>
    public static VibeVoiceConfig V7B => new()
    {
        ModelType = "vibevoice",
        AcousticTokenizer = VibeVoiceTokenizerConfig.AcousticDefault,
        SemanticTokenizer = VibeVoiceTokenizerConfig.SemanticDefault,
        Decoder = VibeVoiceDecoderConfig.Qwen25_7B,
        DiffusionHead = VibeVoiceDiffusionHeadConfig.Default with { HiddenSize = 3_584 },
    };

    /// <summary><c>VibeVoice-Streaming-0.5B</c> preset — Qwen2.5-0.5B split-LM, single
    /// speaker, no semantic VAE, binary EOS classifier.</summary>
    public static VibeVoiceConfig Streaming05B => new()
    {
        ModelType = "vibevoice_streaming",
        AcousticTokenizer = VibeVoiceTokenizerConfig.AcousticDefault,
        SemanticTokenizer = null,
        Decoder = VibeVoiceDecoderConfig.Qwen25_0_5B,
        // 20, not the Default's 10 — the real diffusion_head_config.ddpm_num_inference_steps in this
        // checkpoint's config.json (confirmed 2026-08-10 against the actual microsoft/VibeVoice-Realtime-0.5B
        // config; the 1.5B/7B variants' own inference script default of 10 does not apply here).
        DiffusionHead = VibeVoiceDiffusionHeadConfig.Default with { HiddenSize = 896, DdpmNumInferenceSteps = 20 },
        TtsBackboneNumHiddenLayers = 20,
    };
}

/// <summary>Shared shape for both the acoustic VAE (encoder+decoder) and the semantic VAE
/// (encoder only). Differences are encoded in <see cref="VaeDim"/>, <see cref="FixStd"/>,
/// and <see cref="StdDistType"/>; all other fields are identical for the official
/// checkpoints.
///
/// <para><see cref="EncoderRatios"/> is the downsample chain in the public config order;
/// the encoder forward reverses it internally (Python's <c>list(reversed(...))</c>).
/// Total downsample = product = 3200 → 7.5 Hz on 24 kHz input.</para></summary>
public sealed record VibeVoiceTokenizerConfig
{
    /// <summary>Latent dimensionality. 64 for acoustic, 128 for semantic.</summary>
    public int VaeDim { get; init; } = 64;

    /// <summary>Fixed std for the Gaussian sampling at the VAE output. 0.5 (acoustic), 0
    /// (semantic — deterministic).</summary>
    public float FixStd { get; init; } = 0.5f;

    /// <summary>Sampling distribution. <c>"gaussian"</c> draws a per-batch std multiplier
    /// from <c>N(0, fix_std / 0.8)</c> on top of the fixed std (acoustic); <c>"none"</c>
    /// returns the deterministic mean (semantic); <c>"fix"</c> uses the fixed std.</summary>
    public string StdDistType { get; init; } = "gaussian";

    /// <summary>Input audio channels. Always 1 (mono 24 kHz).</summary>
    public int Channels { get; init; } = 1;

    /// <summary>Causal convolutions everywhere. Required by streaming decode/encode.</summary>
    public bool Causal { get; init; } = true;

    /// <summary>Stem filter count. Both encoder + decoder use 32 in the official checkpoints.</summary>
    public int EncoderNFilters { get; init; } = 32;

    /// <summary>Decoder stem filter count. 32 for the acoustic decoder.</summary>
    public int DecoderNFilters { get; init; } = 32;

    /// <summary>Per-stage downsample factors. Forward-order list; the encoder consumes
    /// <c>reversed(ratios)</c> internally so that low-rate dims are crossed last. Product
    /// = 3200.</summary>
    public IReadOnlyList<int> EncoderRatios { get; init; } = [8, 5, 5, 4, 2, 2];

    /// <summary>Per-stage upsample factors for the acoustic decoder. Same as
    /// <see cref="EncoderRatios"/> for the canonical checkpoints. Null on the semantic
    /// VAE (encoder only — no decoder).</summary>
    public IReadOnlyList<int>? DecoderRatios { get; init; } = [8, 5, 5, 4, 2, 2];

    /// <summary>Per-stage ConvNeXt block depths. Encoded as a dash-separated string
    /// matching the Python config <c>"3-3-3-3-3-3-8"</c> — 7 entries (stem stage + 6
    /// downsample stages), the last stage runs 8 blocks at the smallest temporal rate.</summary>
    public string EncoderDepths { get; init; } = "3-3-3-3-3-3-8";

    /// <summary>Optional override for decoder depths. <c>null</c> → reverse of encoder
    /// depths. Acoustic decoder follows this default; semantic VAE has no decoder.</summary>
    public string? DecoderDepths { get; init; }

    /// <summary>Mixer type inside <see cref="VibeVoiceConvNeXtBlock"/>. Always
    /// <c>"depthwise_conv"</c> for the official checkpoints — full Conv1D with groups=channels.</summary>
    public string MixerLayer { get; init; } = "depthwise_conv";

    /// <summary>Normalization in convolutional layers. <c>"none"</c> means no weight-norm
    /// / spectral-norm wrapper (just plain Conv1d).</summary>
    public string ConvNorm { get; init; } = "none";

    /// <summary>Conv padding mode. <c>"constant"</c> = zero-pad. Not <c>"reflect"</c>.</summary>
    public string PadMode { get; init; } = "constant";

    /// <summary>Conv bias enabled.</summary>
    public bool ConvBias { get; init; } = true;

    /// <summary>If true, skip the pre-head normalization between the last stage and the
    /// final 1×1 conv. The published checkpoints set this true.</summary>
    public bool DisableLastNorm { get; init; } = true;

    /// <summary>Block normalization style. <c>"RMSNorm"</c> for VibeVoice.</summary>
    public string LayerNorm { get; init; } = "RMSNorm";

    /// <summary>RMSNorm epsilon. 1e-5 per the public configs.</summary>
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>RMSNorm has learnable per-channel scale.</summary>
    public bool LayerNormElementwiseAffine { get; init; } = true;

    /// <summary>ConvNeXt layer-scale init. 1e-6 in the published configs.</summary>
    public float LayerScaleInitValue { get; init; } = 1e-6f;

    /// <summary>Initial value for <c>nn.init.normal_(weight, std=...)</c> in the Python
    /// reference. Inference doesn't use it — kept for parity / parameter-count audits.</summary>
    public float WeightInitValue { get; init; } = 1e-2f;

    /// <summary>Per-corpus normalization scale applied to input audio before encoding. 0.0
    /// disables it (no normalization) for the published checkpoints.</summary>
    public float CorpusNormalize { get; init; } = 0.0f;

    /// <summary>Acoustic VAE preset (encoder + decoder, vae_dim=64, gaussian sampling).</summary>
    public static VibeVoiceTokenizerConfig AcousticDefault => new();

    /// <summary>Semantic VAE preset (encoder only, vae_dim=128, deterministic).</summary>
    public static VibeVoiceTokenizerConfig SemanticDefault => new()
    {
        VaeDim = 128,
        FixStd = 0f,
        StdDistType = "none",
        DecoderRatios = null,
        DecoderDepths = null,
    };
}

/// <summary>Shape of the Qwen2.5 LM that VibeVoice wraps. We only carry the dimensions
/// needed to size the speech connectors and the diffusion head — the actual LM forward
/// is supplied by the dotLLM dependency.</summary>
public sealed record VibeVoiceDecoderConfig
{
    /// <summary>LM hidden size. 1536 (1.5B), 3584 (7B), 896 (0.5B streaming).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Total LM transformer layers (before any split). 28 / 28 / 24 respectively.</summary>
    public required int NumHiddenLayers { get; init; }

    /// <summary>Per-layer attention heads. 12 / 28 / 14.</summary>
    public required int NumAttentionHeads { get; init; }

    /// <summary>GQA key/value heads. 2 / 4 / 2.</summary>
    public required int NumKeyValueHeads { get; init; }

    /// <summary>FFN inner dim. 8960 / 18944 / 4864.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>RoPE theta. 1e6 across all VibeVoice variants (long-context preset).</summary>
    public float RopeTheta { get; init; } = 1_000_000f;

    /// <summary>Maximum positions. 65536 (1.5B) / 32768 (7B) / 8192 (0.5B realtime).</summary>
    public required int MaxPositionEmbeddings { get; init; }

    /// <summary>Qwen2.5 vocab size — 151936 (1.5B / 0.5B) or 152064 (7B). VibeVoice does
    /// NOT extend this vocab; speech tokens are remapped onto Qwen's existing
    /// <c>&lt;|vision_start|&gt;</c> / <c>&lt;|vision_end|&gt;</c> / <c>&lt;|vision_pad|&gt;</c>.</summary>
    public required int VocabSize { get; init; }

    /// <summary>Whether <c>lm_head.weight</c> is tied to <c>embed_tokens.weight</c>. True for
    /// 1.5B, false for 7B and the 0.5B realtime variant.</summary>
    public bool TieWordEmbeddings { get; init; } = true;

    /// <summary>RMSNorm epsilon inside the LM. 1e-6 (Qwen2 default).</summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>Qwen2.5-1.5B preset (multi-speaker 1.5B variant).</summary>
    public static VibeVoiceDecoderConfig Qwen25_1_5B => new()
    {
        HiddenSize = 1_536,
        NumHiddenLayers = 28,
        NumAttentionHeads = 12,
        NumKeyValueHeads = 2,
        IntermediateSize = 8_960,
        MaxPositionEmbeddings = 65_536,
        VocabSize = 151_936,
        TieWordEmbeddings = true,
    };

    /// <summary>Qwen2.5-7B preset (multi-speaker Large variant).</summary>
    public static VibeVoiceDecoderConfig Qwen25_7B => new()
    {
        HiddenSize = 3_584,
        NumHiddenLayers = 28,
        NumAttentionHeads = 28,
        NumKeyValueHeads = 4,
        IntermediateSize = 18_944,
        MaxPositionEmbeddings = 32_768,
        VocabSize = 152_064,
        TieWordEmbeddings = false,
    };

    /// <summary>Qwen2.5-0.5B preset (streaming-only variant). Layer count is total — the
    /// streaming model splits this into lower (4) + upper (20) at runtime per
    /// <see cref="VibeVoiceConfig.TtsBackboneNumHiddenLayers"/>.</summary>
    public static VibeVoiceDecoderConfig Qwen25_0_5B => new()
    {
        HiddenSize = 896,
        NumHiddenLayers = 24,
        NumAttentionHeads = 14,
        NumKeyValueHeads = 2,
        IntermediateSize = 4_864,
        MaxPositionEmbeddings = 8_192,
        VocabSize = 151_936,
        TieWordEmbeddings = false,
    };
}

/// <summary>Shape of the per-token diffusion denoising head. 4-layer FFN-only transformer
/// with AdaLN-Zero modulation, RMSNorm, SwiGLU FFN, zero-init final projection. Hidden
/// dim tracks the LM's hidden size; latent dim equals <see cref="VibeVoiceConfig.AcousticVaeDim"/>.</summary>
public sealed record VibeVoiceDiffusionHeadConfig
{
    /// <summary>Per-layer width. Equal to LM hidden_size — the head reads LM hidden
    /// states directly via the cond projection without dimensionality reduction.</summary>
    public int HiddenSize { get; init; } = 1_536;

    /// <summary>Number of <see cref="VibeVoiceDiffusionHead"/> layers. 4 in the published
    /// checkpoints across all three variants.</summary>
    public int HeadLayers { get; init; } = 4;

    /// <summary>SwiGLU FFN expansion ratio. 3.0× in the published checkpoints (lower than
    /// the typical 4× of DiT blocks).</summary>
    public float HeadFfnRatio { get; init; } = 3.0f;

    /// <summary>RMSNorm epsilon in the diffusion head. 1e-5.</summary>
    public float RmsNormEps { get; init; } = 1e-5f;

    /// <summary>Latent channel count. Mirrors <see cref="VibeVoiceConfig.AcousticVaeDim"/>
    /// — denoting "this head denoises acoustic VAE latents". 64.</summary>
    public int LatentSize { get; init; } = 64;

    /// <summary>Convenience copy of <see cref="VibeVoiceConfig.AcousticVaeDim"/>. The
    /// Python config stores both — we keep the field to ease round-trip JSON loading.</summary>
    public int? SpeechVaeDim { get; init; } = 64;

    /// <summary>Noise-prediction target. Always <c>"v_prediction"</c> for the published
    /// VibeVoice checkpoints. <c>"epsilon"</c> is supported by the upstream code but
    /// unused.</summary>
    public string PredictionType { get; init; } = "v_prediction";

    /// <summary>Diffusion family. <c>"ddpm"</c>.</summary>
    public string DiffusionType { get; init; } = "ddpm";

    /// <summary>Discretized timesteps used at training. 1000.</summary>
    public int DdpmNumSteps { get; init; } = 1_000;

    /// <summary>DDPM steps consumed at inference (DPM-Solver-multistep walks through these). Upstream's
    /// own inference sets <c>set_ddpm_inference_steps(num_steps=10)</c>, so 10 is the documented default;
    /// 5 is reported to degrade quality. Overrideable per request.</summary>
    public int DdpmNumInferenceSteps { get; init; } = 10;

    /// <summary>Noise schedule shape. <c>"cosine"</c> (Nichol-Dhariwal) for VibeVoice.</summary>
    public string DdpmBetaSchedule { get; init; } = "cosine";

    /// <summary>Training-only oversample factor for the diffusion loss; unused at
    /// inference. Kept here so we can round-trip the config JSON without dropping fields.</summary>
    public int DdpmBatchMul { get; init; } = 4;

    /// <summary>Default preset matching the published checkpoints — caller overrides
    /// <see cref="HiddenSize"/> per variant.</summary>
    public static VibeVoiceDiffusionHeadConfig Default => new();
}

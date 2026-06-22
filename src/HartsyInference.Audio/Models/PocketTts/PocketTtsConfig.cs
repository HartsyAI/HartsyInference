using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;

namespace HartsyInference.Audio.Models.PocketTts;

/// <summary>Configuration skeleton for Kyutai's Pocket-TTS (~100M CPU TTS). <b>Not a discrete codec-LM</b> — it
/// is an autoregressive LM over <b>continuous</b> audio latents (a modified Mimi VAE, RVQ bypassed), with each
/// 12.5 Hz frame produced by a per-frame flow-matching / Lagrangian-self-distillation head. 24 kHz, 26 built-in
/// EN voices + zero-shot audio-prompt cloning. See <c>docs/Research/POCKET_TTS_ARCHITECTURE.md</c>.
///
/// <para><b>Config-gated:</b> the released `b6369a24` variant's exact <c>d_model</c>, layer count, and latent
/// dim live in its YAML/safetensors and are <b>NOT public</b> — fields below are the documented structure with
/// placeholder dims to reconcile on first checkpoint load. The build plan (continuous-latent AR + reused
/// <c>ConditionalCfm</c> per frame + the continuous-Mimi path + SentencePiece) is captured; implementation is
/// deferred until the config is read from the checkpoint.</para></summary>
public sealed record PocketTtsConfig
{
    public int SampleRate { get; init; } = 24_000;       // = mimi.sample_rate
    public int FrameRateHz { get; init; } = 12;          // 12.5 Hz Mimi frame rate

    // ── Flow-LM backbone (streaming transformer over continuous latents) — dims NOT public ──
    public int DModel { get; init; } = 0;                // RECONCILE: config.flow_lm.transformer.d_model
    public int NumLayers { get; init; } = 6;             // distilled released model ≈ 6 layers
    public int LatentDim { get; init; } = 0;             // RECONCILE: config.mimi.quantizer.dimension

    /// <summary>Attention query heads of the Flow-LM transformer. RECONCILE: config.flow_lm.transformer.num_heads.</summary>
    public int NumHeads { get; init; } = 0;

    /// <summary>Attention key/value heads (GQA). Defaults to <see cref="NumHeads"/> (MHA) when 0.</summary>
    public int NumKvHeads { get; init; } = 0;

    /// <summary>SwiGLU FFN inner dim of the Flow-LM transformer. RECONCILE: config.flow_lm.transformer.ffn_dim.</summary>
    public int FfnDim { get; init; } = 0;

    /// <summary>Text-token vocabulary size (SentencePiece). RECONCILE: config.flow_lm.conditioner.tokenizer.</summary>
    public int VocabSize { get; init; } = 0;

    /// <summary>RoPE base frequency for the Flow-LM transformer.</summary>
    public float RopeTheta { get; init; } = 10_000f;

    /// <summary>RMSNorm epsilon for the Flow-LM transformer.</summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>Hard cap on text-prefix + primed-prompt + generated frames the KV cache addresses.</summary>
    public int MaxSequenceLength { get; init; } = 4_096;

    // ── Per-frame flow-matching / LSD head ──
    /// <summary>Per-frame latent sampling refinement steps (Lagrangian self-distillation).</summary>
    public int LsdDecodeSteps { get; init; } = 4;

    /// <summary>Hidden width of the per-frame flow head MLP. Defaults to <see cref="DModel"/> when 0.</summary>
    public int FlowHeadDim { get; init; } = 0;

    /// <summary>Latent classifier-free-guidance scale applied inside the per-frame flow solver. ≤ 0 disables CFG.</summary>
    public float LatentCfgScale { get; init; } = 1.0f;

    /// <summary>Sinusoidal time-embedding width fed to the flow head's time conditioning.</summary>
    public int TimeEmbedDim { get; init; } = 0;

    /// <summary>Max generated audio frames per <c>Generate</c> call (safety bound on the AR loop).</summary>
    public int MaxFrames { get; init; } = 1_500;

    /// <summary>Codec config for the continuous-latent Mimi path (24 kHz, 12.5 Hz). RECONCILE: <c>mimi.*</c>.</summary>
    public MimiConfig Mimi { get; init; } = MimiConfig.Mimi24kHz;

    /// <summary>Builds the headless <see cref="Qwen2Config"/> that drives the Flow-LM transformer body. Heads/FFN
    /// fall back to sensible derivations when the (config-gated) real values are still zero, so tiny test configs
    /// and the real checkpoint both produce a valid transformer.</summary>
    public Qwen2Config ToTransformerConfig()
    {
        int heads = NumHeads > 0 ? NumHeads : Math.Max(1, DModel / 64);
        int kv = NumKvHeads > 0 ? NumKvHeads : heads;
        int ffn = FfnDim > 0 ? FfnDim : DModel * 4;
        return new Qwen2Config
        {
            HiddenSize = DModel,
            NumHiddenLayers = NumLayers,
            NumAttentionHeads = heads,
            NumKeyValueHeads = kv,
            IntermediateSize = ffn,
            VocabSize = Math.Max(1, VocabSize),
            MaxPositionEmbeddings = MaxSequenceLength,
            RopeTheta = RopeTheta,
            RmsNormEps = RmsNormEps,
            AttentionBias = false,
            TieWordEmbeddings = true,
        };
    }

    /// <summary>Built-in English voice names (zero-shot cloning also via an audio prompt).</summary>
    public IReadOnlyList<string> Voices { get; init; } =
    [
        "alba", "giovanni", "lola", "juergen", "rafael", "estelle", "anna", "azelma", "bill_boerst",
        "caro_davy", "charles", "cosette", "eponine", "eve", "fantine", "george", "jane", "jean",
        "javert", "marius", "mary", "michael", "paul", "peter_yearsley", "stuart_bell", "vera",
    ];

    public static PocketTtsConfig Default => new();
}

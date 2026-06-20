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

    /// <summary>Per-frame latent sampling refinement steps (Lagrangian self-distillation).</summary>
    public int LsdDecodeSteps { get; init; } = 4;

    /// <summary>Built-in English voice names (zero-shot cloning also via an audio prompt).</summary>
    public IReadOnlyList<string> Voices { get; init; } =
    [
        "alba", "giovanni", "lola", "juergen", "rafael", "estelle", "anna", "azelma", "bill_boerst",
        "caro_davy", "charles", "cosette", "eponine", "eve", "fantine", "george", "jane", "jean",
        "javert", "marius", "mary", "michael", "paul", "peter_yearsley", "stuart_bell", "vera",
    ];

    public static PocketTtsConfig Default => new();
}

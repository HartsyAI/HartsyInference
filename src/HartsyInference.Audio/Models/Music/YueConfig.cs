using HartsyInference.Audio.Models.Codecs.XCodec;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;

namespace HartsyInference.Audio.Models.Music;

/// <summary>Configuration for YuE (乐) — HKUST/M-A-P's open full-song music model. A two-stage LLaMA-2
/// autoregressive pipeline: Stage-1 (7B) emits interleaved codebook-0 tokens for the <b>vocal</b> and
/// <b>accompaniment</b> tracks at 50 Hz (track-decoupled next-token prediction); Stage-2 (~1.5B)
/// upsamples codebook-0 to the full 8 residual codebooks; X-Codec decodes to 16 kHz. See
/// <c>docs/Research/YUE_ARCHITECTURE.md</c>.
///
/// <para><b>Reuse:</b> both LLaMA-2 decoders are plain bias-off Llama bodies → reuse <see cref="Qwen2Model"/>
/// (<see cref="Qwen2Config.AttentionBias"/> = false; Stage-1 is <b>GQA</b> with 4 KV heads per the real
/// config.json, not MHA); sampling reuses <see cref="Sampling.NucleusSampler"/>. The same "LLaMA emits codec
/// tokens" shape as Spark-TTS / CosyVoice.
///
/// <para><b>Parity:</b> Stage-1 LM verified vs real <c>m-a-p/YuE-s1-7B-anneal-en-cot</c> weights (teacher-forced
/// logits corr 1.0, argmax 8/8). The codec decode (<see cref="XCodec"/>) is NOT runnable: the engine's DAC-style
/// XCodec is wrong-architecture for the real SoundStream/EMA-VQ codec — see PARITY_VERIFICATION.md.</para>
///
/// <para><b>Checkpoint-reconciliation pending:</b> the extended-vocab audio-token base IDs come from the
/// YuE tokenizer; they are config fields here. The Vocos 16→44.1 kHz upsampler is deferred.</para></summary>
public sealed record YueConfig
{
    /// <summary>Stage-1 LM: LLaMA-2-7B (32L / 4096 hidden / 32 heads MHA / 11008 FFN / RoPE θ=10k /
    /// no bias) with the YuE-extended vocab.</summary>
    public required Qwen2Config Stage1 { get; init; }

    /// <summary>Stage-2 residual upsampler: ~1.5B LLaMA body.</summary>
    public required Qwen2Config Stage2 { get; init; }

    public int NumCodebooks { get; init; } = 8;
    public int CodebookSize { get; init; } = 1_024;
    public int SampleRate { get; init; } = 16_000;
    public int FrameRateHz { get; init; } = 50;

    // ── Extended-vocab audio-token bases (reconcile against the YuE tokenizer) ──
    public int VocalTokenBase { get; init; } = 45_334;     // <|vocal_cb0_0|> .. (per-track cb0)
    public int AccompTokenBase { get; init; } = 45_334 + 1_024;
    public int AudioEosToken { get; init; } = 32_002;

    public float Temperature { get; init; } = 1.0f;
    public int TopK { get; init; } = 50;
    public float TopP { get; init; } = 0.93f;
    public float RepetitionPenalty { get; init; } = 1.1f;   // mandatory per the YuE README
    /// <summary>Classifier-free guidance scale for stage-1 (YuE uses 1.5 for the first ≤1 segments, 1.2 after).
    /// Applied when a negative/unconditional prompt is supplied; 1.0 disables CFG.</summary>
    public float GuidanceScale { get; init; } = 1.5f;

    public static YueConfig V1 => new()
    {
        // Real m-a-p/YuE-s1-7B-anneal-en-cot config.json: LLaMA-2-7B body but GQA with 4 KV heads
        // (k/v_proj are [512, 4096] = 4 heads x 128 head_dim, NOT MHA) and the YuE-extended vocab is 83968.
        Stage1 = new Qwen2Config
        {
            HiddenSize = 4_096, NumHiddenLayers = 32, NumAttentionHeads = 32, NumKeyValueHeads = 4,
            IntermediateSize = 11_008, VocabSize = 83_968, MaxPositionEmbeddings = 16_384,
            RopeTheta = 10_000f, RmsNormEps = 1e-5f, TieWordEmbeddings = false, AttentionBias = false,
        },
        // Real m-a-p/YuE-s2-1B-general config.json: 32 layers, intermediate 5504, vocab 83840
        // (distinct from the s1 vocab of 83968). NOT numerically verified yet (Stage-2 codec is out of scope).
        Stage2 = new Qwen2Config
        {
            HiddenSize = 2_048, NumHiddenLayers = 32, NumAttentionHeads = 16, NumKeyValueHeads = 16,
            IntermediateSize = 5_504, VocabSize = 83_840, MaxPositionEmbeddings = 8_192,
            RopeTheta = 10_000f, RmsNormEps = 1e-5f, TieWordEmbeddings = false, AttentionBias = false,
        },
    };
}

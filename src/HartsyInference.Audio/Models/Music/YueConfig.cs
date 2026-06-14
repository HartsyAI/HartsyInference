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
/// (<see cref="Qwen2Config.AttentionBias"/> = false, MHA via <c>NumKeyValueHeads == NumAttentionHeads</c>);
/// sampling reuses <see cref="Sampling.NucleusSampler"/>; codec decode reuses the built
/// <see cref="XCodec"/>. The same "LLaMA emits codec tokens → codec decode" shape as Spark-TTS / CosyVoice.</para>
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

    public static YueConfig V1 => new()
    {
        Stage1 = new Qwen2Config
        {
            HiddenSize = 4_096, NumHiddenLayers = 32, NumAttentionHeads = 32, NumKeyValueHeads = 32,
            IntermediateSize = 11_008, VocabSize = 100_000, MaxPositionEmbeddings = 16_384,
            RopeTheta = 10_000f, RmsNormEps = 1e-5f, TieWordEmbeddings = false, AttentionBias = false,
        },
        Stage2 = new Qwen2Config
        {
            HiddenSize = 2_048, NumHiddenLayers = 22, NumAttentionHeads = 16, NumKeyValueHeads = 16,
            IntermediateSize = 5_632, VocabSize = 100_000, MaxPositionEmbeddings = 8_192,
            RopeTheta = 10_000f, RmsNormEps = 1e-5f, TieWordEmbeddings = false, AttentionBias = false,
        },
    };
}

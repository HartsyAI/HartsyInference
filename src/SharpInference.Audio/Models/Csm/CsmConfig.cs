using SharpInference.Audio.Models.LanguageModels.Qwen2;

namespace SharpInference.Audio.Models.Csm;

/// <summary>Configuration for Sesame CSM-1B (`sesame/csm-1b`) — a dual-transformer conversational TTS.
/// A Llama-3.2-1B <b>backbone</b> consumes interleaved text+audio frames and predicts the semantic
/// codebook (0) of the next 80 ms Mimi frame; a small Llama-100M <b>decoder</b> autoregressively predicts
/// the remaining 7 acoustic codebooks of that frame. The 8 codebooks decode to 24 kHz PCM via the Mimi
/// codec. See <c>docs/Research/SESAME_CSM_ARCHITECTURE.md</c>.
///
/// <para><b>Reuse:</b> both transformers are headless Llama-3.2 bodies — reuse <see cref="Qwen2Model"/>
/// with <see cref="Qwen2Config.AttentionBias"/> = false (Llama has no Q/K/V bias). The embedding tables +
/// codebook heads + the backbone→decoder projection live on the outer model. Audio decode reuses the
/// built Mimi codec.</para>
///
/// <para><b>Checkpoint-reconciliation pending:</b> Llama-3.2's RoPE frequency rescaling (scale_factor=32)
/// is not modeled here (plain RoPE θ=500k) — capped at max_seq_len 2048, the divergence is small but
/// should be reconciled. Audio vocab size + key names need the checkpoint.</para></summary>
public sealed record CsmConfig
{
    /// <summary>Backbone: Llama-3.2-1B headless body (16 layers / 2048 hidden / GQA 32:8 / 8192 FFN /
    /// RoPE θ=500k / no attention bias).</summary>
    public required Qwen2Config Backbone { get; init; }

    /// <summary>Audio decoder: Llama-100M headless body (4 layers / 1024 hidden / GQA 8:2 / no bias).</summary>
    public required Qwen2Config Decoder { get; init; }

    public int NumCodebooks { get; init; } = 8;
    public int AudioVocab { get; init; } = 2_051;     // per-codebook (2048 codes + specials)
    public int TextVocab { get; init; } = 128_256;    // Llama-3 tokenizer
    public int SampleRate { get; init; } = 24_000;
    public int FrameSamples { get; init; } = 1_920;   // 80 ms at 24 kHz (12.5 Hz frame rate)

    /// <summary>Audio EOS / end-of-frame marker codebook value.</summary>
    public int AudioEosToken { get; init; } = 2_048;

    public float Temperature { get; init; } = 0.9f;
    public int TopK { get; init; } = 50;
    public float TopP { get; init; } = 1.0f;

    public static CsmConfig V1B => new()
    {
        Backbone = new Qwen2Config
        {
            HiddenSize = 2_048, NumHiddenLayers = 16, NumAttentionHeads = 32, NumKeyValueHeads = 8,
            IntermediateSize = 8_192, VocabSize = 128_256, MaxPositionEmbeddings = 2_048,
            RopeTheta = 500_000f, RmsNormEps = 1e-5f, TieWordEmbeddings = false, AttentionBias = false,
        },
        Decoder = new Qwen2Config
        {
            HiddenSize = 1_024, NumHiddenLayers = 4, NumAttentionHeads = 8, NumKeyValueHeads = 2,
            IntermediateSize = 8_192, VocabSize = 2_051, MaxPositionEmbeddings = 64,
            RopeTheta = 500_000f, RmsNormEps = 1e-5f, TieWordEmbeddings = false, AttentionBias = false,
        },
    };
}

using HartsyInference.Audio.Models.LanguageModels.Qwen2;

namespace HartsyInference.Audio.Models.Csm;

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
/// <para><b>Checkpoint-reconciliation pending:</b> Llama-3.2's RoPE frequency rescaling (Llama3,
/// scale_factor=32) is now modeled on both transformers via <see cref="Qwen2Config.RopeScaling"/>.
/// Audio vocab size + key names still need the checkpoint.</para></summary>
public sealed record CsmConfig
{
    /// <summary>Backbone: Llama-3.2-1B headless body (16 layers / 2048 hidden / GQA 32:8 / 8192 FFN /
    /// RoPE θ=500k / no attention bias).</summary>
    public required Qwen2Config Backbone { get; init; }

    /// <summary>Audio decoder: Llama-100M headless body (4 layers / 1024 hidden / GQA 8:2 / no bias).</summary>
    public required Qwen2Config Decoder { get; init; }

    // PARITY-TODO: verify on real weights; must match the Mimi codec total-codebook count.
    public int NumCodebooks { get; init; } = 32;
    public int AudioVocab { get; init; } = 2_051;     // per-codebook (2048 codes + specials)
    public int TextVocab { get; init; } = 128_256;    // Llama-3 tokenizer
    public int SampleRate { get; init; } = 24_000;
    public int FrameSamples { get; init; } = 1_920;   // 80 ms at 24 kHz (12.5 Hz frame rate)

    /// <summary>Audio EOS / end-of-frame marker codebook value. Note: the reference codebook EOS is 0.</summary>
    public int AudioEosToken { get; init; } = 2_048;

    /// <summary>Codebook padding token id used to fill unset codebook slots.</summary>
    public int CodebookPadToken { get; init; } = 2_050;

    /// <summary>Reference end-of-stream marker codebook value (0).</summary>
    public int CodebookEosToken { get; init; } = 0;

    /// <summary>Text token id marking an audio frame slot in the interleaved sequence.</summary>
    public int AudioTokenId { get; init; } = 128_002;

    /// <summary>Text token id marking the end of a text+audio turn.</summary>
    public int TextAudioEosTokenId { get; init; } = 128_003;

    /// <summary>Text begin-of-sequence token id (Llama-3 tokenizer).</summary>
    public int TextBosTokenId { get; init; } = 128_000;

    /// <summary>Text padding token id (Llama-3 tokenizer).</summary>
    public int TextPadTokenId { get; init; } = 128_002;

    /// <summary>Whether the per-codebook embedding tables are tied to the codebook heads.</summary>
    public bool TieCodebooksEmbeddings { get; init; } = true;

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
            RopeScaling = new HartsyInference.Core.Rope.RopeScaling
            {
                Type = HartsyInference.Core.Rope.RopeScalingType.Llama3, Factor = 32.0,
                OriginalContextLength = 8192, LowFreqFactor = 1.0, HighFreqFactor = 4.0,
            },
        },
        Decoder = new Qwen2Config
        {
            HiddenSize = 1_024, NumHiddenLayers = 4, NumAttentionHeads = 8, NumKeyValueHeads = 2,
            IntermediateSize = 8_192, VocabSize = 2_051, MaxPositionEmbeddings = 33,  // NumCodebooks + 1
            RopeTheta = 500_000f, RmsNormEps = 1e-5f, TieWordEmbeddings = false, AttentionBias = false,
            RopeScaling = new HartsyInference.Core.Rope.RopeScaling
            {
                Type = HartsyInference.Core.Rope.RopeScalingType.Llama3, Factor = 32.0,
                OriginalContextLength = 8192, LowFreqFactor = 1.0, HighFreqFactor = 4.0,
            },
        },
    };
}

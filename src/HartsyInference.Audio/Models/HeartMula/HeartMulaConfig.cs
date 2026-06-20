using HartsyInference.Audio.Models.Csm;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;

namespace HartsyInference.Audio.Models.HeartMula;

/// <summary>Configuration for HeartMuLa (HeartMuLa-oss-3B) music/song generation. Architecturally it is the
/// **CSM/Sesame two-transformer pattern** applied to music: a Llama-3B global backbone predicts codebook 0, a
/// Llama-300M depth decoder predicts the remaining RVQ codebooks (8 total, vocab 8197), conditioned on Llama
/// lyrics tokens + a MuQ-MuLan style embedding, decoded by the HeartCodec (48 kHz / 12.5 Hz, flow-matching
/// decoder). See <c>docs/Research/HEARTMULA_ARCHITECTURE.md</c>.
///
/// <para><b>Reuse:</b> the LM is the already-built+verified <see cref="CsmModel"/> (dual <see cref="Qwen2Model"/>
/// + codebook heads) — HeartMuLa just supplies the music config. HeartCodec (flow-matching codec) + MuQ
/// embedder are net-new (the codec reuses <c>ConditionalCfm</c>).</para></summary>
public sealed record HeartMulaConfig
{
    /// <summary>The CSM-shaped LM config (global Llama-3B backbone + Llama-300M depth decoder, 8 codebooks).</summary>
    public CsmConfig Lm { get; init; } = new()
    {
        Backbone = new Qwen2Config
        {
            HiddenSize = 3_072, NumHiddenLayers = 28, NumAttentionHeads = 24, NumKeyValueHeads = 8,
            IntermediateSize = 8_192, VocabSize = 128_256, MaxPositionEmbeddings = 8_192,
            RopeTheta = 500_000f, RmsNormEps = 1e-5f, AttentionBias = false, TieWordEmbeddings = false,
        },
        Decoder = new Qwen2Config
        {
            HiddenSize = 1_024, NumHiddenLayers = 3, NumAttentionHeads = 16, NumKeyValueHeads = 8,
            IntermediateSize = 4_096, VocabSize = 128_256, MaxPositionEmbeddings = 8_192,
            RopeTheta = 500_000f, RmsNormEps = 1e-5f, AttentionBias = false, TieWordEmbeddings = false,
        },
        NumCodebooks = 8,
        AudioVocab = 8_197,
        TextVocab = 128_256,
        SampleRate = 48_000,
        FrameSamples = 3_840,        // 80 ms at 48 kHz (12.5 Hz frame rate)
        AudioEosToken = 8_192,
    };

    /// <summary>MuQ-MuLan style-conditioning embedding dimension (projected via muq_linear).</summary>
    public int MuqDim { get; init; } = 512;

    // ── HeartCodec (HeartCodec-oss): 48 kHz, 12.5 Hz, 8-codebook RVQ ──
    public int CodecNumQuantizers { get; init; } = 8;
    public int CodecCodebookSize { get; init; } = 8_192;
    public int CodecCodebookDim { get; init; } = 32;

    public static HeartMulaConfig Oss3B => new();
}

using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;

namespace HartsyInference.Audio.Models.Kyutai;

/// <summary>Configuration for Kyutai TTS (`kyutai/tts-1.6b-en_fr`) — a Moshi-style RQ-Transformer: a temporal
/// "Helium" backbone steps once per 12.5 Hz frame and a small <b>depth transformer</b> ("depformer")
/// autoregressively predicts the 32 Mimi codebooks for that frame. Text drives generation through a
/// PAD/EPAD/WORD state machine; voice is a 512-d speaker embedding consumed by cross-attention.
/// See <c>docs/Research/KYUTAI_DSM_ARCHITECTURE.md</c>.
///
/// <para><b>Reuse:</b> the temporal backbone is a Helium decoder → <see cref="Qwen2Model"/> (attn-bias off,
/// driven headless via <c>ForwardEmbeds</c> over a summed text+audio embedding); audio decode reuses the
/// built Mimi (32-codebook DSM variant). Net-new: the depformer and the per-codebook delay roll.</para></summary>
public sealed record KyutaiTtsConfig
{
    /// <summary>Temporal "Helium" backbone (dim 2048 / 16L / 16 heads / SwiGLU / RoPE θ=10000).</summary>
    public required Qwen2Config Temporal { get; init; }

    /// <summary>Depth transformer ("depformer") config.</summary>
    public required MoshiDepthConfig Depth { get; init; }

    /// <summary>Mimi codec (32-codebook DSM variant) used to decode the emitted codes.</summary>
    public MimiConfig Codec { get; init; } = MimiConfig.Mimi24kHzDsm;

    /// <summary>Audio codebooks predicted per frame.</summary>
    public int NumCodebooks { get; init; } = 32;

    /// <summary>Per-codebook audio vocab (2048 Mimi codes).</summary>
    public int CodebookVocab { get; init; } = 2_048;

    /// <summary>Text vocabulary (SentencePiece 8k for en_fr).</summary>
    public int TextCard { get; init; } = 8_000;

    /// <summary>Acoustic-codebook input/output delay in frames (codebooks 1..31; text and codebook 0 = 0).</summary>
    public int AcousticDelay { get; init; } = 2;

    /// <summary>Text-vs-audio stream delay in frames (1.28 s × 12.5 Hz). Text leads the audio.</summary>
    public int StreamDelaySteps { get; init; } = 16;

    /// <summary>Speaker embedding dimension (cross-attention conditioning).</summary>
    public int SpeakerDim { get; init; } = 512;

    /// <summary>Maximum number of speaker embeddings the conditioner accepts.</summary>
    public int MaxSpeakers { get; init; } = 5;

    // ── Sampling defaults ──
    public float Temperature { get; init; } = 0.8f;
    public float TopP { get; init; } = 0.95f;
    public int TopK { get; init; } = 0;

    // ── Token constants (moshi lm.py) ──
    public int TextPad { get; init; } = 3;
    /// <summary>Audio initial/zero code (== codebook size) fed before real codes exist.</summary>
    public int AudioInit { get; init; } = 2_048;
    /// <summary>Text initial token (== text_card).</summary>
    public int TextInit { get; init; } = 8_000;

    /// <summary>Builds the length-(1+NumCodebooks) delay array: text=0, codebook 0=0, codebooks 1..N=AcousticDelay.</summary>
    public int[] BuildDelays()
    {
        int[] d = new int[1 + NumCodebooks];
        for (int k = 1; k < NumCodebooks; k++) d[1 + k] = AcousticDelay;
        return d;
    }

    /// <summary>tts-1.6b-en_fr preset. <b>Reconcile on checkpoint load:</b> exact depformer weight-key
    /// layout (per-step weight sets + low-rank embeddings + multi-linear input projections), the EPAD token
    /// id, the second-stream-ahead lookahead, and the CFG/control LUT conditioners.</summary>
    public static KyutaiTtsConfig Tts1_6B => new()
    {
        Temporal = new Qwen2Config
        {
            HiddenSize = 2_048,
            NumHiddenLayers = 16,
            NumAttentionHeads = 16,
            NumKeyValueHeads = 16,
            IntermediateSize = 8_448,        // hidden_scale 4.125 × 2048
            VocabSize = 8_000,
            MaxPositionEmbeddings = 500,     // 40 s @ 12.5 Hz
            RopeTheta = 10_000f,
            RmsNormEps = 1e-8f,
            AttentionBias = false,
            TieWordEmbeddings = false,
        },
        Depth = MoshiDepthConfig.Tts1_6B,
    };
}

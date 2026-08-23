using HartsyInference.Audio.Models.Csm;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;

namespace HartsyInference.Audio.Models.HeartMula;

/// <summary>Configuration for HeartMuLa (HeartMuLa-oss-3B) music/song generation — the CSM/Sesame two-transformer pattern applied to music. See <c>docs/Research/HEARTMULA_ARCHITECTURE.md</c>.</summary>
// A Llama-3B global backbone predicts codebook 0, a Llama-300M depth decoder predicts the remaining RVQ
// codebooks (8 total, vocab 8197), conditioned on Llama lyrics tokens + a MuQ-MuLan style embedding,
// decoded by the HeartCodec (48 kHz / 12.5 Hz, flow-matching decoder).
// Reuse: the LM is the already-built+verified CsmModel (dual Qwen2Model + codebook heads) — HeartMuLa
// just supplies the music config. HeartCodec (flow-matching codec) + MuQ embedder are net-new (the codec
// reuses ConditionalCfm).
public sealed record HeartMulaConfig
{
    /// <summary>The CSM-shaped LM config (global Llama-3B backbone + Llama-300M depth decoder, 8 codebooks).</summary>
    // Both transformers are torchtune llama3_2 bodies: Llama-3 RoPE rescale (scale_factor=32, low 1, high 4,
    // old_ctx 8192) over θ=500k, and **interleaved** (GPT-J / view_as_complex) RoPE — NOT the HF split-half
    // permuted layout. Verified against the upstream `heartlib` (torchtune `Llama3ScaledRoPE`).
    private static readonly Core.Rope.RopeScaling Llama3Rope = new()
    {
        Type = Core.Rope.RopeScalingType.Llama3, Factor = 32.0,
        LowFreqFactor = 1.0, HighFreqFactor = 4.0, OriginalContextLength = 8_192,
    };

    public CsmConfig Lm { get; init; } = new()
    {
        // Global backbone: torchtune llama-3B (28 layers, 3072 dim, 24 q / 8 kv heads → head_dim 128, FFN 8192).
        Backbone = new Qwen2Config
        {
            HiddenSize = 3_072, NumHiddenLayers = 28, NumAttentionHeads = 24, NumKeyValueHeads = 8,
            IntermediateSize = 8_192, VocabSize = 128_256, MaxPositionEmbeddings = 8_192,
            RopeTheta = 500_000f, RmsNormEps = 1e-5f, AttentionBias = false, TieWordEmbeddings = false,
            RopeScaling = Llama3Rope, Rope = LLM.Transformer.RopeStyle.Interleaved,
        },
        // Depth decoder: torchtune llama-300M — same 3072 dim as the backbone (NOT 1024), 3 layers, 8 q / 4 kv
        // heads → head_dim 384, FFN 8192. The backbone→decoder projection is therefore 3072→3072.
        Decoder = new Qwen2Config
        {
            HiddenSize = 3_072, NumHiddenLayers = 3, NumAttentionHeads = 8, NumKeyValueHeads = 4,
            IntermediateSize = 8_192, VocabSize = 128_256, MaxPositionEmbeddings = 2_048,
            RopeTheta = 500_000f, RmsNormEps = 1e-5f, AttentionBias = false, TieWordEmbeddings = false,
            RopeScaling = Llama3Rope, Rope = LLM.Transformer.RopeStyle.Interleaved,
        },
        NumCodebooks = 8,
        AudioVocab = 8_197,
        TextVocab = 128_256,
        SampleRate = 48_000,
        FrameSamples = 3_840,        // 80 ms at 48 kHz (12.5 Hz frame rate)
        AudioEosToken = 8_193,       // gen_config.audio_eos_id; stop when codebook-0 >= this.
    };

    /// <summary>MuQ-MuLan style-conditioning embedding dimension (projected via muq_linear).</summary>
    public int MuqDim { get; init; } = 512;

    /// <summary>Llama-3 end-of-text token appended to the encoded tags/lyrics token stream (mirrors music_generation.py).</summary>
    public int TextEosToken { get; init; } = 128_001;

    // ── Sampling / guidance (music_generation.py generate defaults) ──
    /// <summary>Sampling temperature (upstream default 1.0, distinct from CSM's 0.9).</summary>
    public float Temperature { get; init; } = 1.0f;
    public int TopK { get; init; } = 50;
    public float TopP { get; init; } = 1.0f;
    /// <summary>Classifier-free guidance scale (upstream <c>cfg_scale</c> = 1.5); applied by re-running the backbone+depth decoder on an unconditional context (no lyrics/style) and blending logits, 1.0 disables it.</summary>
    public float CfgScale { get; init; } = 1.5f;

    // ── HeartCodec (HeartCodec-oss): 48 kHz, 12.5 Hz, 8-codebook RVQ ──
    public int CodecNumQuantizers { get; init; } = 8;
    public int CodecCodebookSize { get; init; } = 8_192;
    public int CodecCodebookDim { get; init; } = 32;

    public static HeartMulaConfig Oss3B => new();
}

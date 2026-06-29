using HartsyInference.Audio.Models.LanguageModels.Qwen2;

namespace HartsyInference.Audio.Models.FishSpeech;

/// <summary>Configuration for Fish-Speech 1.5 (DualAR text2semantic). A Llama-style "slow" backbone predicts
/// the semantic-token stream while a small "fast"/depth transformer predicts the per-frame audio codebooks
/// (firefly-gan-vq, 8 codebooks). See <c>docs/Research/FISH_SPEECH_ARCHITECTURE.md</c>.
///
/// <para><b>Reuse:</b> both transformers are headless Llama → <see cref="Qwen2Model"/>; codebook sampling
/// reuses <c>NucleusSampler</c>; the firefly decoder reuses <c>VitsHiFiGan</c> (SiLU) + <c>Fsq</c>.</para></summary>
public sealed record FishSpeechConfig
{
    /// <summary>Slow backbone (Llama 1024/24L/16h, GQA 2 kv, SwiGLU 4096, RoPE θ=1e6, untied).</summary>
    public Qwen2Config Backbone { get; init; } = new()
    {
        HiddenSize = 1_024, NumHiddenLayers = 24, NumAttentionHeads = 16, NumKeyValueHeads = 2,
        IntermediateSize = 4_096, VocabSize = 102_048, MaxPositionEmbeddings = 8_192,
        RopeTheta = 1_000_000f, RmsNormEps = 1e-6f, AttentionBias = false, TieWordEmbeddings = false,
        Rope = HartsyInference.LLM.Transformer.RopeStyle.Interleaved,
    };

    /// <summary>Fast/depth transformer over the codebook axis (shared weights across depth steps).</summary>
    public Qwen2Config Fast { get; init; } = new()
    {
        HiddenSize = 1_024, NumHiddenLayers = 4, NumAttentionHeads = 16, NumKeyValueHeads = 2,
        IntermediateSize = 4_096, VocabSize = 1_024, MaxPositionEmbeddings = 16,
        RopeTheta = 1_000_000f, RmsNormEps = 1e-6f, AttentionBias = false, TieWordEmbeddings = false,
        Rope = HartsyInference.LLM.Transformer.RopeStyle.Interleaved,
    };

    public int NumCodebooks { get; init; } = 8;
    public int CodebookSize { get; init; } = 1_024;
    public int TextVocab { get; init; } = 102_048;

    /// <summary>Whether to scale the summed text+codebook embedding by <c>1/√(N+1)</c>. The shipped
    /// fish-speech-1.5 checkpoint loads with <c>scale_codebook_embeddings=False</c> (verified at runtime).</summary>
    public bool ScaleCodebookEmbeddings { get; init; } = false;

    // Sampling (inference.py defaults).
    // PARITY-TODO: audit suggests Temperature 0.7 / TopP 0.7 / RepetitionPenalty 1.2 (verify on weights).
    public float Temperature { get; init; } = 1.0f;
    public float TopP { get; init; } = 0.9f;
    public int TopK { get; init; } = 30;
    public float RepetitionPenalty { get; init; } = 1.1f;
    public int MaxNewTokens { get; init; } = 1_500;

    public int SampleRate { get; init; } = 44_100;

    public static FishSpeechConfig V1_5 => new();

    /// <summary>Fish-Speech 1.4 preset: 32k vocab and a 4096 context window (otherwise identical to 1.5).</summary>
    public static FishSpeechConfig V1_4 => new()
    {
        Backbone = new()
        {
            HiddenSize = 1_024, NumHiddenLayers = 24, NumAttentionHeads = 16, NumKeyValueHeads = 2,
            IntermediateSize = 4_096, VocabSize = 32_000, MaxPositionEmbeddings = 4_096,
            RopeTheta = 1_000_000f, RmsNormEps = 1e-6f, AttentionBias = false, TieWordEmbeddings = false,
            Rope = HartsyInference.LLM.Transformer.RopeStyle.Interleaved,
        },
        TextVocab = 32_000,
    };
}

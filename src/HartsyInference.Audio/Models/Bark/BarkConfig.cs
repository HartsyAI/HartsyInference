using HartsyInference.Audio.Models.LanguageModels.Gpt;

namespace HartsyInference.Audio.Models.Bark;

/// <summary>Configuration for Bark (Suno) — a three-stage GPT-2 cascade (semantic → coarse → fine) +
/// EnCodec 24 kHz 8-codebook decode. All three stages reuse the shared <see cref="GptBackbone"/>; this
/// config carries their per-stage vocab sizes + the Bark token-offset constants. <c>Full</c> = hidden
/// 1024 / 24 layers / 16 heads; <c>Small</c> = 768 / 12 / 12.</summary>
public sealed record BarkConfig
{
    /// <summary>Semantic transformer: BERT-text (offset) → 10k HuBERT-style semantic tokens, causal.</summary>
    public required GptConfig Stage { get; init; }

    public int SemanticInputVocab { get; init; } = 129_600;
    public int SemanticOutputVocab { get; init; } = 10_048;
    public int CoarseVocab { get; init; } = 12_096;     // in == out (interleaved 2 codebooks)
    public int FineVocab { get; init; } = 1_056;        // per-codebook
    public int NumCodebooks { get; init; } = 8;
    public int NumCoarseCodebooks { get; init; } = 2;
    public int CodebookSize { get; init; } = 1_024;     // EnCodec acoustic codebook entries

    // ── Bark token-offset constants (from the upstream `bark/generation.py`) ──
    public int TextEncodingOffset { get; init; } = 10_048;
    public int SemanticPadToken { get; init; } = 10_000;     // also semantic EOS
    public int SemanticInferToken { get; init; } = 129_599;
    public int TextPadToken { get; init; } = 129_595;
    public int CoarseSemanticPadToken { get; init; } = 12_048;
    public int CoarseInferToken { get; init; } = 12_050;     // upstream bark/generation.py COARSE_INFER_TOKEN

    public int SampleRate { get; init; } = 24_000;

    // ── Sampling defaults ──
    public float SemanticTemperature { get; init; } = 0.7f;
    public float CoarseTemperature { get; init; } = 0.7f;
    public float FineTemperature { get; init; } = 0.5f;
    public int TopK { get; init; } = 0;            // Bark uses top-p only by default
    public float TopP { get; init; } = 1.0f;

    public static BarkConfig Full => new() { Stage = GptConfig.BarkFull };
    public static BarkConfig Small => new() { Stage = GptConfig.BarkSmall };
}

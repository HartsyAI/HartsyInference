namespace HartsyInference.Audio.Models.GptSoVits;

/// <summary>Configuration for GPT-SoVITS stage 1 — the T2S (text→semantic) AR GPT (`Text2SemanticDecoder`).
/// A custom <b>post-LayerNorm</b> transformer (NOT Llama: no RoPE/RMSNorm/SwiGLU) with sinusoidal positions,
/// ReLU MLP, biased fused-QKV attention; predicts single-codebook semantic tokens. See
/// <c>docs/Research/GPT_SOVITS_ARCHITECTURE.md</c>.</summary>
public sealed record Text2SemanticConfig
{
    public int Hidden { get; init; } = 512;
    public int NumLayers { get; init; } = 24;
    public int NumHeads { get; init; } = 16;
    public int HeadDim => Hidden / NumHeads;     // 32
    public int FfnDim { get; init; } = 2_048;    // hidden * 4

    // s1's AR phoneme vocab is 512 (distinct from the s2 SoVITS text_embedding's 732 — GPT-SoVITS uses
    // different phoneme symbol sets for the two stages). Confirmed vs s1bert25hz ar_text_embedding [512, 512].
    public int PhonemeVocab { get; init; } = 512;
    public int SemanticVocab { get; init; } = 1_025;  // 1024 codes + EOS
    public int EosToken { get; init; } = 1_024;
    public int BertDim { get; init; } = 1_024;        // chinese-roberta-wwm-ext-large
    public float NormEps { get; init; } = 1e-5f;

    // ── Sampling (yaml inference defaults) ──
    public int TopK { get; init; } = 15;
    public float Temperature { get; init; } = 1.0f;
    public float RepetitionPenalty { get; init; } = 1.35f;
    public int MaxNewTokens { get; init; } = 1_500;

    public static Text2SemanticConfig V2 => new();
}

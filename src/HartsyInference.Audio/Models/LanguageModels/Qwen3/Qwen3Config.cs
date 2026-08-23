namespace HartsyInference.Audio.Models.LanguageModels.Qwen3;

/// <summary>Configuration for a Qwen3 decoder; vs Qwen2 it adds <b>per-head q_norm/k_norm</b> (RMSNorm over head_dim, applied post-projection before RoPE), <b>decouples head_dim from hidden_size</b> (q/k/v projections are sized <c>n_heads·head_dim</c>, not hidden), and drops QKV bias — otherwise Llama-family (RMSNorm, SwiGLU, GQA, RoPE). See <c>docs/Research/QWEN3_TTS_ARCHITECTURE.md</c>.</summary>
public sealed record Qwen3Config
{
    public required int HiddenSize { get; init; }
    public required int NumHiddenLayers { get; init; }
    public required int NumAttentionHeads { get; init; }
    public required int NumKeyValueHeads { get; init; }
    public required int HeadDim { get; init; }           // explicit — NOT hidden/heads
    public required int IntermediateSize { get; init; }
    public int MaxPositionEmbeddings { get; init; } = 32_768;
    public float RopeTheta { get; init; } = 1_000_000f;
    public float RmsNormEps { get; init; } = 1e-6f;

    public int QDim => NumAttentionHeads * HeadDim;
    public int KvDim => NumKeyValueHeads * HeadDim;
    public int KvGroup => NumAttentionHeads / NumKeyValueHeads;

    /// <summary>Qwen3-TTS talker backbone (1.7B): hidden 2048 / 28L / 16 heads / 8 KV / head_dim 128 / SwiGLU 6144 / RoPE θ=1e6.</summary>
    public static Qwen3Config Talker1_7B => new()
    {
        HiddenSize = 2_048, NumHiddenLayers = 28, NumAttentionHeads = 16, NumKeyValueHeads = 8,
        HeadDim = 128, IntermediateSize = 6_144,
    };

    /// <summary>Qwen3-TTS talker backbone (0.6B): hidden 1024 / 28L / 16 heads / 8 KV / head_dim 128 / SwiGLU 3072 / RoPE θ=1e6 — verified against Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice config.json.</summary>
    public static Qwen3Config Talker0_6B => new()
    {
        HiddenSize = 1_024, NumHiddenLayers = 28, NumAttentionHeads = 16, NumKeyValueHeads = 8,
        HeadDim = 128, IntermediateSize = 3_072,
    };

    /// <summary>Qwen3-TTS code-predictor (MTP) — 5-layer depth transformer (plain 1D RoPE).</summary>
    public static Qwen3Config CodePredictor => new()
    {
        HiddenSize = 1_024, NumHiddenLayers = 5, NumAttentionHeads = 16, NumKeyValueHeads = 8,
        HeadDim = 128, IntermediateSize = 3_072,
    };
}

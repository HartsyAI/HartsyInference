namespace SharpInference.Audio.Models.LanguageModels.Qwen2;

/// <summary>Configuration for a Qwen2 / Qwen2.5 autoregressive language model. Mirrors the
/// HuggingFace <c>transformers.models.qwen2.configuration_qwen2.Qwen2Config</c> fields we
/// actually use at inference. Qwen2's defining structural features vs Llama:
/// <list type="bullet">
///   <item><b>Q/K/V projections carry biases</b> (Llama / Mistral / Qwen3 do not).</item>
///   <item><b>O projection has no bias</b>.</item>
///   <item><b>No per-head Q/K RMSNorm</b> (Qwen3 added this, Qwen2 does not).</item>
///   <item><b>Tied <c>lm_head</c></b> to <c>embed_tokens</c> for the small variants (Qwen2.5-0.5B,
///         1.5B); independent <c>lm_head</c> for the 7B.</item>
///   <item><b>SwiGLU MLP</b> with <c>down(silu(gate(x)) * up(x))</c>.</item>
///   <item><b>Long-context RoPE</b> with <c>theta = 1e6</c>.</item>
/// </list>
///
/// <para>VibeVoice wraps a Qwen2.5 LM and reuses several existing token IDs
/// (<c>&lt;|vision_start|&gt;</c> etc.) as TTS control tokens — no vocab extension.</para></summary>
public sealed record Qwen2Config
{
    /// <summary>Hidden dimension. 1536 / 3584 / 896 for VibeVoice 1.5B / 7B / 0.5B.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of decoder layers. 28 / 28 / 24.</summary>
    public required int NumHiddenLayers { get; init; }

    /// <summary>Number of attention query heads. 12 / 28 / 14.</summary>
    public required int NumAttentionHeads { get; init; }

    /// <summary>Number of attention key/value heads for GQA. 2 / 4 / 2 — the K/V cache
    /// shape uses this, not <see cref="NumAttentionHeads"/>.</summary>
    public required int NumKeyValueHeads { get; init; }

    /// <summary>SwiGLU FFN inner dimension. 8 960 / 18 944 / 4 864.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Vocabulary size. 151 936 (Qwen2.5-1.5B / 0.5B) or 152 064 (7B). VibeVoice
    /// does not extend the vocab.</summary>
    public required int VocabSize { get; init; }

    /// <summary>Maximum positions the RoPE table can address. 65 536 for the 1.5B
    /// long-context preset; 32 768 for 7B and 0.5B. Acts as a hard cap on prompt + AR
    /// generation length.</summary>
    public required int MaxPositionEmbeddings { get; init; }

    /// <summary>RoPE base frequency. 1e6 across Qwen2.5 (long-context preset).</summary>
    public float RopeTheta { get; init; } = 1_000_000f;

    /// <summary>RMSNorm epsilon. 1e-6 across Qwen2.5.</summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>Whether Q/K/V projections carry biases. <c>true</c> for Qwen2/Qwen2.5 (its defining
    /// feature). Set <c>false</c> to reuse this exact transformer for <b>Llama-3.2</b>-family backbones
    /// (e.g. Sesame CSM), which have no attention bias. The O projection never has a bias either way.</summary>
    public bool AttentionBias { get; init; } = true;

    /// <summary>Whether <c>lm_head.weight</c> is tied to <c>embed_tokens.weight</c>.
    /// True for 1.5B / 0.5B, false for 7B.</summary>
    public bool TieWordEmbeddings { get; init; } = true;

    /// <summary>Per-head dimension. Inferred from <see cref="HiddenSize"/> /
    /// <see cref="NumAttentionHeads"/>. 128 across all VibeVoice variants.</summary>
    public int HeadDim => HiddenSize / NumAttentionHeads;

    /// <summary>Total K and V projection output dim — <see cref="NumKeyValueHeads"/> ×
    /// <see cref="HeadDim"/>. Smaller than <see cref="HiddenSize"/> due to GQA.</summary>
    public int KvHiddenSize => NumKeyValueHeads * HeadDim;

    /// <summary>Number of query heads per KV group — used to repeat the K/V tensors before
    /// SDPA so the multi-head attention sees the right shape.</summary>
    public int KvGroupSize => NumAttentionHeads / NumKeyValueHeads;

    /// <summary>BOS / EOS / pad token IDs (informational — the AR loop owns sampling).</summary>
    public int BosTokenId { get; init; } = 151_643;
    public int EosTokenId { get; init; } = 151_645;
    public int PadTokenId { get; init; } = 151_643;

    /// <summary>Qwen2.5-1.5B preset (matches the LM half of VibeVoice-1.5B).</summary>
    public static Qwen2Config Qwen25_1_5B => new()
    {
        HiddenSize = 1_536,
        NumHiddenLayers = 28,
        NumAttentionHeads = 12,
        NumKeyValueHeads = 2,
        IntermediateSize = 8_960,
        VocabSize = 151_936,
        MaxPositionEmbeddings = 65_536,
        TieWordEmbeddings = true,
    };

    /// <summary>Qwen2.5-7B preset (VibeVoice "Large").</summary>
    public static Qwen2Config Qwen25_7B => new()
    {
        HiddenSize = 3_584,
        NumHiddenLayers = 28,
        NumAttentionHeads = 28,
        NumKeyValueHeads = 4,
        IntermediateSize = 18_944,
        VocabSize = 152_064,
        MaxPositionEmbeddings = 32_768,
        TieWordEmbeddings = false,
    };

    /// <summary>Qwen2.5-0.5B preset (VibeVoice-Streaming-0.5B backbone).</summary>
    public static Qwen2Config Qwen25_0_5B => new()
    {
        HiddenSize = 896,
        NumHiddenLayers = 24,
        NumAttentionHeads = 14,
        NumKeyValueHeads = 2,
        IntermediateSize = 4_864,
        VocabSize = 151_936,
        MaxPositionEmbeddings = 32_768,
        TieWordEmbeddings = true,
    };
}

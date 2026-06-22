namespace HartsyInference.LLM.Transformer;

/// <summary>RoPE pairing convention. <see cref="SplitHalf"/> (Llama/Qwen2/HF-Qwen3-text rotate_half: pairs
/// dim <c>i</c> with <c>i+half</c>) vs <see cref="Interleaved"/> (GPT-J: pairs adjacent dims <c>2i, 2i+1</c>,
/// used by the Qwen3-TTS audio backbone and Moonshine). Not interchangeable.</summary>
public enum RopeStyle
{
    /// <summary>Llama / Qwen2 / HF Qwen3-text rotate_half (dim i paired with i+half).</summary>
    SplitHalf,

    /// <summary>GPT-J interleaved (adjacent dims 2i, 2i+1).</summary>
    Interleaved,
}

/// <summary>Architecture description for the config-driven <see cref="GenericTransformer"/> — one record that
/// covers the dense decoder-LLM family (Qwen2, Qwen3, and Llama-lineage models) so a new model is a preset
/// plus a checkpoint key mapping, not a new transformer class.
///
/// <para>The variation surface across these families is small: per-head Q/K RMSNorm (Qwen3), QKV projection
/// bias (Qwen2), and a head dimension decoupled from <c>hidden / heads</c> (Qwen3). Everything else (pre-norm
/// RMSNorm, GQA, split-half RoPE, SwiGLU MLP, tied or separate <c>lm_head</c>) is shared. M1 scope is
/// pre-norm + SwiGLU + causal attention; sandwich-norm / alternate activations / encoder mode are later.</para></summary>
public sealed record TransformerConfig
{
    /// <summary>Residual-stream width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of decoder layers.</summary>
    public required int NumLayers { get; init; }

    /// <summary>Number of attention query heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Number of key/value heads (GQA). Equal to <see cref="NumHeads"/> for MHA, 1 for MQA.</summary>
    public required int NumKvHeads { get; init; }

    /// <summary>Per-head dimension. Decoupled from <see cref="HiddenSize"/> / <see cref="NumHeads"/> — Qwen3
    /// projects Q to <c>NumHeads · HeadDim</c> which need not equal <see cref="HiddenSize"/>.</summary>
    public required int HeadDim { get; init; }

    /// <summary>SwiGLU feed-forward inner dimension.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Vocabulary size (embedding rows and logit width).</summary>
    public required int VocabSize { get; init; }

    /// <summary>Maximum position the RoPE table addresses; a hard cap on prompt + generation length.</summary>
    public int MaxPositionEmbeddings { get; init; } = 32_768;

    /// <summary>RoPE base frequency.</summary>
    public float RopeTheta { get; init; } = 1_000_000f;

    /// <summary>RMSNorm epsilon.</summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>Whether the Q/K/V projections carry a bias. <c>true</c> for Qwen2/Qwen2.5, <c>false</c> for
    /// Qwen3 / Llama / Mistral. The O projection never has a bias.</summary>
    public bool AttentionBias { get; init; }

    /// <summary>Whether per-head Q/K RMSNorm is applied (over <see cref="HeadDim"/>, after projection and
    /// reshape into heads, before RoPE). <c>true</c> for Qwen3, <c>false</c> for Qwen2/Llama.</summary>
    public bool QkNorm { get; init; }

    /// <summary>Whether <c>lm_head</c> shares the embedding table (no separate <c>lm_head.weight</c>).</summary>
    public bool TieWordEmbeddings { get; init; } = true;

    /// <summary>RoPE pairing convention. Default <see cref="RopeStyle.SplitHalf"/> (Qwen2 / Llama / HF Qwen3
    /// text). The Qwen3-TTS audio backbone uses <see cref="RopeStyle.Interleaved"/>.</summary>
    public RopeStyle Rope { get; init; } = RopeStyle.SplitHalf;

    /// <summary>Total Q projection output dim — <see cref="NumHeads"/> × <see cref="HeadDim"/>.</summary>
    public int QDim => NumHeads * HeadDim;

    /// <summary>Total K/V projection output dim — <see cref="NumKvHeads"/> × <see cref="HeadDim"/>.</summary>
    public int KvDim => NumKvHeads * HeadDim;

    /// <summary>Query heads per KV group (GQA repeat factor).</summary>
    public int KvGroup => NumHeads / NumKvHeads;

    // ── Presets ──────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Qwen2.5-0.5B-Instruct (head_dim coupled = 896/14 = 64, QKV bias, no Q/K norm, tied).</summary>
    public static TransformerConfig Qwen2_5_0_5B => new()
    {
        HiddenSize = 896, NumLayers = 24, NumHeads = 14, NumKvHeads = 2, HeadDim = 64,
        IntermediateSize = 4_864, VocabSize = 151_936, MaxPositionEmbeddings = 32_768,
        AttentionBias = true, QkNorm = false, TieWordEmbeddings = true,
    };

    /// <summary>Qwen2.5-1.5B (matches the LM backbone inside VibeVoice-1.5B).</summary>
    public static TransformerConfig Qwen2_5_1_5B => new()
    {
        HiddenSize = 1_536, NumLayers = 28, NumHeads = 12, NumKvHeads = 2, HeadDim = 128,
        IntermediateSize = 8_960, VocabSize = 151_936, MaxPositionEmbeddings = 65_536,
        AttentionBias = true, QkNorm = false, TieWordEmbeddings = true,
    };

    /// <summary>Qwen3-0.6B (decoupled head_dim 16·128=2048 ≠ hidden 1024, no QKV bias, per-head Q/K norm,
    /// tied). Exercises all three Qwen3 variation axes.</summary>
    public static TransformerConfig Qwen3_0_6B => new()
    {
        HiddenSize = 1_024, NumLayers = 28, NumHeads = 16, NumKvHeads = 8, HeadDim = 128,
        IntermediateSize = 3_072, VocabSize = 151_936, MaxPositionEmbeddings = 40_960,
        AttentionBias = false, QkNorm = true, TieWordEmbeddings = true,
    };
}

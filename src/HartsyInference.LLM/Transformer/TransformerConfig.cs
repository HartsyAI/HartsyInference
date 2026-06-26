using HartsyInference.Core.Rope;

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

/// <summary>Gated-FFN activation. <see cref="Silu"/> (Llama/Qwen/Mistral SwiGLU) vs <see cref="GeluTanh"/>
/// (Gemma's GeGLU — the tanh-approximation GELU, <c>gelu_pytorch_tanh</c>).</summary>
public enum ActivationKind
{
    /// <summary>SiLU / swish gate (SwiGLU). Llama, Qwen, Mistral.</summary>
    Silu,

    /// <summary>tanh-approximation GELU gate (GeGLU). Gemma 2 / Gemma 3.</summary>
    GeluTanh,
}

/// <summary>How the MoE router turns expert logits into selection scores. <see cref="Softmax"/> (Qwen2-MoE /
/// Qwen3-MoE / Mixtral: softmax over all experts, then top-k) vs <see cref="Sigmoid"/> (DeepSeek-V3:
/// independent sigmoid per expert).</summary>
public enum MoeScoring
{
    /// <summary>Softmax over all expert logits, then top-k (Qwen/Mixtral).</summary>
    Softmax,

    /// <summary>Independent sigmoid per expert, then top-k (DeepSeek-V3).</summary>
    Sigmoid,
}

/// <summary>Mixture-of-Experts feed-forward configuration. The dense SwiGLU FFN is replaced (on MoE layers) by a
/// router that selects <see cref="NumExpertsPerTok"/> of <see cref="NumExperts"/> experts per token plus optional
/// always-on shared expert(s). Mirrors the HF config fields (<c>num_experts</c>, <c>num_experts_per_tok</c>,
/// <c>norm_topk_prob</c>, <c>moe_intermediate_size</c>, <c>shared_expert_intermediate_size</c>).</summary>
public sealed record MoeConfig
{
    /// <summary>Number of routed experts.</summary>
    public required int NumExperts { get; init; }

    /// <summary>Experts selected per token (top-k).</summary>
    public required int NumExpertsPerTok { get; init; }

    /// <summary>Per-expert SwiGLU inner dimension (often smaller than the dense <see cref="TransformerConfig.IntermediateSize"/>).</summary>
    public required int MoeIntermediateSize { get; init; }

    /// <summary>Whether the top-k routing weights are renormalized to sum to 1 (Qwen3-MoE / Mixtral: true).</summary>
    public bool NormTopKProb { get; init; } = true;

    /// <summary>Router scoring function.</summary>
    public MoeScoring Scoring { get; init; } = MoeScoring.Softmax;

    /// <summary>Always-on shared expert FFN inner dimension (Qwen2-MoE / DeepSeek). 0 disables the shared expert.</summary>
    public int SharedExpertIntermediateSize { get; init; }

    /// <summary>Layers <c>[0, FirstDenseLayers)</c> stay dense; the rest are MoE (DeepSeek <c>first_k_dense_replace</c>).
    /// 0 (default) makes every layer MoE.</summary>
    public int FirstDenseLayers { get; init; }
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

    /// <summary>RoPE inverse-frequency scaling (long-context extrapolation). Default <see cref="RopeScaling.None"/>
    /// (standard RoPE for Qwen/Mistral). Llama-3.x, yarn, and Phi-longrope models set this; required for them to
    /// produce coherent output. Applied host-side in the cos/sin table via <see cref="RopeFrequencyBuilder"/>.</summary>
    public RopeScaling RopeScaling { get; init; } = RopeScaling.None;

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

    /// <summary>Gated-FFN activation. Default SiLU (SwiGLU); Gemma uses <see cref="ActivationKind.GeluTanh"/>.</summary>
    public ActivationKind Activation { get; init; } = ActivationKind.Silu;

    /// <summary>Gemma "sandwich norm": an extra RMSNorm on the attention output and on the FFN output, each
    /// applied <i>before</i> the residual add (in addition to the pre-attn / pre-FFN norms). false for the
    /// standard pre-norm layout (Qwen/Llama).</summary>
    public bool SandwichNorm { get; init; }

    /// <summary>Gemma's <c>(1 + weight)</c> RMSNorm: the norm weight is offset by 1 (the GGUF/HF weight stores
    /// the centered value). Baked into the loaded norm weights at load time, so the runtime norm op is unchanged.</summary>
    public bool RmsNormAddOne { get; init; }

    /// <summary>Multiplier applied to token embeddings after lookup. 1 for Qwen/Llama; Gemma uses
    /// <c>sqrt(HiddenSize)</c> (the embedding normalizer).</summary>
    public float EmbeddingScale { get; init; } = 1f;

    /// <summary>Attention score scale numerator: when &gt; 0 the scale is <c>1/sqrt(QueryPreAttnScalar)</c>
    /// (Gemma's <c>query_pre_attn_scalar</c>, which can differ from head_dim, e.g. Gemma-2-27B); 0 (default)
    /// uses the standard <c>1/sqrt(HeadDim)</c>.</summary>
    public float QueryPreAttnScalar { get; init; }

    /// <summary>Gemma-3 dual-RoPE: local (sliding-window) layers use this base frequency while global layers use
    /// <see cref="RopeTheta"/>. 0 (default) means a single RoPE base for all layers.</summary>
    public float RopeLocalTheta { get; init; }

    /// <summary>Sliding-window layer pattern: layer <c>i</c> is a global (full-attention) layer when this is
    /// 0 (all global) or <c>(i+1) % pattern == 0</c>; otherwise it is a local (sliding-window) layer. Gemma-2
    /// alternates (pattern 2); Gemma-3 is 5 local : 1 global (pattern 6).</summary>
    public int SlidingWindowPattern { get; init; }

    /// <summary>Sliding-window span for local layers (Gemma-2 4096, Gemma-3 512/1024). Only affects sequences
    /// longer than the window; shorter contexts are unaffected (every position is in-window).</summary>
    public int SlidingWindow { get; init; }

    /// <summary>Attention-logit soft-cap (Gemma-2 = 50): scores are capped via <c>cap·tanh(score/cap)</c> before
    /// the softmax. 0 (default) disables it. Requires the soft-capping attention path.</summary>
    public float AttnLogitSoftcap { get; init; }

    /// <summary>Final-logit soft-cap (Gemma-2 = 30): output logits are capped via <c>cap·tanh(logit/cap)</c>.
    /// 0 (default) disables it.</summary>
    public float FinalLogitSoftcap { get; init; }

    /// <summary>Whether layer <paramref name="i"/> uses global (full) attention vs a local sliding window.</summary>
    public bool IsGlobalLayer(int i) => SlidingWindowPattern <= 0 || (i + 1) % SlidingWindowPattern == 0;

    /// <summary>RoPE base frequency for layer <paramref name="i"/> (Gemma-3 dual-RoPE: local layers use
    /// <see cref="RopeLocalTheta"/>, global use <see cref="RopeTheta"/>).</summary>
    public float RopeThetaFor(int i) => RopeLocalTheta > 0 && !IsGlobalLayer(i) ? RopeLocalTheta : RopeTheta;

    /// <summary>Attention score scale: <c>1/sqrt(QueryPreAttnScalar)</c> when set, else <c>1/sqrt(HeadDim)</c>.</summary>
    public float AttnScale => 1f / MathF.Sqrt(QueryPreAttnScalar > 0 ? QueryPreAttnScalar : HeadDim);

    /// <summary>For quantized (GGUF) weights: when <c>false</c> (default) the dequantized F16 weight is cached
    /// per weight (fast decode, but the weight set occupies F16-sized VRAM); when <c>true</c> the low-VRAM
    /// <see cref="IBackend.QuantizedMatMul"/> path is used (weights stay compressed on-device, dequant is
    /// transient per call), trading decode speed for a much smaller footprint so large models fit. No effect on
    /// float weights.</summary>
    public bool LowVramQuant { get; init; }

    /// <summary>Total Q projection output dim — <see cref="NumHeads"/> × <see cref="HeadDim"/>.</summary>
    public int QDim => NumHeads * HeadDim;

    /// <summary>Total K/V projection output dim — <see cref="NumKvHeads"/> × <see cref="HeadDim"/>.</summary>
    public int KvDim => NumKvHeads * HeadDim;

    /// <summary>Query heads per KV group (GQA repeat factor).</summary>
    public int KvGroup => NumHeads / NumKvHeads;

    /// <summary>Mixture-of-Experts FFN config, or <c>null</c> for a dense SwiGLU FFN (Qwen2/Qwen3/Llama dense).</summary>
    public MoeConfig? Moe { get; init; }

    /// <summary>Whether layer <paramref name="layerIndex"/> uses the MoE FFN (vs dense SwiGLU).</summary>
    public bool IsMoeLayer(int layerIndex) => Moe is not null && layerIndex >= Moe.FirstDenseLayers;

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

    /// <summary>Gemma-3-1B-it: decoupled head_dim 256 (4·256=1024 ≠ hidden 1152), MQA (1 KV head), Q/K norm,
    /// GeGLU, sandwich + (1+w) norms, embedding scale √1152, dual-RoPE (local 10k / global 1M, 5:1 pattern),
    /// no logit soft-cap. Exercises the full Gemma-3 surface.</summary>
    public static TransformerConfig Gemma3_1B => new()
    {
        HiddenSize = 1_152, NumLayers = 26, NumHeads = 4, NumKvHeads = 1, HeadDim = 256,
        IntermediateSize = 6_912, VocabSize = 262_144, MaxPositionEmbeddings = 32_768,
        RopeTheta = 1_000_000f, RmsNormEps = 1e-6f, AttentionBias = false, QkNorm = true, TieWordEmbeddings = true,
        Activation = ActivationKind.GeluTanh, SandwichNorm = true, RmsNormAddOne = true,
        EmbeddingScale = 33.941124f /* sqrt(1152) */, QueryPreAttnScalar = 256f,
        RopeLocalTheta = 10_000f, SlidingWindowPattern = 6, SlidingWindow = 512,
    };

    /// <summary>Gemma-2-2B-it: head_dim 256, GQA (8/4), GeGLU, sandwich + (1+w) norms, embedding scale √2304,
    /// attn/final logit soft-cap (50/30), alternating sliding window (pattern 2, window 4096), single RoPE 10k.</summary>
    public static TransformerConfig Gemma2_2B => new()
    {
        HiddenSize = 2_304, NumLayers = 26, NumHeads = 8, NumKvHeads = 4, HeadDim = 256,
        IntermediateSize = 9_216, VocabSize = 256_000, MaxPositionEmbeddings = 8_192,
        RopeTheta = 10_000f, RmsNormEps = 1e-6f, AttentionBias = false, QkNorm = false, TieWordEmbeddings = true,
        Activation = ActivationKind.GeluTanh, SandwichNorm = true, RmsNormAddOne = true,
        EmbeddingScale = 48f /* sqrt(2304) */, QueryPreAttnScalar = 256f,
        SlidingWindowPattern = 2, SlidingWindow = 4_096,
        AttnLogitSoftcap = 50f, FinalLogitSoftcap = 30f,
    };
}

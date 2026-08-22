using HartsyInference.Core.Rope;

namespace HartsyInference.LLM.Transformer;

/// <summary>RoPE pairing convention: <see cref="SplitHalf"/> (Llama/Qwen2/HF-Qwen3-text rotate_half: pairs dim <c>i</c> with <c>i+half</c>) vs <see cref="Interleaved"/> (GPT-J: pairs adjacent dims <c>2i, 2i+1</c>, used by the Qwen3-TTS audio backbone and Moonshine); not interchangeable.</summary>
public enum RopeStyle
{
    /// <summary>Llama / Qwen2 / HF Qwen3-text rotate_half (dim i paired with i+half).</summary>
    SplitHalf,

    /// <summary>GPT-J interleaved (adjacent dims 2i, 2i+1).</summary>
    Interleaved,
}

/// <summary>FFN activation. For a gated FFN (<see cref="TransformerConfig.GatedFfn"/> = true) this is the gate activation: <see cref="Silu"/> (Llama/Qwen/Mistral SwiGLU) vs <see cref="GeluTanh"/> (Gemma's GeGLU, <c>gelu_pytorch_tanh</c>); for a non-gated FFN (GPT-2/Falcon/BLOOM/MPT/Nemotron) it applies to the single up-projection: <see cref="GeluTanh"/>, <see cref="Relu"/>, or <see cref="ReluSquared"/> (Nemotron's <c>relu²</c>).</summary>
public enum ActivationKind
{
    /// <summary>SiLU / swish gate (SwiGLU). Llama, Qwen, Mistral.</summary>
    Silu,

    /// <summary>tanh-approximation GELU (GeGLU gate, or the non-gated FFN act of GPT-2/Falcon/BLOOM).</summary>
    GeluTanh,

    /// <summary>ReLU (non-gated FFN). Some StarCoder / older arches.</summary>
    Relu,

    /// <summary>ReLU² (squared ReLU, non-gated FFN). Nemotron.</summary>
    ReluSquared,
}

/// <summary>Where the per-sublayer norm sits relative to the residual: <see cref="PreNorm"/> (Llama/Qwen/Gemma: norm the input to attn/FFN) vs <see cref="PostNorm"/> (OLMo-2: attn/FFN read the <i>raw</i> residual and their output is normed before the add — <c>x = x + norm(attn(x))</c>, <c>x = x + norm(mlp(x))</c>; no input/pre-FFN norm).</summary>
public enum NormPlacement
{
    /// <summary>Pre-norm: normalize the input to each sublayer (Llama/Qwen/Gemma/most).</summary>
    PreNorm,

    /// <summary>Post-norm: sublayers read the raw residual; their output is normed before the residual add (OLMo-2).</summary>
    PostNorm,
}

/// <summary>How the MoE router turns expert logits into selection scores: <see cref="Softmax"/> (Qwen2-MoE/Qwen3-MoE/Mixtral: softmax over all experts, then top-k) vs <see cref="Sigmoid"/> (DeepSeek-V3: independent sigmoid per expert).</summary>
public enum MoeScoring
{
    /// <summary>Softmax over all expert logits, then top-k (Qwen/Mixtral).</summary>
    Softmax,

    /// <summary>Independent sigmoid per expert, then top-k (DeepSeek-V3).</summary>
    Sigmoid,
}

/// <summary>Mixture-of-Experts feed-forward configuration: the dense SwiGLU FFN is replaced (on MoE layers) by a router that selects <see cref="NumExpertsPerTok"/> of <see cref="NumExperts"/> experts per token plus optional always-on shared expert(s), mirroring the HF config fields.</summary>
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

    /// <summary>Expert/shared-expert FFN activation; default SiLU (SwiGLU: Qwen-MoE/Mixtral/DeepSeek), Gemma-4's MoE layers use <see cref="ActivationKind.GeluTanh"/> (GeGLU), matching its dense FFN.</summary>
    public ActivationKind Activation { get; init; } = ActivationKind.Silu;

    /// <summary>Layers <c>[0, FirstDenseLayers)</c> stay dense; the rest are MoE (DeepSeek <c>first_k_dense_replace</c>); 0 (default) makes every layer MoE.</summary>
    public int FirstDenseLayers { get; init; }

    /// <summary>DeepSeek-V3/Kimi-K2 node-limited (group-limited) routing: experts are partitioned into <see cref="ExpertGroupCount"/> groups; only the top <see cref="ExpertGroupUsedCount"/> groups (scored by the sum of their top-2 experts) are eligible for the per-token top-k. 0 (default) disables grouping. When grouped, the router adds a per-expert correction bias to the selection scores only, and the routing weight stays the raw sigmoid score scaled by <see cref="RoutedScalingFactor"/>.</summary>
    public int ExpertGroupCount { get; init; }

    /// <summary>Number of expert groups kept per token under group-limited routing (DeepSeek-V3 <c>topk_group</c>).</summary>
    public int ExpertGroupUsedCount { get; init; }

    /// <summary>Multiplier applied to the routed-expert weights (DeepSeek-V3 <c>routed_scaling_factor</c>, default 1).</summary>
    public float RoutedScalingFactor { get; init; } = 1f;
}

/// <summary>Multi-head Latent Attention (DeepSeek-V2/V3, Kimi-K2): keys/values are compressed to a shared low-rank latent (<see cref="KvLoraRank"/>) plus a small shared RoPE key (<see cref="QkRopeHeadDim"/>), then up-projected per head; queries are either projected directly (<see cref="QLoraRank"/> = 0, DeepSeek-V2-Lite) or themselves compressed/up-projected. Each head's Q/K split into a no-position part (<see cref="QkNopeHeadDim"/>) and a RoPE part; attention scores use the full <see cref="QkHeadDim"/> but value/output use <see cref="VHeadDim"/>.</summary>
public sealed record MlaConfig
{
    /// <summary>Compressed KV latent rank (DeepSeek-V2-Lite: 512).</summary>
    public required int KvLoraRank { get; init; }

    /// <summary>Compressed query rank, or 0 for a direct query projection (DeepSeek-V2-Lite = 0).</summary>
    public int QLoraRank { get; init; }

    /// <summary>Per-head no-position (non-RoPE) Q/K dimension (DeepSeek-V2-Lite: 128).</summary>
    public required int QkNopeHeadDim { get; init; }

    /// <summary>Per-head RoPE Q/K dimension — the decoupled rotary part (DeepSeek-V2-Lite: 64).</summary>
    public required int QkRopeHeadDim { get; init; }

    /// <summary>Per-head value/output dimension (DeepSeek-V2-Lite: 128).</summary>
    public required int VHeadDim { get; init; }

    /// <summary>Full per-head Q/K dimension used for the attention scores: nope + rope.</summary>
    public int QkHeadDim => QkNopeHeadDim + QkRopeHeadDim;

    /// <summary>Pre-computed softmax score scale (DeepSeek YaRN: <c>mscale²/√QkHeadDim</c> where <c>mscale = 1 + yarn_log_multiplier·ln(factor)</c>); 0 (default) falls back to the plain <c>1/√QkHeadDim</c>. llama.cpp folds the YaRN <c>mscale²</c> into the attention score scale (not the cos/sin), so for DeepSeek long-context the cos/sin mscale is left neutral and this carries it.</summary>
    public float AttnScale { get; init; }
}

/// <summary>Architecture description for the config-driven <see cref="GenericTransformer"/> — one record that covers the dense decoder-LLM family (Qwen2, Qwen3, and Llama-lineage models) so a new model is a preset plus a checkpoint key mapping, not a new transformer class.</summary>
/// <remarks>The variation surface across these families is small: per-head Q/K RMSNorm (Qwen3), QKV projection bias (Qwen2), and a head dimension decoupled from <c>hidden / heads</c> (Qwen3). Everything else (pre-norm RMSNorm, GQA, split-half RoPE, SwiGLU MLP, tied or separate <c>lm_head</c>) is shared. M1 scope is pre-norm + SwiGLU + causal attention; sandwich-norm / alternate activations / encoder mode are later.</remarks>
public sealed record TransformerConfig
{
    /// <summary>Residual-stream width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of decoder layers actually run for next-token decoding; for models with Multi-Token Prediction (DeepSeek-V3 / GLM-4-MoE / bailingmoe2) this already excludes the trailing <c>nextn_predict_layers</c> MTP layer(s) — those are speculative-draft heads, not part of the main path.</summary>
    public required int NumLayers { get; init; }

    /// <summary>Number of attention query heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Number of key/value heads (GQA). Equal to <see cref="NumHeads"/> for MHA, 1 for MQA.</summary>
    public required int NumKvHeads { get; init; }

    /// <summary>Per-head dimension, decoupled from <see cref="HiddenSize"/> / <see cref="NumHeads"/> — Qwen3 projects Q to <c>NumHeads · HeadDim</c> which need not equal <see cref="HiddenSize"/>.</summary>
    public required int HeadDim { get; init; }

    /// <summary>RoPE rotary dimension: how many leading dims of each head are rotated (partial rotary); 0 (default) = full (<see cref="HeadDim"/>). Phi-4-mini (96 of 128) and StableLM-2 use partial rotary; the remaining dims pass through unrotated.</summary>
    public int RotaryDim { get; init; }

    /// <summary>Effective rotary dimension (<see cref="RotaryDim"/> if set, else the full <see cref="HeadDim"/>).</summary>
    public int EffRotaryDim => RotaryDim > 0 && RotaryDim < HeadDim ? RotaryDim : HeadDim;

    /// <summary>SwiGLU feed-forward inner dimension.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Vocabulary size (embedding rows and logit width).</summary>
    public required int VocabSize { get; init; }

    /// <summary>Maximum position the RoPE table addresses; a hard cap on prompt + generation length.</summary>
    public int MaxPositionEmbeddings { get; init; } = 32_768;

    /// <summary>RoPE base frequency.</summary>
    public float RopeTheta { get; init; } = 1_000_000f;

    /// <summary>RoPE inverse-frequency scaling (long-context extrapolation); default <see cref="RopeScaling.None"/> (standard RoPE for Qwen/Mistral). Llama-3.x, yarn, and Phi-longrope models set this; required for them to produce coherent output. Applied host-side in the cos/sin table via <see cref="RopeFrequencyBuilder"/>.</summary>
    public RopeScaling RopeScaling { get; init; } = RopeScaling.None;

    /// <summary>RMSNorm epsilon.</summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>Whether the Q/K/V projections carry a bias; <c>true</c> for Qwen2/Qwen2.5, <c>false</c> for Qwen3/Llama/Mistral. The O projection never has a bias.</summary>
    public bool AttentionBias { get; init; }

    /// <summary>Whether per-head Q/K RMSNorm is applied (over <see cref="HeadDim"/>, after projection and reshape into heads, before RoPE); <c>true</c> for Qwen3, <c>false</c> for Qwen2/Llama.</summary>
    public bool QkNorm { get; init; }

    /// <summary>When <see cref="QkNorm"/> is set: <c>false</c> (Qwen3) normalizes each head over <see cref="HeadDim"/>; <c>true</c> (OLMoE) normalizes the whole Q vector over <see cref="QDim"/> (and K over <see cref="KvDim"/>) before splitting into heads. Detected from the q_norm weight length.</summary>
    public bool QkNormFullDim { get; init; }

    /// <summary>Whether <c>lm_head</c> shares the embedding table (no separate <c>lm_head.weight</c>).</summary>
    public bool TieWordEmbeddings { get; init; } = true;

    /// <summary>RoPE pairing convention; default <see cref="RopeStyle.SplitHalf"/> (Qwen2 / Llama / HF Qwen3 text). The Qwen3-TTS audio backbone uses <see cref="RopeStyle.Interleaved"/>.</summary>
    public RopeStyle Rope { get; init; } = RopeStyle.SplitHalf;

    /// <summary>FFN activation. Default SiLU (SwiGLU); Gemma uses <see cref="ActivationKind.GeluTanh"/>.</summary>
    public ActivationKind Activation { get; init; } = ActivationKind.Silu;

    /// <summary>Whether the FFN is gated (SwiGLU/GeGLU: <c>down(act(gate(x)) · up(x))</c>, the Llama/Qwen/Gemma default) or a plain non-gated MLP (<c>down(act(up(x)))</c>, no <c>gate_proj</c>) for GPT-2/Falcon/BLOOM/MPT/Nemotron/StarCoder.</summary>
    public bool GatedFfn { get; init; } = true;

    /// <summary>Whether the output (o_proj) and FFN (up/down/gate) projections carry a bias; <c>true</c> for the GPT-2/Falcon/BLOOM/MPT/StarCoder lineage, <c>false</c> for Llama/Qwen/Gemma. Loaded only when the bias tensors are present, so a stray <c>true</c> on a biasless model is harmless.</summary>
    public bool FfnBias { get; init; }

    /// <summary>Absolute learned position embeddings (GPT-2 / StarCoder): a <c>position_embd</c> table is added to the token embeddings and the model uses NO RoPE; <c>false</c> (default) = RoPE / ALiBi.</summary>
    public bool AbsolutePositionEmbeddings { get; init; }

    /// <summary>BLOOM's word-embedding LayerNorm: a LayerNorm (weight + bias) applied to the token embeddings <i>before</i> the first decoder block; <c>false</c> (default) for everything else.</summary>
    public bool EmbeddingLayerNorm { get; init; }

    /// <summary>Norm placement relative to the residual; default <see cref="NormPlacement.PreNorm"/> (Llama/Qwen/Gemma), OLMo-2 uses <see cref="NormPlacement.PostNorm"/> (attn/FFN read the raw residual, their output is normed before the add — no input/pre-FFN norm).</summary>
    public NormPlacement NormPlacement { get; init; } = NormPlacement.PreNorm;

    /// <summary>Gemma "sandwich norm": an extra RMSNorm on the attention output and on the FFN output, each applied <i>before</i> the residual add (in addition to the pre-attn / pre-FFN norms); false for the standard pre-norm layout (Qwen/Llama).</summary>
    public bool SandwichNorm { get; init; }

    /// <summary>Gemma's <c>(1 + weight)</c> RMSNorm: the norm weight is offset by 1 (the GGUF/HF weight stores the centered value); baked into the loaded norm weights at load time, so the runtime norm op is unchanged.</summary>
    public bool RmsNormAddOne { get; init; }

    /// <summary>Use LayerNorm (mean-centered) instead of RMSNorm for all norms; <c>true</c> for Cohere (Command-R / cohere2), whose GGUF carries no norm bias so a zero bias is supplied.</summary>
    public bool UseLayerNorm { get; init; }

    /// <summary>Parallel residual (GPT-NeoX / Cohere): attention and the FFN both read the <i>same</i> normed input and their outputs are summed into the residual once (a single norm per layer, no pre-FFN norm), instead of the sequential pre-norm path; <c>true</c> for Cohere.</summary>
    public bool ParallelResidual { get; init; }

    /// <summary>Cohere2 interleaves sliding-window layers (with RoPE) and full-attention "global" layers that use <b>no</b> positional encoding (NoPE). When <c>true</c>, RoPE is skipped on the global layers (<see cref="IsGlobalLayer"/>); <c>false</c> for everything else.</summary>
    public bool NoRopeOnGlobalLayers { get; init; }

    /// <summary>Multiplier applied to token embeddings after lookup; 1 for Qwen/Llama, Gemma uses <c>sqrt(HiddenSize)</c> (the embedding normalizer).</summary>
    public float EmbeddingScale { get; init; } = 1f;

    /// <summary>Attention score scale numerator: when &gt; 0 the scale is <c>1/sqrt(QueryPreAttnScalar)</c> (Gemma's <c>query_pre_attn_scalar</c>, which can differ from head_dim, e.g. Gemma-2-27B); 0 (default) uses the standard <c>1/sqrt(HeadDim)</c>.</summary>
    public float QueryPreAttnScalar { get; init; }

    /// <summary>Gemma-3 dual-RoPE: local (sliding-window) layers use this base frequency while global layers use <see cref="RopeTheta"/>; 0 (default) means a single RoPE base for all layers.</summary>
    public float RopeLocalTheta { get; init; }

    /// <summary>Gemma-4: local (sliding-window) layers use a SMALLER head dimension than global layers — not just a different RoPE base (Gemma-3's <see cref="RopeLocalTheta"/>), the Q/K/V/O projection weights themselves are narrower (E2B: 256 local vs 512 global). 0 (default) means every layer shares <see cref="HeadDim"/> — every architecture except Gemma-4. See <see cref="HeadDimFor"/>.</summary>
    public int HeadDimSwa { get; init; }

    /// <summary>Local layers' partial-rotary dimension (parallels <see cref="RotaryDim"/> for <see cref="HeadDimSwa"/>); 0 = full rotary over <see cref="HeadDimSwa"/>. Only meaningful when <see cref="HeadDimSwa"/> is set.</summary>
    public int RotaryDimSwa { get; init; }

    /// <summary>The head dimension layer <paramref name="i"/> actually uses: <see cref="HeadDimSwa"/> for a local/SWA layer when set, else <see cref="HeadDim"/> for every layer (every architecture but Gemma-4).</summary>
    public int HeadDimFor(int i) => HeadDimSwa > 0 && !IsGlobalLayer(i) ? HeadDimSwa : HeadDim;

    /// <summary>The (possibly partial) rotary dimension layer <paramref name="i"/> actually uses — parallels <see cref="HeadDimFor"/> for <see cref="RotaryDim"/>/<see cref="RotaryDimSwa"/>.</summary>
    public int RotaryDimFor(int i) => HeadDimSwa > 0 && !IsGlobalLayer(i) ? RotaryDimSwa : RotaryDim;

    /// <summary>Sliding-window layer pattern: layer <c>i</c> is a global (full-attention) layer when this is 0 (all global) or <c>(i+1) % pattern == 0</c>; otherwise it is a local (sliding-window) layer. Gemma-2 alternates (pattern 2); Gemma-3 is 5 local : 1 global (pattern 6).</summary>
    public int SlidingWindowPattern { get; init; }

    /// <summary>Sliding-window span for local layers (Gemma-2 4096, Gemma-3 512/1024); only affects sequences longer than the window, shorter contexts are unaffected (every position is in-window).</summary>
    public int SlidingWindow { get; init; }

    /// <summary>Granite's direct attention-score multiplier (its <c>attention.scale</c>, e.g. 0.015625), used in place of <c>1/sqrt(HeadDim)</c>; 0 (default) means the standard scale. Takes precedence over <see cref="QueryPreAttnScalar"/>.</summary>
    public float AttentionMultiplier { get; init; }

    /// <summary>Granite's residual multiplier (its <c>residual_scale</c>, e.g. 0.22): each sublayer output is scaled by this before being added to the residual stream; 1 (default) is a no-op.</summary>
    public float ResidualMultiplier { get; init; } = 1f;

    /// <summary>Granite's logit scaling (its <c>logit_scale</c>, e.g. 8.0): final logits are divided by this; 1 (default) is a no-op.</summary>
    public float LogitScale { get; init; } = 1f;

    /// <summary>Attention-logit soft-cap (Gemma-2 = 50): scores are capped via <c>cap·tanh(score/cap)</c> before the softmax; 0 (default) disables it. Requires the soft-capping attention path.</summary>
    public float AttnLogitSoftcap { get; init; }

    /// <summary>ALiBi (Attention with Linear Biases) maximum bias (llama.cpp <c>max_alibi_bias</c>, typically 8). When &gt; 0 the model uses NO RoPE; instead each head adds a per-head linear distance penalty <c>-slope_h·(q_pos − k_pos)</c> to the attention scores (the geometric slope schedule of MPT/BLOOM/Falcon-classic/Jais/RefAct). 0 (default) = RoPE/standard attention, no ALiBi.</summary>
    public float AlibiMaxBias { get; init; }

    /// <summary>Per-head ALiBi slopes for <see cref="NumHeads"/> heads (geometric schedule from <see cref="AlibiMaxBias"/>), or an empty array when ALiBi is off. Computed once at config build.</summary>
    public IReadOnlyList<float> AlibiSlopes { get; init; } = System.Array.Empty<float>();

    /// <summary>GPT-OSS attention sinks: each layer carries a learned per-head <c>self_attn.sinks</c> logit that joins the softmax denominator but contributes no value (lets a head attend to "nothing"). When true the per-layer sink tensor is loaded and passed to FlashAttention.</summary>
    public bool AttnSink { get; init; }

    /// <summary>Final-logit soft-cap (Gemma-2 = 30): output logits are capped via <c>cap·tanh(logit/cap)</c>; 0 (default) disables it.</summary>
    public float FinalLogitSoftcap { get; init; }

    /// <summary>Explicit per-layer global/local override (Gemma-4: GGUF stores <c>attention.sliding_window_pattern</c> as a real per-layer bool array, not a broadcast period); null (default) falls back to <see cref="SlidingWindowPattern"/>'s modulo period — every other architecture.</summary>
    public IReadOnlyList<bool>? GlobalLayers { get; init; }

    /// <summary>Whether layer <paramref name="i"/> uses global (full) attention vs a local sliding window.</summary>
    public bool IsGlobalLayer(int i) => GlobalLayers is not null ? GlobalLayers[i]
        : SlidingWindowPattern <= 0 || (i + 1) % SlidingWindowPattern == 0;

    /// <summary>RoPE base frequency for layer <paramref name="i"/> (Gemma-3 dual-RoPE: local layers use <see cref="RopeLocalTheta"/>, global use <see cref="RopeTheta"/>).</summary>
    public float RopeThetaFor(int i) => RopeLocalTheta > 0 && !IsGlobalLayer(i) ? RopeLocalTheta : RopeTheta;

    /// <summary>Attention score scale: Granite's direct <see cref="AttentionMultiplier"/> if set, else <c>1/sqrt(QueryPreAttnScalar)</c> when set, else the standard <c>1/sqrt(HeadDim)</c>.</summary>
    public float AttnScale => AttentionMultiplier > 0 ? AttentionMultiplier
        : 1f / MathF.Sqrt(QueryPreAttnScalar > 0 ? QueryPreAttnScalar : HeadDim);

    /// <summary>For quantized (GGUF) weights: when <c>false</c> (default) the dequantized F16 weight is cached per weight (fast decode, but the weight set occupies F16-sized VRAM); when <c>true</c> the low-VRAM <see cref="IBackend.QuantizedMatMul"/> path is used (weights stay compressed on-device, dequant is transient per call), trading decode speed for a much smaller footprint so large models fit. No effect on float weights.</summary>
    /// <remarks>Default is taken from the <c>HARTSY_LOWVRAM_QUANT</c> env var (set "1" to enable) so it can be flipped for A/B benchmarking without recompiling (see docs/Checklists/QUANT_GEMM_PERF_PLAN.md); an explicit <c>init</c> value always overrides the env default.</remarks>
    public bool LowVramQuant { get; init; } = Environment.GetEnvironmentVariable("HARTSY_LOWVRAM_QUANT") == "1";

    /// <summary>Total Q projection output dim — <see cref="NumHeads"/> × <see cref="HeadDim"/>.</summary>
    public int QDim => NumHeads * HeadDim;

    /// <summary>Total K/V projection output dim — <see cref="NumKvHeads"/> × <see cref="HeadDim"/>.</summary>
    public int KvDim => NumKvHeads * HeadDim;

    /// <summary>Query heads per KV group (GQA repeat factor).</summary>
    public int KvGroup => NumHeads / NumKvHeads;

    /// <summary>Mixture-of-Experts FFN config, or <c>null</c> for a dense SwiGLU FFN (Qwen2/Qwen3/Llama dense).</summary>
    public MoeConfig? Moe { get; init; }

    /// <summary>Multi-head Latent Attention config (DeepSeek-V2/V3, Kimi-K2), or <c>null</c> for standard per-head Q/K/V attention. When set, the layer uses the latent-KV attention path.</summary>
    public MlaConfig? Mla { get; init; }

    /// <summary>Per-routed-expert output scale (DeepSeek <c>routed_scaling_factor</c> / <c>expert_weights_scale</c>); 1 (default) is a no-op. Applied to the combined routed-expert contribution.</summary>
    public float ExpertWeightsScale { get; init; } = 1f;

    /// <summary>Whether layer <paramref name="layerIndex"/> uses the MoE FFN (vs dense SwiGLU).</summary>
    public bool IsMoeLayer(int layerIndex) => Moe is not null && layerIndex >= Moe.FirstDenseLayers;

    /// <summary>Layer indices that are gated cross-attention layers (Llama-3.2-Vision / mllama: <c>[3,8,13,18,23,28,33,38]</c> for 11B); empty for every text-only model. A cross-attention layer reads the encoded image features via <see cref="HartsyInference.LLM.Multimodal.MllamaCrossAttentionLayer"/> instead of the causal self-attention path.</summary>
    public IReadOnlySet<int> CrossAttnLayers { get; init; } = System.Collections.Immutable.ImmutableHashSet<int>.Empty;

    /// <summary>Whether layer <paramref name="layerIndex"/> is a gated cross-attention layer (mllama).</summary>
    public bool IsCrossAttnLayer(int layerIndex) => CrossAttnLayers.Contains(layerIndex);

    /// <summary>Gemma-4 per-layer embeddings (PLE): width of a small extra embedding mixed into every layer's output (a Gemma-3n-lineage mechanism carried into Gemma-4); 0 (default) disables it — every other architecture. When set, <see cref="GenericTransformer"/> loads the top-level <c>per_layer_token_embd</c>/<c>per_layer_model_proj</c>/<c>per_layer_proj_norm</c> tensors plus each layer's own <c>per_layer_inp_gate</c>/<c>per_layer_proj</c>/<c>per_layer_post_norm</c>.</summary>
    public int PerLayerEmbeddingDim { get; init; }

    /// <summary>Gemma-4 (E2B/E4B mobile variants): from this layer index onward, layers compute NO K/V of their own — they reuse an earlier "donor" layer's already-populated KV cache slot (a Gemma-3n-lineage memory-saving trick); 0 (default) disables this — every layer owns its KV. See <see cref="HasOwnKv"/>/<see cref="KvDonorLayer"/>.</summary>
    public int KvSharedFromLayer { get; init; }

    /// <summary>Whether layer <paramref name="i"/> computes its own K/V projections (true for every layer when <see cref="KvSharedFromLayer"/> is 0/disabled, or for any layer before that cutoff).</summary>
    public bool HasOwnKv(int i) => KvSharedFromLayer <= 0 || i < KvSharedFromLayer;

    /// <summary>The donor layer index whose KV cache slot layer <paramref name="i"/> reads instead of its own (only meaningful when <see cref="HasOwnKv"/> is false for <paramref name="i"/>). Ported from llama.cpp's reuse callback: the last layer of the SAME type (global vs local/SWA) among the layers that still compute their own KV — <c>cutoff-1</c> for a global layer, <c>cutoff-2</c> for a local one (the layer immediately before the cutoff is always global, so the last local layer sits one further back).</summary>
    public int KvDonorLayer(int i) => IsGlobalLayer(i) ? KvSharedFromLayer - 1 : KvSharedFromLayer - 2;

    /// <summary>Gemma-4: RMS-normalizes V (weightless — no learned scale, just <c>v / rms(v)</c>) after the V projection, before RoPE/cache-append; <c>false</c> (default) for everything else, which apply no V norm.</summary>
    public bool VNorm { get; init; }

    /// <summary>Gemma-4 MoE layers run the routed-expert FFN and the plain dense (gated) FFN as two INDEPENDENT parallel branches — each reading the same attention output through its own pre-norm, each normed again after its own FFN, then summed — rather than the usual "shared expert fused into the router output" pattern every other MoE architecture here uses. <c>false</c> (default) for everything else.</summary>
    public bool ParallelDenseMoeBranch { get; init; }

    /// <summary>Computes the per-head ALiBi slopes (llama.cpp / the original ALiBi paper's geometric schedule): with <c>n2 = 2^floor(log2(nHead))</c>, <c>m0 = 2^(−maxBias/n2)</c>, <c>m1 = 2^(−maxBias/2/n2)</c>, head <c>h</c> gets <c>m0^(h+1)</c> for <c>h &lt; n2</c> else <c>m1^(2(h−n2)+1)</c> (handles non-power-of-two head counts).</summary>
    public static float[] ComputeAlibiSlopes(int nHead, float maxBias)
    {
        float[] slopes = new float[nHead];
        if (maxBias <= 0f || nHead <= 0) return slopes;
        int n2 = 1;
        while (n2 * 2 <= nHead) n2 *= 2;   // 2^floor(log2(nHead))
        double m0 = Math.Pow(2.0, -maxBias / n2);
        double m1 = Math.Pow(2.0, -(maxBias / 2.0) / n2);
        for (int h = 0; h < nHead; h++)
            slopes[h] = (float)(h < n2 ? Math.Pow(m0, h + 1) : Math.Pow(m1, 2 * (h - n2) + 1));
        return slopes;
    }

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

    /// <summary>Qwen3-0.6B (decoupled head_dim 16·128=2048 ≠ hidden 1024, no QKV bias, per-head Q/K norm, tied); exercises all three Qwen3 variation axes.</summary>
    public static TransformerConfig Qwen3_0_6B => new()
    {
        HiddenSize = 1_024, NumLayers = 28, NumHeads = 16, NumKvHeads = 8, HeadDim = 128,
        IntermediateSize = 3_072, VocabSize = 151_936, MaxPositionEmbeddings = 40_960,
        AttentionBias = false, QkNorm = true, TieWordEmbeddings = true,
    };

    /// <summary>Gemma-3-1B-it: decoupled head_dim 256 (4·256=1024 ≠ hidden 1152), MQA (1 KV head), Q/K norm, GeGLU, sandwich + (1+w) norms, embedding scale √1152, dual-RoPE (local 10k / global 1M, 5:1 pattern), no logit soft-cap; exercises the full Gemma-3 surface.</summary>
    public static TransformerConfig Gemma3_1B => new()
    {
        HiddenSize = 1_152, NumLayers = 26, NumHeads = 4, NumKvHeads = 1, HeadDim = 256,
        IntermediateSize = 6_912, VocabSize = 262_144, MaxPositionEmbeddings = 32_768,
        RopeTheta = 1_000_000f, RmsNormEps = 1e-6f, AttentionBias = false, QkNorm = true, TieWordEmbeddings = true,
        Activation = ActivationKind.GeluTanh, SandwichNorm = true, RmsNormAddOne = true,
        EmbeddingScale = 33.941124f /* sqrt(1152) */, QueryPreAttnScalar = 256f,
        RopeLocalTheta = 10_000f, SlidingWindowPattern = 6, SlidingWindow = 512,
    };

    /// <summary>Gemma-2-2B-it: head_dim 256, GQA (8/4), GeGLU, sandwich + (1+w) norms, embedding scale √2304, attn/final logit soft-cap (50/30), alternating sliding window (pattern 2, window 4096), single RoPE 10k.</summary>
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

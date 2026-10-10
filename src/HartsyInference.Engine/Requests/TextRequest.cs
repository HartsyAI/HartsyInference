using HartsyInference.Core.Configuration;
using HartsyInference.Core.MemoryManagement;

namespace HartsyInference.Engine.Requests;

/// <summary>Native chat/completion request. Carries the conversation, sampling knobs, tool definitions, and the decode hints the local backend honors. Per-request knobs (temperature/topP/seed/maxTokens) live here; nullable knobs fall back to the model/engine default when unset.</summary>
public sealed record TextRequest
{
    /// <summary>The conversation so far, oldest first.</summary>
    public required IReadOnlyList<TextMessage> Messages { get; init; }

    /// <summary>Pre-tokenized prompt ids, fed to the model as given: no chat template, no tokenizer and no output parser. Wins over
    /// <see cref="Messages"/>, which is then not read. Null (the default) renders <see cref="Messages"/> as usual.</summary>
    public IReadOnlyList<int>? RawTokenIds { get; init; }

    /// <summary>System prompt applied ahead of <see cref="Messages"/>; null/empty for none.</summary>
    public string? SystemPrompt { get; init; }

    /// <summary>Sampling temperature; &lt;= 0 selects greedy decoding.</summary>
    public double Temperature { get; init; } = 0.7;

    /// <summary>Nucleus (top-P) cutoff.</summary>
    public double TopP { get; init; } = 0.95;

    /// <summary>Top-K cutoff; null uses the model default.</summary>
    public int? TopK { get; init; }

    /// <summary>Min-P cutoff; null uses the model default.</summary>
    public double? MinP { get; init; }

    /// <summary>Repetition penalty; null uses the model default.</summary>
    public double? RepetitionPenalty { get; init; }

    /// <summary>Maximum tokens to generate.</summary>
    public int MaxTokens { get; init; } = 4096;

    /// <summary>Sampling seed; negative means a random seed per request.</summary>
    public long Seed { get; init; } = -1;

    /// <summary>Force greedy decoding regardless of temperature.</summary>
    public bool Greedy { get; init; }

    /// <summary>Sets the model's chat-template <c>enable_thinking</c> variable (Qwen3-family reasoning-block toggle); null leaves it undefined so the template falls back to its own default. Ignored by templates without a thinking slot.</summary>
    public bool? EnableThinking { get; init; }

    /// <summary>The tenant this request runs under, for per-tenant state such as the prefix cache. Null takes the current caller's identity (<see cref="TenantContext"/>),
    /// else <see cref="TenantContext.Local"/>. The API never sets this from client input.</summary>
    public string? TenantId { get; init; }

    /// <summary>Reasoning effort in [1, 100] for a template that takes one (DeepSeek-V4.1 thinking mode); null uses the template's default (75). Templates without an effort slot ignore it.</summary>
    public int? ReasoningEffort { get; init; }

    /// <summary>The caller's own user identifier (OpenAI's <c>user</c>), carried on the request; the engine does not use it yet. It does not choose the tenant: the tenant comes from the API key, so a client cannot claim another tenant's cache.</summary>
    public string? User { get; init; }

    /// <summary>Queue priority when the server is busy: a higher priority is admitted first, and equal priorities keep arrival order. Null is <see cref="RequestPriority.Normal"/>.</summary>
    public RequestPriority? Priority { get; init; }

    /// <summary>Target device key (e.g. "cpu", "cuda:0"); null uses the backend's primary device.</summary>
    /// <remarks>The slot builds its own backend for this key and gates only that ordinal, so an engine built on one card
    /// can serve its LLM on another. On such an engine every generate and stream request must set this, or the model
    /// loads on the engine's own card. <see cref="Services.ITextService.CountTokens"/> takes no device and loads
    /// nothing.</remarks>
    public string? Device { get; init; }

    /// <summary>Tool definitions offered to the model; null/empty disables tool calling.</summary>
    public IReadOnlyList<ToolDefinition>? Tools { get; init; }

    /// <summary>Force the model to call this tool by name; null lets it choose. Not implemented yet: carried for API compatibility, no template or grammar forcing happens.</summary>
    public string? ForceToolId { get; init; }

    /// <summary>Enable graph-mode decode when supported; null uses the engine default.</summary>
    public bool? GraphDecode { get; init; }

    /// <summary>Enable speculative decode when supported; null uses the engine default.</summary>
    public bool? SpeculativeDecode { get; init; }

    /// <summary>Low-VRAM on-the-fly quant to load weights at (e.g. "q8_0"); null loads at full precision.</summary>
    public string? LowVramQuant { get; init; }

    /// <summary>Where the model is placed when this request loads it: <c>auto</c>, <c>gpu</c>, <c>split</c> or <c>offload</c>.
    /// Null uses the <c>vram.textPlacement</c> setting, which defaults to <c>auto</c>. Applies to a GGUF loaded on a single CUDA
    /// device key; an explicit multi-device key (<c>cuda:0+cuda:1</c>) is a split already. Takes effect at load, like
    /// <see cref="LowVramQuant"/>.</summary>
    public string? Placement { get; init; }

    /// <summary>Free the model's device memory after this request completes; null uses the engine default.</summary>
    public bool? AlwaysFreeMemory { get; init; }

    /// <summary>Per-request VRAM lever overrides; null follows the backend's policy.</summary>
    public VramOverrides? Vram { get; init; }

    /// <summary>Per-request engine settings (profile + individual overrides); null keeps the machine's configuration.</summary>
    public RequestSettings? Settings { get; init; }

    /// <summary>Opt-in prefix-KV reuse key: calls sharing the same non-null key on the same device/model slot reuse
    /// the longest common token-id prefix of their rendered prompts instead of each prefilling from scratch — e.g.
    /// one phone call's conversation across turns. Null (the default) is the original per-call behavior: every
    /// request prefills its whole prompt and the KV cache is discarded when the call returns. A second concurrent
    /// request on a busy key (already mid-generation) runs uncached rather than waiting or corrupting it. See
    /// <see cref="HartsyInference.LLM.Generation.RetainedSequenceStore"/>.</summary>
    public string? PrefixCacheKey { get; init; }

    /// <summary>Sizes a brand-new retained sequence's first KV allocation (tokens) the first time
    /// <see cref="PrefixCacheKey"/> is used; null sizes it to this request's own prompt + <see cref="MaxTokens"/>.
    /// Rarely worth setting: what a request retains is shrunk to its length plus <c>vram.prefixCacheHeadroomTokens</c>
    /// when it ends, and a later request that needs more room grows it by copying the reusable prefix on device, so
    /// the hint no longer has to cover the conversation's growth.</summary>
    public int? PrefixCacheCapacityHint { get; init; }

    /// <summary>Overrides the device backend's <c>CacheWeightCasts</c> (cache a dequantized copy of quantized
    /// weights vs. a transient per-GEMM dequant); null leaves the backend's own default (on). Takes effect when
    /// the slot's backend is first created for this device, like <see cref="LowVramQuant"/> — a later request on
    /// an already-loaded slot does not change it without a reload. Measured on Qwen3-4B-Q4_K_M/4090: on costs
    /// ~7.3 GB resident once warm; off costs a ~50 ms fixed dequant tax per prefill call (prompt-length
    /// independent — decode's quantized GEMV path is unaffected either way) but nothing else resident.</summary>
    public bool? CacheWeightCasts { get; init; }

    /// <summary>Overrides whether a single-device load's initial weight upload includes the load-time-fused
    /// Q/K/V and gate/up projections' ORIGINAL split tensors, on top of their fused replacements — see
    /// <see cref="HartsyInference.LLM.Transformer.GenericTransformer.EnumerateWeights"/>'s <c>includeRedundantSplits</c>.
    /// Null preserves the existing default (included, matching every other caller of this request type historically).
    /// False excludes them: the fused tensors alone serve every <c>TextGenerationPipeline</c> single-sequence decode
    /// and prefill path, so for that path the split originals are pure duplicate upload — only the batch scheduler's
    /// mixed-dtype split-projection path reads them (via its own lazy auto-promotion on first use, unaffected by
    /// this flag). Takes effect when the slot's backend is first created for this device, like
    /// <see cref="CacheWeightCasts"/> — a later request on an already-loaded slot does not change it without a
    /// reload. Measured on Qwen3-4B-Q4_K_M: the split originals are ~1.21 GiB of the ~3.53 GiB default single-device
    /// upload (file is 2.33 GiB); only the sharded and tensor-parallel load paths already exclude them by default.</summary>
    public bool? PreloadRedundantWeightSplits { get; init; }
}

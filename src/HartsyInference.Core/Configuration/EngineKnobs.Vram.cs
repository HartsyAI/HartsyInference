namespace HartsyInference.Core.Configuration;

/// <summary>Residency, streaming, chunking and cache precision.</summary>
/// <remarks>Generated from the pre-migration call sites; defaults and grammars are those the code already had.</remarks>
public static partial class EngineKnobs
{
    /// <summary>Minimum activation/workspace headroom in MB a Wan-Animate-2 chunk reserves before weights are placed.</summary>
    public static readonly Knob<long> Animate2HeadroomMb =
        Long("vram.animate2HeadroomMb", 3072L, KnobScope.Runtime, KnobDomain.Vram, "Minimum activation/workspace headroom in MB a Wan-Animate-2 chunk reserves before weights are placed.");

    /// <summary>Minimum VRAM headroom reserved for Wan-Animate block streaming; floors the token-load-derived estimate.</summary>
    public static readonly Knob<long> AnimateHeadroomMb =
        Long("vram.animateHeadroomMb", 3072L, KnobScope.Runtime, KnobDomain.Vram, "Minimum VRAM headroom reserved for Wan-Animate block streaming; floors the token-load-derived estimate.");

    /// <summary>Routes capture-time intermediate allocations through a persistent bump arena instead of per-replay pool nodes.</summary>
    public static readonly Knob<bool> GraphArena =
        Bool("vram.graphArena", true, KnobScope.Runtime, KnobDomain.Vram, "Routes capture-time intermediate allocations through a persistent bump arena instead of per-replay pool nodes.");

    /// <summary>Stores the single-sequence decode KV cache as F16 instead of F32, halving its VRAM (CUDA only).</summary>
    public static readonly Knob<bool> KvF16 =
        Bool("vram.kvF16", false, KnobScope.Runtime, KnobDomain.Vram, "Stores the single-sequence decode KV cache as F16 instead of F32, halving its VRAM (CUDA only).");

    /// <summary>Makes LLMs keep GGUF weights compressed and use QuantizedMatMul instead of caching an F16 dequant.</summary>
    public static readonly Knob<bool> LowvramQuant =
        Bool("vram.lowvramQuant", false, KnobScope.Construction, KnobDomain.Vram, "Makes LLMs keep GGUF weights compressed and use QuantizedMatMul instead of caching an F16 dequant.");

    /// <summary>Spare KV capacity, in tokens, a retained prefix-cache sequence keeps past the tokens it holds; a
    /// larger allocation is copied down to its length plus this when its request ends, and grown again by copy when a
    /// later request needs more.</summary>
    public static readonly Knob<int> PrefixCacheHeadroomTokens =
        Int("vram.prefixCacheHeadroomTokens", 256, KnobScope.Runtime, KnobDomain.Vram,
            "Spare KV tokens a retained prefix-cache sequence keeps past its length; larger allocations shrink to this.",
            v => Math.Max(0, v));

    /// <summary>Routes a batch-capable model's chat requests through the continuous-batching scheduler, so concurrent requests join one shared decode round, instead of the per-slot pipeline that runs one request at a time. Read when a model loads, so a change applies from that model's next load. Off by default: the device-backed path has no real-generation A/B yet (AGENTS.md "Shipping a change" step 3). The V4.1 host model, which has no KV pool, decodes at most four sequences at once.</summary>
    public static readonly Knob<bool> ContinuousBatching =
        Bool("vram.continuousBatching", false, KnobScope.Runtime, KnobDomain.Vram,
            "Routes a batch-capable model's chat requests through the continuous-batching scheduler (concurrent requests share one decode round) instead of the per-slot pipeline. "
            + "Read when a model loads, so a change applies on its next load. A scheduled request holds its device's lease for its whole run, so a reload or a pipeline "
            + "request on that device waits, up to the reload timeout, for it. The V4.1 host model, which has no KV pool, decodes at most four sequences at once.");

    /// <summary>Most retained sequences (<c>TextRequest.PrefixCacheKey</c>) one device slot's prefix-KV store keeps at once; least-recently-used entries are evicted first past this.</summary>
    public static readonly Knob<int> PrefixCacheMaxEntries =
        Int("vram.prefixCacheMaxEntries", 4, KnobScope.Runtime, KnobDomain.Vram, "Most retained prefix-KV sequences one device slot's store keeps at once (LRU past this).");

    /// <summary>Byte budget for one device slot's prefix-KV store: the sum of every retained sequence's KV after
    /// shrinking, and the most one sequence may keep — a longer one is freed when its request ends instead of
    /// retained. Least-recently-used entries are evicted first past it.</summary>
    public static readonly Knob<long> PrefixCacheMaxBytes =
        Long("vram.prefixCacheMaxBytes", 1536L << 20, KnobScope.Runtime, KnobDomain.Vram,
            "Byte budget for one device slot's retained prefix-KV sequences, and the most one may keep (LRU past this).",
            v => Math.Max(1L, v));

    /// <summary>Pins the LTX-2.5 diffusion decoder's temporal-chunk workspace in MB instead of sizing the plan off free VRAM.</summary>
    public static readonly Knob<long> Ltx25VaeChunkMb =
        Long("vram.ltx25VaeChunkMb", 0L, KnobScope.Construction, KnobDomain.Vram, "Pins the LTX-2.5 diffusion decoder's temporal-chunk workspace in MB instead of sizing the plan off free VRAM.");

    /// <summary>Activation/workspace headroom in MB (default 3072) the LTX-2 denoise loop reserves before weight placement.</summary>
    public static readonly Knob<long> Ltx2HeadroomMb =
        Long("vram.ltx2HeadroomMb", 3072L, KnobScope.Runtime, KnobDomain.Vram, "Activation/workspace headroom in MB (default 3072) the LTX-2 denoise loop reserves before weight placement.");

    /// <summary>Keeps freed activation buffers warm in the CUDA stream-ordered mempool instead of releasing to the driver.</summary>
    public static readonly Knob<bool> MempoolKeep =
        Bool("vram.mempoolKeep", true, KnobScope.Construction, KnobDomain.Vram, "Keeps freed activation buffers warm in the CUDA stream-ordered mempool instead of releasing to the driver.");

    /// <summary>=1 disables promoting repeatedly-uploaded weights to resident device copies, restoring re-upload per use.</summary>
    public static readonly Knob<bool> NoAutopromote =
        Bool("vram.noAutopromote", false, KnobScope.Runtime, KnobDomain.Vram, "=1 disables promoting repeatedly-uploaded weights to resident device copies, restoring re-upload per use.");

    /// <summary>Kill-switch for freeing activation buffers displaced by a cache rebind; 0 restores the pre-fix leak.</summary>
    public static readonly Knob<bool> OrphanSweep =
        Bool("vram.orphanSweep", true, KnobScope.Runtime, KnobDomain.Vram, "Kill-switch for freeing activation buffers displaced by a cache rebind; 0 restores the pre-fix leak.");

    /// <summary>Forces every CUDA peer-access query to false, so cross-GPU copies never use direct P2P/NVLink addressing.</summary>
    public static readonly Knob<bool> P2pDisable =
        Bool("vram.p2pDisable", false, KnobScope.Runtime, KnobDomain.Vram, "Forces every CUDA peer-access query to false, so cross-GPU copies never use direct P2P/NVLink addressing.");

    /// <summary>Disables the process-wide gate that serializes generations sharing one GPU ordinal.</summary>
    public static readonly Knob<bool> SameGpuConcurrent =
        Bool("vram.sameGpuConcurrent", false, KnobScope.Runtime, KnobDomain.Vram, "Disables the process-wide gate that serializes generations sharing one GPU ordinal.");

    /// <summary>Forces the query-tiled F32 SDPA path on CUDA and Vulkan instead of materializing the full score matrix.</summary>
    public static readonly Knob<bool> SdpaForceTiled =
        Bool("vram.sdpaForceTiled", false, KnobScope.Runtime, KnobDomain.Vram, "Forces the query-tiled F32 SDPA path on CUDA and Vulkan instead of materializing the full score matrix.");

    /// <summary>=1 pages the step cache's cross-step residual and indicator snapshot to host memory as they are produced.</summary>
    public static readonly Knob<bool> StepCacheOffload =
        Bool("vram.stepCacheOffload", false, KnobScope.Runtime, KnobDomain.Vram, "=1 pages the step cache's cross-step residual and indicator snapshot to host memory as they are produced.");

    /// <summary>Registers block-streaming host weight sources as pinned memory so H2D uploads overlap compute.</summary>
    public static readonly Knob<bool> StreamPin =
        Bool("vram.streamPin", true, KnobScope.Runtime, KnobDomain.Vram, "Registers block-streaming host weight sources as pinned memory so H2D uploads overlap compute.");

    /// <summary>Kill-switch for the VAE decoder's full-resolution direct-decode attempt; 0 forces always-tiled decoding.</summary>
    public static readonly Knob<bool> VaeFullres =
        Bool("vram.vaeFullres", true, KnobScope.Runtime, KnobDomain.Vram, "Kill-switch for the VAE decoder's full-resolution direct-decode attempt; 0 forces always-tiled decoding.");

    /// <summary>Weights stay resident on the device between generations. The <c>Auto</c> tier's fallback answer.</summary>
    public static readonly Knob<bool> KeepModels =
        Bool("vram.keepModels", true, KnobScope.Runtime, KnobDomain.Vram,
            "Weights stay resident on the device between generations.");

    /// <summary>Streaming/eviction posture when nothing else specifies one: <c>auto</c>, <c>on</c>, or <c>off</c>.</summary>
    /// <remarks>Kept as a string because it is a three-state word, and <c>LowVramMode.Parse</c> owns the spelling
    /// table plus its warn-on-garbage log line.</remarks>
    public static readonly Knob<string?> LowVram =
        Str("vram.lowVram", null, KnobScope.Runtime, KnobDomain.Vram,
            "Streaming and eviction posture when nothing else specifies one: auto, on, or off.");
    /// <summary>Across-step feature-cache drift threshold: unset/0 off, 1 the calibrated profile, or a non-negative float.</summary>
    /// <remarks>The step-cache family stays <c>string?</c> because <c>StepCacheEnv</c> owns a three-way grammar and
    /// THROWS on a malformed value by design — silently ignoring a mistyped perf knob would invalidate an A/B run.</remarks>
    public static readonly Knob<string?> StepCache =
        Str("vram.stepCache", null, KnobScope.Runtime, KnobDomain.Vram,
            "Across-step feature-cache drift threshold: unset/0 off, 1 the calibrated profile, or a non-negative float.");

    /// <summary>Max consecutive cached steps; default 3. Must be a positive integer.</summary>
    public static readonly Knob<string?> StepCacheCap =
        Str("vram.stepCacheCap", null, KnobScope.Runtime, KnobDomain.Vram,
            "Maximum consecutive cached steps; default 3.");

    /// <summary>TeaCache-style gate calibration polynomial, lowest power first, comma separated.</summary>
    public static readonly Knob<string?> StepCachePoly =
        Str("vram.stepCachePoly", null, KnobScope.Runtime, KnobDomain.Vram,
            "TeaCache-style gate calibration polynomial, lowest power first, comma separated.");

    /// <summary>Fraction of the schedule measured from the END where cache reuse is allowed; 0 means the whole schedule.</summary>
    public static readonly Knob<string?> StepCacheLate =
        Str("vram.stepCacheLate", null, KnobScope.Runtime, KnobDomain.Vram,
            "Fraction of the schedule from the end where cache reuse is allowed; 0 is the whole schedule.");

    /// <summary>CSV path to log indicator/residual-drift pairs to, for fitting the gate polynomial.</summary>
    public static readonly Knob<string?> StepCacheCalib =
        Str("vram.stepCacheCalib", null, KnobScope.Runtime, KnobDomain.Vram,
            "CSV path to log indicator and residual-drift pairs to, for fitting the gate polynomial.");

    /// <summary>Wan-Animate-2 driving-cache dtype: on/1 = BF16, off/0 = F32, auto/unset decides from measured VRAM.</summary>
    /// <remarks>Kept as a string: the policy owns an on/off/auto spelling table plus its own dedup logging, and
    /// "on" here means BF16 — a precision trade, not a streaming one.</remarks>
    public static readonly Knob<string?> Animate2Bf16DrivingCache =
        Str("vram.animate2Bf16DrivingCache", null, KnobScope.Runtime, KnobDomain.Vram,
            "Wan-Animate-2 driving-cache dtype: on = BF16, off = F32, auto decides from measured free VRAM.");
}

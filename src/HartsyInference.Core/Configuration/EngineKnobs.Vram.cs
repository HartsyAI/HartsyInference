namespace HartsyInference.Core.Configuration;

/// <summary>Residency, streaming, chunking and cache precision.</summary>
/// <remarks>Generated from the pre-migration call sites; defaults and grammars are those the code already had.</remarks>
public static partial class EngineKnobs
{
    /// <summary>Minimum activation/workspace headroom in MB a Wan-Animate-2 chunk reserves before weights are placed.</summary>
    public static readonly Knob<long> Animate2HeadroomMb =
        Long("vram.animate2HeadroomMb", "HARTSY_ANIMATE2_HEADROOM_MB", 3072L, KnobScope.Runtime, KnobDomain.Vram, "Minimum activation/workspace headroom in MB a Wan-Animate-2 chunk reserves before weights are placed.");

    /// <summary>Minimum VRAM headroom reserved for Wan-Animate block streaming; floors the token-load-derived estimate.</summary>
    public static readonly Knob<long> AnimateHeadroomMb =
        Long("vram.animateHeadroomMb", "HARTSY_ANIMATE_HEADROOM_MB", 3072L, KnobScope.Runtime, KnobDomain.Vram, "Minimum VRAM headroom reserved for Wan-Animate block streaming; floors the token-load-derived estimate.");

    /// <summary>Routes capture-time intermediate allocations through a persistent bump arena instead of per-replay pool nodes.</summary>
    public static readonly Knob<bool> GraphArena =
        Bool("vram.graphArena", "HARTSY_GRAPH_ARENA", true, KnobScope.Runtime, KnobDomain.Vram, "Routes capture-time intermediate allocations through a persistent bump arena instead of per-replay pool nodes.");

    /// <summary>Stores the single-sequence decode KV cache as F16 instead of F32, halving its VRAM (CUDA only).</summary>
    public static readonly Knob<bool> KvF16 =
        Bool("vram.kvF16", "HARTSY_KV_F16", false, KnobScope.Runtime, KnobDomain.Vram, "Stores the single-sequence decode KV cache as F16 instead of F32, halving its VRAM (CUDA only).");

    /// <summary>Makes LLMs keep GGUF weights compressed and use QuantizedMatMul instead of caching an F16 dequant.</summary>
    public static readonly Knob<bool> LowvramQuant =
        Bool("vram.lowvramQuant", "HARTSY_LOWVRAM_QUANT", false, KnobScope.Construction, KnobDomain.Vram, "Makes LLMs keep GGUF weights compressed and use QuantizedMatMul instead of caching an F16 dequant.");

    /// <summary>Pins the LTX-2.5 diffusion decoder's temporal-chunk workspace in MB instead of sizing the plan off free VRAM.</summary>
    public static readonly Knob<long> Ltx25VaeChunkMb =
        Long("vram.ltx25VaeChunkMb", "HARTSY_LTX25_VAE_CHUNK_MB", 0L, KnobScope.Construction, KnobDomain.Vram, "Pins the LTX-2.5 diffusion decoder's temporal-chunk workspace in MB instead of sizing the plan off free VRAM.");

    /// <summary>Activation/workspace headroom in MB (default 3072) the LTX-2 denoise loop reserves before weight placement.</summary>
    public static readonly Knob<long> Ltx2HeadroomMb =
        Long("vram.ltx2HeadroomMb", "HARTSY_LTX2_HEADROOM_MB", 3072L, KnobScope.Runtime, KnobDomain.Vram, "Activation/workspace headroom in MB (default 3072) the LTX-2 denoise loop reserves before weight placement.");

    /// <summary>Keeps freed activation buffers warm in the CUDA stream-ordered mempool instead of releasing to the driver.</summary>
    public static readonly Knob<bool> MempoolKeep =
        Bool("vram.mempoolKeep", "HARTSY_MEMPOOL_KEEP", true, KnobScope.Construction, KnobDomain.Vram, "Keeps freed activation buffers warm in the CUDA stream-ordered mempool instead of releasing to the driver.");

    /// <summary>=1 disables promoting repeatedly-uploaded weights to resident device copies, restoring re-upload per use.</summary>
    public static readonly Knob<bool> NoAutopromote =
        Bool("vram.noAutopromote", "HARTSY_NO_AUTOPROMOTE", false, KnobScope.Runtime, KnobDomain.Vram, "=1 disables promoting repeatedly-uploaded weights to resident device copies, restoring re-upload per use.");

    /// <summary>Kill-switch for freeing activation buffers displaced by a cache rebind; 0 restores the pre-fix leak.</summary>
    public static readonly Knob<bool> OrphanSweep =
        Bool("vram.orphanSweep", "HARTSY_ORPHAN_SWEEP", true, KnobScope.Runtime, KnobDomain.Vram, "Kill-switch for freeing activation buffers displaced by a cache rebind; 0 restores the pre-fix leak.");

    /// <summary>Forces every CUDA peer-access query to false, so cross-GPU copies never use direct P2P/NVLink addressing.</summary>
    public static readonly Knob<bool> P2pDisable =
        Bool("vram.p2pDisable", "HARTSY_P2P_DISABLE", false, KnobScope.Runtime, KnobDomain.Vram, "Forces every CUDA peer-access query to false, so cross-GPU copies never use direct P2P/NVLink addressing.");

    /// <summary>Disables the process-wide gate that serializes generations sharing one GPU ordinal.</summary>
    public static readonly Knob<bool> SameGpuConcurrent =
        Bool("vram.sameGpuConcurrent", "HARTSY_SAME_GPU_CONCURRENT", false, KnobScope.Runtime, KnobDomain.Vram, "Disables the process-wide gate that serializes generations sharing one GPU ordinal.");

    /// <summary>Forces the query-tiled F32 SDPA path on CUDA and Vulkan instead of materializing the full score matrix.</summary>
    public static readonly Knob<bool> SdpaForceTiled =
        Bool("vram.sdpaForceTiled", "HARTSY_SDPA_FORCE_TILED", false, KnobScope.Runtime, KnobDomain.Vram, "Forces the query-tiled F32 SDPA path on CUDA and Vulkan instead of materializing the full score matrix.");

    /// <summary>=1 pages the step cache's cross-step residual and indicator snapshot to host memory as they are produced.</summary>
    public static readonly Knob<bool> StepCacheOffload =
        Bool("vram.stepCacheOffload", "HARTSY_STEP_CACHE_OFFLOAD", false, KnobScope.Runtime, KnobDomain.Vram, "=1 pages the step cache's cross-step residual and indicator snapshot to host memory as they are produced.");

    /// <summary>Registers block-streaming host weight sources as pinned memory so H2D uploads overlap compute.</summary>
    public static readonly Knob<bool> StreamPin =
        Bool("vram.streamPin", "HARTSY_STREAM_PIN", true, KnobScope.Runtime, KnobDomain.Vram, "Registers block-streaming host weight sources as pinned memory so H2D uploads overlap compute.");

    /// <summary>Kill-switch for the VAE decoder's full-resolution direct-decode attempt; 0 forces always-tiled decoding.</summary>
    public static readonly Knob<bool> VaeFullres =
        Bool("vram.vaeFullres", "HARTSY_VAE_FULLRES", true, KnobScope.Runtime, KnobDomain.Vram, "Kill-switch for the VAE decoder's full-resolution direct-decode attempt; 0 forces always-tiled decoding.");

    /// <summary>Weights stay resident on the device between generations. The <c>Auto</c> tier's fallback answer.</summary>
    public static readonly Knob<bool> KeepModels =
        Bool("vram.keepModels", "HARTSY_KEEP_MODELS", true, KnobScope.Runtime, KnobDomain.Vram,
            "Weights stay resident on the device between generations.");

    /// <summary>Streaming/eviction posture when nothing else specifies one: <c>auto</c>, <c>on</c>, or <c>off</c>.</summary>
    /// <remarks>Kept as a string because it is a three-state word, and <c>LowVramMode.Parse</c> owns the spelling
    /// table plus its warn-on-garbage log line.</remarks>
    public static readonly Knob<string?> LowVram =
        Str("vram.lowVram", "HARTSY_LOWVRAM", null, KnobScope.Runtime, KnobDomain.Vram,
            "Streaming and eviction posture when nothing else specifies one: auto, on, or off.");
    /// <summary>Across-step feature-cache drift threshold: unset/0 off, 1 the calibrated profile, or a non-negative float.</summary>
    /// <remarks>The step-cache family stays <c>string?</c> because <c>StepCacheEnv</c> owns a three-way grammar and
    /// THROWS on a malformed value by design — silently ignoring a mistyped perf knob would invalidate an A/B run.</remarks>
    public static readonly Knob<string?> StepCache =
        Str("vram.stepCache", "HARTSY_STEP_CACHE", null, KnobScope.Runtime, KnobDomain.Vram,
            "Across-step feature-cache drift threshold: unset/0 off, 1 the calibrated profile, or a non-negative float.");

    /// <summary>Max consecutive cached steps; default 3. Must be a positive integer.</summary>
    public static readonly Knob<string?> StepCacheCap =
        Str("vram.stepCacheCap", "HARTSY_STEP_CACHE_CAP", null, KnobScope.Runtime, KnobDomain.Vram,
            "Maximum consecutive cached steps; default 3.");

    /// <summary>TeaCache-style gate calibration polynomial, lowest power first, comma separated.</summary>
    public static readonly Knob<string?> StepCachePoly =
        Str("vram.stepCachePoly", "HARTSY_STEP_CACHE_POLY", null, KnobScope.Runtime, KnobDomain.Vram,
            "TeaCache-style gate calibration polynomial, lowest power first, comma separated.");

    /// <summary>Fraction of the schedule measured from the END where cache reuse is allowed; 0 means the whole schedule.</summary>
    public static readonly Knob<string?> StepCacheLate =
        Str("vram.stepCacheLate", "HARTSY_STEP_CACHE_LATE", null, KnobScope.Runtime, KnobDomain.Vram,
            "Fraction of the schedule from the end where cache reuse is allowed; 0 is the whole schedule.");

    /// <summary>CSV path to log indicator/residual-drift pairs to, for fitting the gate polynomial.</summary>
    public static readonly Knob<string?> StepCacheCalib =
        Str("vram.stepCacheCalib", "HARTSY_STEP_CACHE_CALIB", null, KnobScope.Runtime, KnobDomain.Vram,
            "CSV path to log indicator and residual-drift pairs to, for fitting the gate polynomial.");

    /// <summary>Wan-Animate-2 driving-cache dtype: on/1 = BF16, off/0 = F32, auto/unset decides from measured VRAM.</summary>
    /// <remarks>Kept as a string: the policy owns an on/off/auto spelling table plus its own dedup logging, and
    /// "on" here means BF16 — a precision trade, not a streaming one.</remarks>
    public static readonly Knob<string?> Animate2Bf16DrivingCache =
        Str("vram.animate2Bf16DrivingCache", "HARTSY_ANIMATE2_BF16_DRIVING_CACHE", null, KnobScope.Runtime, KnobDomain.Vram,
            "Wan-Animate-2 driving-cache dtype: on = BF16, off = F32, auto decides from measured free VRAM.");
}

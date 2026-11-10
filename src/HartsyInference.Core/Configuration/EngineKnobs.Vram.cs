namespace HartsyInference.Core.Configuration;

/// <summary>Residency, streaming, chunking and cache precision.</summary>
/// <remarks>Generated from the pre-migration call sites; defaults and grammars are those the code already had.</remarks>
public static partial class EngineKnobs
{
    /// <summary>Longest audio, in seconds, a windowed speech model transcribes in one request; longer input is refused.</summary>
    public static readonly Knob<int> TranscribeMaxSeconds =
        Int("vram.transcribeMaxSeconds", 7200, KnobScope.Runtime, KnobDomain.Vram,
            "Longest audio in seconds a windowed speech model transcribes in one request; longer input is refused.",
            v => v > 0 ? v : 7200);

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
    /// <remarks>The published opt-in, unchanged: <c>true</c> forces F16 for every model type that builds its cache through the shared factories. Text models do
    /// not need it any more, see <see cref="TextKvF16"/>.</remarks>
    public static readonly Knob<bool> KvF16 =
        Bool("vram.kvF16", false, KnobScope.Runtime, KnobDomain.Vram, "Stores the single-sequence decode KV cache as F16 instead of F32, halving its VRAM (CUDA only).");

    /// <summary>The KV cache element type of text LLMs: unset = F16 on a backend that supports it (CUDA), F32 elsewhere.</summary>
    /// <remarks>Unset is automatic: text LLMs on a backend that reports <see cref="HartsyInference.Core.Backends.IBackend.SupportsF16KvCache"/> use an F16 cache, which halves
    /// the cache and the bytes every decode step reads. <c>false</c> keeps F32 for text models (the <c>reference</c> profile pins it), <c>true</c> asks for F16 wherever the
    /// backend can. Audio decoders and other cache users follow <see cref="KvF16"/> only. Writes into an F16 cache saturate at the F16 maximum, so an extreme key
    /// or value cannot become infinity. The paged KV pool of the opt-in continuous-batching scheduler stays F32.</remarks>
    public static readonly Knob<bool?> TextKvF16 =
        BoolOverride("vram.textKvF16", KnobScope.Runtime, KnobDomain.Vram, "KV cache element type of text LLMs: unset = F16 on CUDA and F32 elsewhere, false = F32, true = F16 where the backend supports it. vram.kvF16 forces F16 for every model type.");

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

    /// <summary>Latent frames the SANA-WM streaming refiner's attention window holds, the seed frame included.</summary>
    /// <remarks>11 is upstream's <c>--refiner_kv_max_frames</c> and the trained setup. The window's keys and values are the
    /// refiner's largest carried state (about 0.69 GB per frame at 720p in F16), so a smaller window is the first memory
    /// lever; below 4 (the seed frame plus one chunk) the decoded video flickers at chunk boundaries.</remarks>
    public static readonly Knob<int> SanaWmRefinerWindowFrames =
        Int("vram.sanaWmRefinerWindowFrames", 11, KnobScope.Runtime, KnobDomain.Vram,
            "Latent frames the SANA-WM streaming refiner's attention window holds, the seed frame included (default 11, upstream's kv_max_frames).",
            v => v >= 1 ? v : 11);

    /// <summary>Keeps the SANA-WM streaming refiner's sliding history on every N-th transformer block only (and the last).</summary>
    /// <remarks>1 keeps it on every block, as trained. A larger stride divides the history's memory by about that much and is
    /// an approximation (upstream <c>SANA_WM_REFINER_HISTORY_LAYER_STRIDE</c>).</remarks>
    public static readonly Knob<int> SanaWmRefinerLayerStride =
        Int("vram.sanaWmRefinerLayerStride", 1, KnobScope.Runtime, KnobDomain.Vram,
            "Keeps the SANA-WM streaming refiner's sliding history on every N-th transformer block only (default 1: every block).",
            v => v >= 1 ? v : 1);

    /// <summary>How a SANA-WM stream holds its softmax windows and key tails between chunks: <c>auto</c>, <c>f32</c> or <c>bf16</c>.</summary>
    /// <remarks>BF16 is upstream's storage and halves those states (about 0.9 GB at 720p) but widens them again on every pass.
    /// <c>auto</c> (the default) uses BF16 on a device of 16 GiB or less, where F32 runs out of memory decoding chunk 1, and
    /// F32 on a larger one.</remarks>
    public static readonly Knob<string?> SanaWmStreamState =
        Str("vram.sanaWmStreamState", null, KnobScope.Runtime, KnobDomain.Vram,
            "How a SANA-WM stream holds its softmax windows and key tails between chunks: auto (default: BF16 on a device of 16 GiB or less, else F32), f32 or bf16.");

    /// <summary>Device memory in MB the SANA-WM streaming refiner's block weights leave free beyond its own attention window.</summary>
    /// <remarks>Covers the refiner's activations and the per-chunk transients of the DiT and the VAE decoder that share the
    /// device with it. Blocks that do not fit beside it are streamed from the host on every forward.</remarks>
    public static readonly Knob<long> SanaWmRefinerHeadroomMb =
        Long("vram.sanaWmRefinerHeadroomMb", 4096L, KnobScope.Runtime, KnobDomain.Vram, "Device memory in MB the SANA-WM streaming refiner's block weights leave free beyond its attention window: its activations and the DiT's and VAE decoder's per-chunk transients.");

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

    /// <summary>Lets a resident NVFP4 weight keep its unpacked dense copy in the weight-cast cache; off by default, so it unpacks per GEMM.</summary>
    public static readonly Knob<bool> ResidentCastCache =
        Bool("vram.residentCastCache", false, KnobScope.Construction, KnobDomain.Vram, "=1 lets a resident NVFP4 weight keep its unpacked dense copy in the weight-cast cache, trading VRAM for a few percent of step time; off by default.");

    /// <summary>Forces the query-tiled F32 SDPA path on CUDA and Vulkan (CUDA BF16 calls run it on their F32 casts)
    /// instead of materializing the full score matrix.</summary>
    public static readonly Knob<bool> SdpaForceTiled =
        Bool("vram.sdpaForceTiled", false, KnobScope.Runtime, KnobDomain.Vram, "Forces the query-tiled F32 SDPA path on CUDA and Vulkan (CUDA BF16 calls run it on their F32 casts) instead of materializing the full score matrix.");

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

    /// <summary>How a text model is placed when the request does not say: <c>auto</c> (the default: one GPU, else a layer split
    /// across GPUs, else expert offload), <c>gpu</c>, <c>split</c> or <c>offload</c>.</summary>
    /// <remarks>A string because <c>TextPlacementModes.Parse</c> owns the spelling and refuses a misspelled value.</remarks>
    public static readonly Knob<string?> TextPlacement =
        Str("vram.textPlacement", null, KnobScope.Construction, KnobDomain.Vram,
            "How a text model is placed when the request does not say: auto, gpu, split or offload.");

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

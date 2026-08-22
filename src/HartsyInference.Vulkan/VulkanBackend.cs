using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

/// <summary>Vulkan compute backend implementing <see cref="IBackend"/> via SPIR-V compute shaders.</summary>
// Mirrors the CUDA backend's GPU weight cache + lazy-sync activation cache so model code that works
// on CUDA works unchanged here.
public sealed class VulkanBackend : IBackend
{
    private readonly VulkanInstance _instance;
    private readonly VulkanDevice _vkDevice;
    private readonly VulkanMemoryAllocator _allocator;
    private readonly VulkanCommandStream _stream;
    private readonly VulkanDescriptorManager _descriptors;
    private readonly VulkanPipelineCache _pipelineCache;
    private readonly VulkanKernelRegistry _kernels;
    private readonly VulkanGpuTransferHelper _xfer;
    private readonly VulkanProfiler _profiler = new();
    private readonly string _spvDir;
    private bool _disposed;

    // ── GEMM fast-path engagement counters ───────────────────────────────────────────────────────────────
    // Answers "is coopmat actually engaging?" directly instead of inferring it from dispatch counts, which
    // can't distinguish a 1-dispatch coopmat launch from a 1-dispatch matmul_tiled fallback. Always counted
    // (not gated on _profiler.IsEnabled — a `long` increment is free) so GemmEngagementCounts is accurate
    // for tests without needing HARTSYINFERENCE_VK_PROFILE=1; the stderr summary in Dispose() stays
    // profiler-gated since THAT'S a logging concern, not a tracking one.
    private long _coopmatGemmCount;
    private long _coopmat2GemmCount;
    private long _tiledGemmCount;
    // Host-wall-clock ticks (Stopwatch.GetTimestamp() units) spent inside each dispatch path's own call —
    // NOT GPU execution time (see MeasureGpuTimeMs for that) but the thing that actually determines real
    // wall-clock: per-call host overhead (buffer acquisition, spec-constant/push-constant construction,
    // kernel lookup, command recording) that GPU-only timing structurally can't see. Gated on
    // _profiler.IsEnabled like the rest of the op-timing machinery (Stopwatch.GetTimestamp() has a real,
    // if small, cost — unlike the count-only engagement counters above, which are always-on).
    private long _coopmat2GemmTicks;
    private long _coopmatGemmTicks;
    private long _tiledGemmTicks;

    /// <summary>Test/diagnostic accessor for the GEMM fast-path engagement counters — lets a test prove WHICH path actually handled a dispatch (coopmat2 vs coopmat1 vs tiled fallback), since correctness alone can't distinguish them (all three produce the same numeric result for a shape all three can handle).</summary>
    internal (long CoopMat2, long CoopMat, long Tiled) GemmEngagementCounts => (_coopmat2GemmCount, _coopmatGemmCount, _tiledGemmCount);

    /// <summary>Diagnostic pass-through to <see cref="VulkanMemoryAllocator.VkAllocateMemoryStats"/> — isolates driver allocation cost from everything else a GEMM call pays for.</summary>
    internal (long CallCount, double TotalMs) VkAllocateMemoryStats => _allocator.VkAllocateMemoryStats;

    /// <summary>Diagnostic pass-through to <see cref="VulkanMemoryAllocator.SnapshotBlocks"/>.</summary>
    internal IReadOnlyList<(bool IsDedicated, int LiveAllocations, ulong Size)> AllocatorBlockSnapshot() => _allocator.SnapshotBlocks();

    // ── Step-graph capture (Phase 6e/7; see VulkanStepGraph's doc comment for the full design) ──────────────
    private VulkanStepGraph? _stepGraph;
    private bool _capturingStepGraph;

    public DeviceKind Device { get; }
    public BackendCapabilities Capabilities { get; }
    public VulkanCapabilities Vk => _vkDevice.Capabilities;

    /// <summary>Count of lazy D2H syncs since <see cref="ResetD2hSyncCount"/>; mirrors <c>CudaBackend</c>'s counter of the same name — ~0 means the traced region stayed GPU-resident.</summary>
    public long GetD2hSyncCount() => _xfer.GetSyncCount();

    /// <summary>Resets the D2H sync counter.</summary>
    public void ResetD2hSyncCount() => _xfer.ResetSyncCount();

    /// <summary>Weight/activation transfer-cache hit and miss counts since backend construction (a miss is a fresh H2D upload — the other half of a residency break that <see cref="GetD2hSyncCount"/> alone doesn't show, since a CPU-loop-default <c>IBackend</c> member that reads a GPU-resident tensor pays a D2H sync going in AND forces an H2D re-upload the next time a GPU op needs that tensor back). Cumulative since construction, not reset by <see cref="ResetD2hSyncCount"/> — diff two calls around the region you're measuring.</summary>
    public (long hits, long misses) GetTransferCacheStats()
    {
        (_, long hits, long misses) = _xfer.GetStats();
        return (hits, misses);
    }

    /// <summary>Filesystem path of the on-disk SPIR-V pipeline cache; exposed for persist/reload tests.</summary>
    public string PipelineCachePath => _pipelineCache.CachePath;

    /// <summary>Diagnostic snapshot of device-memory usage, aggregated across all DEVICE_LOCAL heaps.</summary>
    // Used by the leak-validation tests to assert that VRAM returns to baseline after a generation loop.
    // Values are stable across slab boundaries (slab-internal free regions are subtracted from UsedDeviceBytes).
    public (long usedDeviceBytes, long reservedDeviceBytes, int slabBlocks, long cachedTensorBytes) MemoryStats
    {
        get
        {
            ulong used = 0, reserved = 0;
            for (uint h = 0; h < _vkDevice.MemoryProperties.memoryHeapCount; h++)
            {
                VkMemoryHeap heap = _vkDevice.MemoryProperties.GetMemoryHeap((int)h);
                if ((heap.flags & VkMemoryHeapFlags.DeviceLocal) == 0) continue;
                used += _allocator.UsedBytes(h);
                reserved += _allocator.ReservedBytes(h);
            }
            (long cachedBytes, _, _) = _xfer.GetStats();
            return ((long)used, (long)reserved, _allocator.BlockCount, cachedBytes);
        }
    }

    /// <summary>Creates a Vulkan backend on the best discrete GPU. Validation layers enabled if HARTSYINFERENCE_VK_VALIDATION=1.</summary>
    public VulkanBackend(int deviceOrdinal = 0, string? spvDir = null)
    {
        _instance = new VulkanInstance();
        _vkDevice = VulkanDevice.Create(_instance, deviceOrdinal);

        // Resolve SPV dir: explicit override wins, else look next to assembly
        _spvDir = spvDir ?? Path.Combine(AppContext.BaseDirectory, "Spirv");
        if (!Directory.Exists(_spvDir))
            throw new DirectoryNotFoundException($"SPIR-V directory not found: {_spvDir}");

        VkPhysicalDeviceMemoryProperties memProps = _vkDevice.MemoryProperties;
        _allocator = new VulkanMemoryAllocator(_vkDevice.Handle, in memProps);
        _stream = new VulkanCommandStream(_vkDevice.Handle, _vkDevice.ComputeQueue, Vk.ComputeQueueFamilyIndex);
        // Push descriptors (VK_KHR_push_descriptor) are *available* on most NVIDIA / AMD / Intel
        // drivers, but on NVIDIA's RTX 30xx-class hardware the pool-ring path is faster:
        // Phase-C2 measurements show push-descriptors regress Flux Schnell wall-clock by ~7%
        // (129.5 s → 139.6 s) because the per-dispatch descriptor write into the command buffer
        // costs more than the pool-flip approach. Default off; enable via HARTSYINFERENCE_VK_PUSH_DESCRIPTORS=1
        // when measuring on AMD/Intel — outcome there may differ.
        bool enablePushDescriptor = Vk.HasPushDescriptor
            && Environment.GetEnvironmentVariable("HARTSYINFERENCE_VK_PUSH_DESCRIPTORS") == "1";
        _descriptors = new VulkanDescriptorManager(_vkDevice.Handle, enablePushDescriptor: enablePushDescriptor);
        _pipelineCache = new VulkanPipelineCache(_vkDevice.Handle, Vk);
        _kernels = new VulkanKernelRegistry(_vkDevice.Handle, Vk, _pipelineCache, _descriptors, _spvDir);
        _xfer = new VulkanGpuTransferHelper(_vkDevice.Handle, _allocator, in memProps, Vk, _stream);

        // Opt-in INT8 dot-product GEMM path for Linear (see TryDispatchInt8Linear). Strict "1" opt-in,
        // matching this constructor's push-descriptor switch above — an experimental switch, not a proven
        // default-on profile feature (see EnvSwitch.cs's remarks on that distinction).
        EnableInt8Linear = Environment.GetEnvironmentVariable("HARTSYINFERENCE_VK_INT8") == "1";

        // OOM retry path: when an allocation fails, force the stream to submit and wait for the
        // GPU, drain the deferred-free list, then release any fully-empty slab blocks back to the
        // device. Mirrors CudaMemory.Allocate's retry path.
        _allocator.OnOutOfMemory = () =>
        {
            try
            {
                _stream.WaitIdleHost();
                _allocator.ReleaseEmptySlabs();
            }
            catch (Exception ex)
            {
                Logs.Error("Vulkan OOM-retry callback failed while releasing empty slabs", ex);
            }
        };
        // Genuine (post-retry) OOM diagnostics: the allocator's own byte/block counters don't cover
        // weight/activation caches or step-graph-retained buffers, both of which live in the transfer
        // helper — dump the full picture so a real OOM's root cause (leak vs. a genuine peak-VRAM spike,
        // e.g. step-graph capture retaining every intermediate activation for the graph's lifetime) is
        // visible in the log instead of needing a repro + ad-hoc instrumentation every time.
        _allocator.OnOutOfMemoryDiagnostic = () =>
        {
            (long usedBytes, long reservedBytes, int blocks, _) = MemoryStats;
            return $"  allocator: used={usedBytes / (1024.0 * 1024):F1} MB, reserved={reservedBytes / (1024.0 * 1024):F1} MB, blocks={blocks}\n" +
                   _xfer.DiagnosticsSummary();
        };

        Device = DeviceKind.Vulkan(deviceOrdinal);
        Capabilities = new BackendCapabilities
        {
            Name = $"Vulkan ({Vk.DeviceName}, {Vk.VendorString}, {Vk.DeviceType})",
            SupportsF32 = true,
            SupportsF16 = Vk.SupportsFp16 && Vk.Storage16Bit,
            SupportsBF16 = false,
            SupportsQuantized = false,
            SupportsConv2D = true,
            SupportsSdpa = true,
            SupportsFft = false,
            MaxRank = 6,
        };
    }

    /// <summary>Preloads weights to GPU memory. Cached by Tensor reference.</summary>
    public void PreloadWeights(IEnumerable<Tensor> weights) => _xfer.PreloadWeights(weights);

    /// <summary>Master kill-switch for weight dtype-cast caching — mirrors <c>CudaBackend.CacheWeightCasts</c> exactly (same name, same purpose): turn off for a large FP8/quantized model whose full cast set (e.g. FP8→F32, 4x expansion) doesn't fit VRAM alongside its own raw weights, trading recompute for memory via transient (dispatched-and-freed-per-call) dequant instead of a cast cached forever. Defaults from <c>HARTSYINFERENCE_VK_NO_WEIGHT_CAST_CACHE=1</c>.</summary>
    public bool CacheWeightCasts
    {
        get => _xfer.CacheWeightCasts;
        set => _xfer.CacheWeightCasts = value;
    }

    /// <summary>Peak per-tile im2col buffer size for <see cref="Conv2D"/> (see its tiling comment). Settable so tests can force a small value and exercise the multi-tile path deterministically without needing a multi-GB conv shape; production code never needs to touch this. Default 256 MiB.</summary>
    public ulong Conv2DMaxColTileBytes { get; set; } = 256UL * 1024 * 1024;

    public void Sync()
    {
        _xfer.DrainTransients();
        _stream.WaitIdleHost();
        _dispatchesSinceSubmit = 0;
    }

    public void FreeWeights(IEnumerable<Tensor> weights) => _xfer.FreeWeights(weights);

    /// <summary>Materializes a cached activation to host and releases its device buffer, by firing the lazy sync callback <c>VulkanGpuTransferHelper.CacheActivation</c> plants. Overridden rather than left as the interface no-op because Vulkan has its own lazy activation cache: the callers of this are cross-model caches that used to spell it <c>_ = t.DataPointer</c>, and a no-op here would silently leave them device-only.</summary>
    public unsafe void OffloadActivation(Tensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        _ = tensor.DataPointer;
    }

    #region Helpers

    private const uint LocalX1D = 256;

    /// <summary>Default spec-constant set for 1D-dispatch kernels (local_x = <see cref="LocalX1D"/>, local_y = local_z = 1).</summary>
    private static readonly SpecConstant[] _default1DSpec =
    {
        SpecConstant.UInt(0, LocalX1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)
    };

    private static uint GroupCount(long total, uint localX)
        => (uint)((total + localX - 1) / localX);

    /// <summary>Resolve the dtype suffix for kernel selection: f16 if both inputs/output are F16; f32 otherwise.</summary>
    private static string DtypeSuffix(DType dt) => dt == DType.F16 ? "_f16" : "_f32";

    /// <summary>The kernel binding count (number of SSBOs) for a given shape — drives descriptor-set-layout selection.</summary>
    private VulkanKernel GetKernel(string shaderName, int storageBufferCount, ReadOnlySpan<SpecConstant> spec)
        => _kernels.Get(shaderName, storageBufferCount, spec, forCapture: _capturingStepGraph);

    /// <summary>Recorded dispatches since last submit — flushed periodically so deferred-free buffers can be reclaimed.</summary>
    // Otherwise large models accumulate hundreds of allocations between submits and exhaust the heap.
    private int _dispatchesSinceSubmit;
    /// <summary>Dispatches recorded by the current outermost op — reset at op entry, read by the profiler.</summary>
    // Kept separate from _dispatchesSinceSubmit (which resets only on an actual submit) so batching
    // submits across ops doesn't corrupt the per-op dispatch count the profiler reports.
    private int _dispatchesThisOp;
    private const int FlushThreshold = 8;

    /// <summary>When set, submits a command buffer per op instead of batching until <see cref="FlushThreshold"/> dispatches accumulate.</summary>
    // Batching cuts one vkQueueSubmit2 per tiny op — the dominant per-dispatch host cost for small-op-heavy workloads.
    private static readonly bool _submitPerOp =
        Environment.GetEnvironmentVariable("HARTSYINFERENCE_VK_SUBMIT_PER_OP") == "1";

    private unsafe void Dispatch(
        VulkanKernel kernel,
        ReadOnlySpan<ulong> bufferHandles,
        ReadOnlySpan<byte> pushConstants,
        uint groupX, uint groupY = 1, uint groupZ = 1)
    {
        // Step-graph capture: record onto the graph's own persistent command buffer instead of the normal
        // stream, and ALWAYS push-descriptor-bind (regardless of the process's default eager-mode strategy —
        // see VulkanStepGraph's doc comment for why a pool-allocated set is unsafe for a replayed buffer).
        // kernel itself is already the push-descriptor-flavored variant here: GetKernel passes
        // forCapture: _capturingStepGraph through to the registry, which builds against the matching layout.
        if (_capturingStepGraph)
        {
            nint capCb = _stepGraph!.RecordingBuffer;
            VulkanApi.vkCmdBindPipeline(capCb, VkPipelineBindPoint.Compute, kernel.Pipeline);
            _descriptors.PushSet(capCb, kernel.PipelineLayout, bufferHandles);
            if (pushConstants.Length > 0)
            {
                fixed (byte* p = pushConstants)
                    VulkanApi.vkCmdPushConstants(capCb, kernel.PipelineLayout, VkShaderStageFlags.Compute, 0, (uint)pushConstants.Length, (nint)p);
            }
            VulkanApi.vkCmdDispatch(capCb, groupX, groupY, groupZ);
            VulkanCommandStream.RecordGlobalComputeBarrierOn(capCb);
            // No dispatch-count/flush bookkeeping: this buffer isn't part of the normal stream's tick/submit
            // lifecycle at all — it's ended and submitted explicitly by StepGraphEndAndLaunch/Launch.
            return;
        }

        nint cb = _stream.AcquireRecording();
        VulkanApi.vkCmdBindPipeline(cb, VkPipelineBindPoint.Compute, kernel.Pipeline);

        ulong layout = kernel.PipelineLayout;
        if (_descriptors.PushDescriptorActive)
        {
            // Fast path: write descriptor bindings directly into the command buffer. Skips the
            // pool allocation + vkUpdateDescriptorSets + vkCmdBindDescriptorSets sequence.
            _descriptors.PushSet(cb, layout, bufferHandles);
        }
        else
        {
            ulong setLayout = kernel.DescriptorSetLayout;
            ulong dstSet = _descriptors.AllocateSet(setLayout);
            _descriptors.WriteSet(dstSet, bufferHandles);
            VulkanApi.vkCmdBindDescriptorSets(cb, VkPipelineBindPoint.Compute, layout,
                0, 1, (nint)(&dstSet), 0, 0);
        }

        if (pushConstants.Length > 0)
        {
            fixed (byte* p = pushConstants)
                VulkanApi.vkCmdPushConstants(cb, layout, VkShaderStageFlags.Compute, 0, (uint)pushConstants.Length, (nint)p);
        }

        VulkanApi.vkCmdDispatch(cb, groupX, groupY, groupZ);

        // Conservative: emit a global compute->compute barrier so the next dispatch sees this output.
        _stream.RecordGlobalComputeBarrier();

        _dispatchesSinceSubmit++;
        _dispatchesThisOp++;
        // CRITICAL: do NOT drain transients here. Multi-dispatch ops like SDPA (24 heads × 3
        // dispatches) reference the same Q/K/V upload buffers across many dispatches. Draining
        // mid-op would tag those buffers for deferred-free, then the next flush completes them
        // and the buffers get vkDestroy'd while later heads' descriptor sets still reference
        // them — silent garbage output for heads beyond the first ~2.
        // Transient drain is now done explicitly at op boundaries via DrainAndFlush.
        if (_dispatchesSinceSubmit >= FlushThreshold && _opNestingDepth == 0)
        {
            DrainAndFlush();
        }
    }

    /// <summary>Tracks whether we're currently inside a backend op; while >0, auto-flush is suppressed.</summary>
    // Draining transients mid-op would free buffers still referenced by later dispatches in the same
    // op (e.g. SDPA's per-head loop sharing Q/K/V uploads).
    private int _opNestingDepth;

    private OpScope EnterOp([CallerMemberName] string opName = "")
        => new(this, opName);
    private readonly struct OpScope : IDisposable
    {
        private readonly VulkanBackend _b;
        private readonly string _opName;
        private readonly long _startTicks;

        public OpScope(VulkanBackend b, string opName)
        {
            _b = b;
            _opName = opName;
            _startTicks = b._profiler.IsEnabled ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            if (b._opNestingDepth == 0) b._dispatchesThisOp = 0;
            b._opNestingDepth++;
        }

        public void Dispose()
        {
            _b._opNestingDepth--;
            if (_b._opNestingDepth == 0)
            {
                int dispatches = _b._dispatchesThisOp;
                // Step-graph capture: every dispatch this op issued was recorded onto the capture buffer, not
                // _stream (see VulkanBackend.Dispatch) — _stream has nothing pending, so a submit here would
                // be at best a wasted no-op and at worst an assumption this comment exists to keep from ever
                // becoming false. DrainTransients() is capture-safe on its own (redirects to the retain list),
                // but skip the whole block explicitly rather than relying on that alone.
                if (_b._capturingStepGraph) return;
                // Tag this op's transient uploads for deferred-free at the op boundary (safe — every
                // dispatch that referenced them is already recorded). Submitting, however, is batched:
                // only flush the command buffer once enough dispatches accumulate, so a stream of tiny
                // ops costs one vkQueueSubmit2 instead of one each. Sync()/lazy-sync still force a flush.
                _b._xfer.DrainTransients();
                if (_submitPerOp || _b._dispatchesSinceSubmit >= FlushThreshold)
                {
                    _b._stream.SubmitAndAdvance();
                    _b._dispatchesSinceSubmit = 0;
                }
                if (_b._profiler.IsEnabled)
                    _b._profiler.Record(_opName, System.Diagnostics.Stopwatch.GetTimestamp() - _startTicks, dispatches);
            }
        }
    }

    private void DrainAndFlush()
    {
        // Tag transient buffers at the upcoming tick (DrainTransients calls _stream.DeferredFree,
        // which uses _value + 1 — see VulkanCommandStream.DeferredFree).
        _xfer.DrainTransients();
        // Submit but don't host-wait. The GPU will complete in the background; deferred-free
        // reclamation happens opportunistically inside SubmitAndAdvance once the GPU passes the
        // tagged tick. Activation-cache reads from CPU pull a host wait inside the lazy-sync
        // callback, and explicit Sync() callers still get a synchronous WaitIdleHost. This
        // eliminates ~2800 host waits per Flux Schnell 4-step generation and lets the driver
        // overlap submission of op N+1 with execution of op N.
        _stream.SubmitAndAdvance();
        _dispatchesSinceSubmit = 0;
    }

    /// <summary>Promotes a freshly-allocated GPU buffer to the activation cache, keyed by Tensor reference equality.</summary>
    private void CacheOutput(Tensor output, VulkanBuffer buffer) => _xfer.CacheActivation(output, buffer);

    private VulkanBuffer GetBuffer(Tensor t) => _xfer.CopyToDevice(t);

    /// <summary>Selects the (BM, BN, BK, TM, TN) tile size for the tiled matmul kernel based on the GEMM shape.</summary>
    // Profile-driven choice (Flux Schnell, RTX 3060): big tiles 128x128/32 for Flux/SDXL Linear shapes
    // (M, N >= 128) cut workgroup count by ~16x vs the old 32x32 default — the primary perf win from
    // Phase C2 step 1. Smaller tiles are picked for shapes where big tiles would leave most threads idle
    // (e.g. SDPA per-head 64x64 matmuls). MAX_BM/BN/BK/TM/TN in the shader (matmul_tiled.comp.glsl) cap
    // what's selectable here — keep them in sync if you bump these values.
    private (uint BM, uint BN, uint BK, uint TM, uint TN) PickMatmulTile(long M, long N, long K)
    {
        // matmul_tiled's shared arrays are FP32 and sized BM*(BK+1) + BK*BN (see the shader). The 128
        // tile needs 33,280 B — over the 32 KB Vulkan-spec minimum — so on AMD GCN / Intel-class devices
        // we must fall to a tile that fits, or pipeline creation is invalid. NVIDIA (48–64 KB) keeps 128.
        uint shmem = Vk.MaxComputeSharedMemoryBytes;
        if (M >= 128 && N >= 128 && TileSharedBytes(128, 128, 32) <= shmem) return (128, 128, 32, 8, 8);
        if (M >= 64  && N >= 64  && TileSharedBytes(64, 64, 16) <= shmem)   return ( 64,  64, 16, 8, 8);
        return (32, 32, 16, 4, 4);   // (32*17 + 16*32)*4 = 4,224 B — fits even a 16 KB shared-mem floor
    }

    /// <summary>Shared-memory bytes matmul_tiled's FP32 Asub+Bsub consume for a tile (must stay ≤ device limit).</summary>
    private static uint TileSharedBytes(uint BM, uint BN, uint BK) => (BM * (BK + 1) + BK * BN) * sizeof(float);

    /// <summary>Env override to force-disable the cooperative-matrix fast path (perf comparison / debugging).</summary>
    private static readonly bool _disableCoopmat =
        Environment.GetEnvironmentVariable("HARTSYINFERENCE_VK_DISABLE_COOPMAT") == "1";

    /// <summary>Attempts the cooperative-matrix matmul fast path; returns true when the kernel was dispatched (op fully handled).</summary>
    // Returns false when the path doesn't apply and the caller should fall through to matmul_tiled. Constraints
    // for v1: (a) HasCooperativeMatrix capability, (b) GEMM dtype is FP16, (c) N, K multiples of FRAG=16 (see
    // below for M), (d) no fused activation. Bias is handled via a follow-up BroadcastAdd(N, 1) dispatch — see
    // comments below.
    //
    // M does NOT need to be a multiple of 16 when transposeA is false: matmul_coopmat_partial_m.comp.glsl (a
    // separate shader file, not a branch in matmul_coopmat.comp.glsl — see its doc comment) handles a
    // non-aligned M via bounds-checked shared-memory staging, mirroring matmul_tiled.comp.glsl's own proven
    // idiom. This is the SECOND attempt at this problem (2026-07-31): the first (a host-side scratch-buffer +
    // device-to-device copy) passed every test thrown at it but caused a real `ErrorDeviceLost` on a full
    // Krea2 run and was reverted — see docs/Checklists/TROUBLESHOOTING.md. This shared-memory design avoids
    // that whole risk class (no separate command buffer, no cross-submission barrier — everything happens
    // inside one dispatch). Scoped to transposeA=false because that's the only case Linear (the real caller
    // this exists for) ever uses and the only case tested; transposeB may be either.
    private bool TryDispatchCoopmat(
        Tensor output, VulkanBuffer aRes, VulkanBuffer bRes,
        int M, int N, int K, bool transposeA, bool transposeB, DType gemmDtype,
        VulkanBuffer outBuf, Tensor? bias, VulkanBuffer? biasRes, VulkanBuffer? biasRaw)
    {
        const uint FRAG = 16;
        if (_disableCoopmat) return false;
        if (!Vk.HasCooperativeMatrix) return false;
        if (gemmDtype != DType.F16) return false;
        bool mAligned = (M % FRAG) == 0;
        if ((!mAligned && transposeA) || (N % FRAG) != 0 || (K % FRAG) != 0) return false;
        // Output must be FP16 or FP32 — coopmat shader supports both via OUTPUT_F32 spec const.
        // Other output dtypes (BF16, FP8, etc.) fall through to the tiled path.
        bool outputIsF32 = output.DType == DType.F32;
        if (output.DType != DType.F16 && !outputIsF32) return false;

        // BM/BN: 64×64 covers Flux Linear shapes well (1280×3072×3072 → 20×48 = 960 wgs).
        // Drop to 32×32 when M or N is exactly 16 or 32 (typical of small attention heads).
        uint BM = (M >= 64) ? 64u : (M >= 32 ? 32u : 16u);
        uint BN = (N >= 64) ? 64u : (N >= 32 ? 32u : 16u);
        // SUBGROUP_SIZE matches what we pin via VkPipelineShaderStageRequiredSubgroupSizeCreateInfo.
        uint subgroupSize = Vk.SubgroupSize;
        // Workgroup invocations = (BM/16)*(BN/16) * subgroupSize. On NVIDIA (subgroup 32) a 64×64 tile is
        // 16*32 = 512 — fine. On AMD wave64 it's 16*64 = 1024, and on small devices that can exceed
        // maxComputeWorkGroupInvocations / sizeX. Shrink the tile (never below the 16×16 fragment, which is
        // always one subgroup ≤ the limit) so the dispatch stays valid cross-vendor. No-op on NVIDIA.
        // The partial-M kernel ALSO needs its BM*FRAG_K*2 + BM*BN*4 byte shared-memory scratch (sA + sC —
        // see matmul_coopmat_partial_m.comp.glsl) to fit the device's budget; the aligned kernel uses no
        // shared memory at all, so this second condition is a no-op (always false) when mAligned.
        uint maxInvocations = Math.Min(Vk.MaxComputeWorkGroupInvocations, Vk.MaxComputeWorkGroupSizeX);
        while ((BM > FRAG || BN > FRAG) &&
               ((BM / FRAG) * (BN / FRAG) * subgroupSize > maxInvocations ||
                (!mAligned && (BM * FRAG * 2 + BM * BN * 4) > Vk.MaxComputeSharedMemoryBytes)))
        {
            if (BN >= BM && BN > FRAG) BN /= 2;
            else if (BM > FRAG) BM /= 2;
            else BN /= 2;
        }
        // subgroups per workgroup = (BM/16) * (BN/16). Workgroup size = subgroups * subgroupSize.
        uint subgroups = (BM / FRAG) * (BN / FRAG);
        uint localX = subgroups * subgroupSize;

        try
        {
            string shader = mAligned ? "matmul_coopmat" : "matmul_coopmat_partial_m";
            ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
            {
                SpecConstant.UInt(0, localX),
                SpecConstant.UInt(1, 1),
                SpecConstant.UInt(2, 1),
                SpecConstant.UInt(10, BM),
                SpecConstant.UInt(11, BN),
                SpecConstant.UInt(12, subgroupSize),
                SpecConstant.Bool(13, transposeA),
                SpecConstant.Bool(14, transposeB),
                SpecConstant.Bool(15, outputIsF32),
                SpecConstant.Bool(16, bias is not null),
            };

            VulkanKernel k = GetKernel(shader, storageBufferCount: 5, spec);

            Span<byte> pc = stackalloc byte[11 * 4];
            BinaryWriteUInt(pc, 0, (uint)M);
            BinaryWriteUInt(pc, 4, (uint)N);
            BinaryWriteUInt(pc, 8, (uint)K);
            BinaryWriteUInt(pc, 12, (uint)(transposeA ? M : K));
            BinaryWriteUInt(pc, 16, (uint)(transposeB ? K : N));
            BinaryWriteUInt(pc, 20, (uint)N);
            BinaryWriteFloat(pc, 24, 1.0f);
            BinaryWriteFloat(pc, 28, 0.0f);
            BinaryWriteUInt(pc, 32, 0u);
            BinaryWriteUInt(pc, 36, 0u);
            BinaryWriteUInt(pc, 40, 0u);

            // Five descriptor bindings: 0=A, 1=B, 2=C_fp16, 3=Bias(FP32), 4=C_fp32.
            // The shader writes slot 2 (fp16) OR slot 4 (fp32) selected by the OUTPUT_F32 spec
            // constant; both point at the single real output buffer (allocated in output.DType,
            // so exactly one binding's type matches and is written). The unwritten output slot
            // binds to outBuf as a placeholder. Slot 3 carries the per-column bias as FP32 when
            // HAS_BIAS — the shader adds it in the epilogue (fused, no extra dispatch). The bias is
            // cast to FP32 once and cached (preloaded weight), matching the coopmat accumulator type.
            ulong outHandle = outBuf.Handle;
            VulkanBuffer? biasF32Owned = null;
            ulong biasHandle = outHandle;
            if (bias is not null)
            {
                (VulkanBuffer biasF32, VulkanBuffer? owned) = CastIfNeeded(bias, biasRaw!, DType.F32);
                biasF32Owned = owned;
                biasHandle = biasF32.Handle;
            }
            Span<ulong> bufs = stackalloc ulong[] { aRes.Handle, bRes.Handle, outHandle, biasHandle, outHandle };

            uint groupsX = (uint)((N + BN - 1) / BN);
            uint groupsY = (uint)((M + BM - 1) / BM);
            Dispatch(k, bufs, pc, groupsX, groupsY, 1);

            CacheOutput(output, outBuf);

            if (biasF32Owned is not null) _xfer.FreeDevice(biasF32Owned);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan TryDispatchCoopmat dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }

        return true;
    }

    /// <summary>Diagnostic-only entry point for <c>matmul_coopmat_blocked.comp.glsl</c> (2026-07-31) — NOT called from <see cref="DispatchMatmul"/> or any production path. Exists purely so correctness/throughput can be measured in isolation (unit test + GPU benchmark) before any decision to integrate register blocking into the real dispatch path. Fixed BM=BN=64, WM=WN=32 (no device-aware tile shrinking yet — this is a diagnostic, not production code); requires N, K exact multiples of 16 (M may be anything — the shader bounds-checks it the same way <c>matmul_coopmat_partial_m.comp.glsl</c> does) and F16 GEMM dtype. See docs/Checklists/TROUBLESHOOTING.md for the full writeup and benchmark results.</summary>
    internal bool TryDispatchCoopmatBlockedDiagnostic(
        Tensor output, Tensor a, Tensor b, bool transposeA, bool transposeB, Tensor? bias, uint wm = 32, uint wn = 32)
    {
        if (!Vk.HasCooperativeMatrix) return false;
        int N = transposeB ? (int)b.Shape[0] : (int)b.Shape[b.Shape.Rank - 1];
        int M = (int)(output.ElementCount / N);
        int K = transposeA ? (int)a.Shape[0] : (int)a.Shape[a.Shape.Rank - 1];
        if ((N % 16) != 0 || (K % 16) != 0) return false;
        if (output.DType != DType.F16 && output.DType != DType.F32) return false;
        bool outputIsF32 = output.DType == DType.F32;

        const uint BM = 64, BN = 64;
        uint WM = wm, WN = wn;
        uint subgroupSize = Vk.SubgroupSize;
        uint subgroupsPerWg = (BM / WM) * (BN / WN);
        uint localX = subgroupsPerWg * subgroupSize;
        ulong sharedBytes = (ulong)(BM * 32 * 2 + 32 * BN * 2 + subgroupsPerWg * 16 * 16 * 4);
        if (sharedBytes > Vk.MaxComputeSharedMemoryBytes)
            throw new InvalidOperationException($"matmul_coopmat_blocked needs {sharedBytes} B shared memory, device has {Vk.MaxComputeSharedMemoryBytes} B.");

        VulkanBuffer aBuf = GetBuffer(a);
        VulkanBuffer bBuf = GetBuffer(b);
        (VulkanBuffer aRes, VulkanBuffer? aOwned) = CastIfNeeded(a, aBuf, DType.F16);
        (VulkanBuffer bRes, VulkanBuffer? bOwned) = CastIfNeeded(b, bBuf, DType.F16);
        VulkanBuffer? biasRaw = null, biasOwned = null;
        if (bias is not null)
        {
            biasRaw = GetBuffer(bias);
            (VulkanBuffer biasF32, VulkanBuffer? owned) = CastIfNeeded(bias, biasRaw, DType.F32);
            biasRaw = biasF32;
            biasOwned = owned;
        }

        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
            {
                SpecConstant.UInt(0, localX),
                SpecConstant.UInt(1, 1),
                SpecConstant.UInt(2, 1),
                SpecConstant.UInt(10, BM),
                SpecConstant.UInt(11, BN),
                SpecConstant.UInt(12, subgroupSize),
                SpecConstant.Bool(13, transposeA),
                SpecConstant.Bool(14, transposeB),
                SpecConstant.Bool(15, outputIsF32),
                SpecConstant.Bool(16, bias is not null),
                SpecConstant.UInt(17, WM),
                SpecConstant.UInt(18, WN),
            };
            VulkanKernel k = GetKernel("matmul_coopmat_blocked", storageBufferCount: 5, spec);

            Span<byte> pc = stackalloc byte[11 * 4];
            BinaryWriteUInt(pc, 0, (uint)M);
            BinaryWriteUInt(pc, 4, (uint)N);
            BinaryWriteUInt(pc, 8, (uint)K);
            BinaryWriteUInt(pc, 12, (uint)(transposeA ? M : K));
            BinaryWriteUInt(pc, 16, (uint)(transposeB ? K : N));
            BinaryWriteUInt(pc, 20, (uint)N);
            BinaryWriteFloat(pc, 24, 1.0f);
            BinaryWriteFloat(pc, 28, 0.0f);
            BinaryWriteUInt(pc, 32, 0u);
            BinaryWriteUInt(pc, 36, 0u);
            BinaryWriteUInt(pc, 40, 0u);

            ulong outHandle = outBuf.Handle;
            ulong biasHandle = bias is not null ? biasRaw!.Handle : outHandle;
            Span<ulong> bufs = stackalloc ulong[] { aRes.Handle, bRes.Handle, outHandle, biasHandle, outHandle };

            uint groupsX = (uint)((N + BN - 1) / BN);
            uint groupsY = (uint)((M + BM - 1) / BM);
            Dispatch(k, bufs, pc, groupsX, groupsY, 1);

            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan TryDispatchCoopmatBlockedDiagnostic dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
        finally
        {
            if (aOwned is not null) _xfer.FreeDevice(aOwned);
            if (bOwned is not null) _xfer.FreeDevice(bOwned);
            if (biasOwned is not null) _xfer.FreeDevice(biasOwned);
        }
        return true;
    }

    /// <summary>Switch for <see cref="TryDispatchCoopMat2"/> being tried (before <see cref="TryDispatchCoopmat"/>) from <see cref="DispatchMatmul"/>. Default-ON as of 2026-07-31 (settable/overridable via <c>HARTSYINFERENCE_VK_COOPMAT2=0</c>) after a full validation pass against real Krea2 weights on the RTX 4090: correctness held byte-identical to the coopmat1 baseline across every configuration tested — 2-step, 4-step, and 8-step generations (the exact step count at which an EARLIER, unrelated coopmat1 M-padding fix passed every synthetic test and then hit a real `ErrorDeviceLost` — see docs/Checklists/TROUBLESHOOTING.md), aligned and non-16-aligned M, with and without bias, F16 and F32 output, across dozens of real generations. Performance: a controlled same-session comparison (4 runs each config, alternating, isolating this box's real GPU-contention variance) showed a statistically significant ~4% real-wall-clock win at 2 steps (Welch's t≈2.88) and a ~10% win at 8 steps — this SUPERSEDES an earlier same-day finding of a "~5% regression," which turned out to be a cross-session comparison artifact (this shared box's run-to-run variance from contending processes, not a real property of coopmat2 — see the full writeup for how that got sorted out). The only failure mode found anywhere in this investigation (a step-graph-capture VRAM peak causing a graceful OOM-and-fallback on this box at 4+ steps) is root-caused, unrelated to coopmat2 specifically (reproduces identically with this flag off), and already recovers correctly via the existing capture-fallback path.</summary>
    public bool EnableCoopMat2 { get; set; } = Environment.GetEnvironmentVariable("HARTSYINFERENCE_VK_COOPMAT2") != "0";

    /// <summary>Cooperative-matrix-2 fast path for <see cref="DispatchMatmul"/>, tried before <see cref="TryDispatchCoopmat"/> when <see cref="EnableCoopMat2"/> is set. Built on <c>matmul_coopmat2.comp.glsl</c> (<c>VK_NV_cooperative_matrix2</c> — workgroup-scope, tensor-layout-addressed, hardware-clamped GEMM; see docs/Checklists/TROUBLESHOOTING.md for the full coopmat1-vs-coopmat2 investigation this came out of). Unlike <see cref="TryDispatchCoopmat"/>, this has NO M/N/K alignment requirement at all — the hardware clamp mode zero-fills/drops out-of-bounds tensor accesses. Specialized for transposeA=false, transposeB=true (the only combination <see cref="Linear"/> — the real caller this exists for — ever uses) — throws for any other combination rather than silently computing the wrong answer. Bias is fused directly into the shader via a broadcast tensorLayoutNV (2026-07-31 revision) — an earlier follow-up-BroadcastAdd-dispatch design measured FASTER in isolated GPU-only-time benchmarks but SLOWER in a real Krea2 e2e run: the extra dispatch's host-side submission + the unconditional per-dispatch VkMemoryBarrier2 (see ROADMAP.md's "per-dispatch barrier scoping" entry) isn't visible to VkQueryPool-timestamp-only measurement, but it's very real — see docs/Checklists/TROUBLESHOOTING.md for the full writeup.</summary>
    internal bool TryDispatchCoopMat2(
        Tensor output, Tensor a, Tensor b, bool transposeA, bool transposeB, Tensor? bias, uint bk = 0)
    {
        if (!Vk.HasCooperativeMatrix2) return false;
        if (transposeA || !transposeB)
            throw new NotSupportedException("TryDispatchCoopMat2 only supports transposeA=false, transposeB=true (the production Linear-layer convention).");
        if (output.DType != DType.F16 && output.DType != DType.F32) return false;
        bool outputIsF32 = output.DType == DType.F32;

        int N = (int)b.Shape[0];
        int M = (int)(output.ElementCount / N);
        int K = (int)a.Shape[a.Shape.Rank - 1];

        uint BM = Vk.CoopMat2MGranularity;
        uint BN = Vk.CoopMat2NGranularity;
        // Swept BK in {16,32,64,128,256} against real shapes (see docs/Checklists/TROUBLESHOOTING.md):
        // BK=16 (the bare K-granularity minimum) was measurably WORSE than larger values on K=3072 shapes
        // (14983us vs ~9260-9925us) — each coopMatLoadTensorNV is itself an expensive workgroup-cooperative
        // op, so more, smaller K-steps means paying that overhead more often. BK=64 was best-or-near-best
        // across both swept shapes; use it as the default when the caller doesn't override.
        uint BK = bk != 0 ? bk : 64u;
        uint localX = Vk.CoopMat2WorkgroupInvocations;

        VulkanBuffer aBuf = GetBuffer(a);
        VulkanBuffer bBuf = GetBuffer(b);
        (VulkanBuffer aRes, VulkanBuffer? aOwned) = CastIfNeeded(a, aBuf, DType.F16);
        (VulkanBuffer bRes, VulkanBuffer? bOwned) = CastIfNeeded(b, bBuf, DType.F16);

        // Bias always cast to F32, matching the accumulator's precision and matmul_coopmat.comp.glsl's own
        // convention (see that kernel's binding-3 comment) — independent of the GEMM's F16 input dtype.
        VulkanBuffer? biasOwned = null;
        VulkanBuffer? biasRaw = null;
        ulong biasHandle;
        if (bias is not null)
        {
            biasRaw = GetBuffer(bias);
            (VulkanBuffer biasF32, VulkanBuffer? owned) = CastIfNeeded(bias, biasRaw, DType.F32);
            biasOwned = owned;
            biasHandle = biasF32.Handle;
        }
        else
        {
            biasHandle = aRes.Handle;   // placeholder — HAS_BIAS=false means the shader never reads binding 4
        }

        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
            {
                SpecConstant.UInt(0, localX),
                SpecConstant.UInt(1, 1),
                SpecConstant.UInt(2, 1),
                SpecConstant.UInt(10, BM),
                SpecConstant.UInt(11, BN),
                SpecConstant.UInt(12, BK),
                SpecConstant.Bool(15, outputIsF32),
                SpecConstant.Bool(16, bias is not null),
            };
            VulkanKernel k = GetKernel("matmul_coopmat2", storageBufferCount: 5, spec);

            Span<byte> pc = stackalloc byte[9 * 4];
            BinaryWriteUInt(pc, 0, (uint)M);
            BinaryWriteUInt(pc, 4, (uint)N);
            BinaryWriteUInt(pc, 8, (uint)K);
            BinaryWriteUInt(pc, 12, (uint)K);   // lda: A is [M,K] row-major
            BinaryWriteUInt(pc, 16, (uint)K);   // ldb: B is [N,K] row-major (transposeB)
            BinaryWriteUInt(pc, 20, (uint)N);   // ldc: C is [M,N] row-major
            BinaryWriteUInt(pc, 24, 0u);
            BinaryWriteUInt(pc, 28, 0u);
            BinaryWriteUInt(pc, 32, 0u);

            ulong outHandle = outBuf.Handle;
            Span<ulong> bufs = stackalloc ulong[] { aRes.Handle, bRes.Handle, outHandle, outHandle, biasHandle };

            uint groupsX = (uint)((N + BN - 1) / BN);
            uint groupsY = (uint)((M + BM - 1) / BM);
            Dispatch(k, bufs, pc, groupsX, groupsY, 1);

            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan TryDispatchCoopMat2 dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
        finally
        {
            if (aOwned is not null) _xfer.FreeDevice(aOwned);
            if (bOwned is not null) _xfer.FreeDevice(bOwned);
            if (biasOwned is not null) _xfer.FreeDevice(biasOwned);
        }
        return true;
    }

    /// <summary>Diagnostic-only: measures PURE GPU execution time (via <see cref="VulkanGpuTimer"/>'s <c>VkQueryPool</c> timestamps) of <paramref name="iterations"/> calls to <paramref name="dispatchOne"/> — separates real kernel execution time from host-side submission/dispatch/wait overhead, which a plain <c>Stopwatch</c>-around-<c>Sync()</c> measurement cannot do (it includes both). Not used on any production path. See docs/Checklists/TROUBLESHOOTING.md for what this was built to answer.</summary>
    internal double MeasureGpuTimeMs(int iterations, Action dispatchOne)
    {
        using VulkanGpuTimer timer = new(_vkDevice.Handle, Vk.TimestampPeriod);
        nint cb = _stream.AcquireRecording();
        timer.RecordReset(cb);
        timer.RecordStart(cb);
        for (int i = 0; i < iterations; i++) dispatchOne();
        nint cbEnd = _stream.AcquireRecording();   // same buffer unless an internal auto-flush happened
        timer.RecordEnd(cbEnd);
        _stream.WaitIdleHost();   // submits + host-waits, guaranteeing RecordEnd's write has completed
        return timer.ReadElapsedMs();
    }

    /// <summary>Dequantizes a GGUF weight (Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K) to F32 via <see cref="CastIfNeeded"/>'s dequant path; test/tooling hook (mirrors <c>CudaBackend.DequantizeToF32</c>), not a hot path — the hot path dequantizes lazily inside <c>Linear</c>/<c>MatMul</c>'s own weight-cast-to-gemm-dtype call.</summary>
    public Tensor DequantizeToF32(Tensor quant)
    {
        using OpScope _op = EnterOp();
        long count = quant.ElementCount;
        Tensor output = new(new TensorShape(count), DType.F32);
        VulkanBuffer srcBuf = GetBuffer(quant);
        (VulkanBuffer resultBuf, VulkanBuffer? owned) = CastIfNeeded(quant, srcBuf, DType.F32);
        CacheOutput(output, owned ?? resultBuf);
        return output;
    }

    /// <summary>Casts <paramref name="srcBuf"/> to <paramref name="want"/> if needed; caller must free the returned ownedTemp.</summary>
    private (VulkanBuffer buf, VulkanBuffer? owned) CastIfNeeded(Tensor src, VulkanBuffer srcBuf, DType want)
    {
        if (src.DType == want) return (srcBuf, null);

        // Preloaded weights are cast once and reused — skip the per-call cast dispatch + temp alloc.
        if (_xfer.TryGetWeightCast(src, want, out VulkanBuffer cachedCast)) return (cachedCast, null);
        bool cacheThis = _xfer.ShouldCacheCast(src);

        long elements = src.ElementCount;
        ulong outBytes = (ulong)(elements * want.SizeInBytes);
        VulkanBuffer dst = _xfer.AllocateDevice(outBytes);

        if (src.DType == DType.F8E4M3 && want == DType.F16)
        {
            VulkanKernel k = GetKernel("cast_f8e4m3_f16", storageBufferCount: 2, _default1DSpec);
            Span<byte> pc = stackalloc byte[8];
            BinaryWriteUInt(pc, 0, (uint)elements);
            BinaryWriteFloat(pc, 4, src.Fp8ScaleFactor);
            Span<ulong> bufs = stackalloc ulong[] { srcBuf.Handle, dst.Handle };
            Dispatch(k, bufs, pc, GroupCount(elements, LocalX1D));
            return FinishCast(src, want, dst, cacheThis);
        }

        // GGUF quantized -> F16 dequant. Wired the same way as every other weight-dtype cast here
        // (lazy, on first use, cached via _weightCastCache when the source is a preloaded weight) —
        // there is no separate "model loading" step, matching CudaBackend.CastOnGpu's identical
        // quantized-source handling in its own CastIfNeeded-equivalent.
        if (src.DType.IsQuantized && want == DType.F16)
        {
            string dequantShader = src.DType.Name switch
            {
                "Q4_0" => "dequant_q4_0",
                "Q5_0" => "dequant_q5_0",
                "Q8_0" => "dequant_q8_0",
                "Q4_K" => "dequant_q4_k",
                "Q5_K" => "dequant_q5_k",
                "Q6_K" => "dequant_q6_k",
                _ => throw new NotSupportedException(
                    $"Vulkan GGUF dequant for {src.DType.Name} not implemented. Supported: Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K."),
            };
            long blockCount = elements / src.DType.BlockElementCount;
            // dequant_q6_k emits 4 outputs/thread (64 threads/super-block); every other dequant kernel
            // is one thread per output element.
            long dispatchCount = dequantShader == "dequant_q6_k" ? blockCount * 64 : elements;

            VulkanKernel dqKernel = GetKernel(dequantShader, storageBufferCount: 2, _default1DSpec);
            Span<byte> dqPc = stackalloc byte[4];
            BinaryWriteUInt(dqPc, 0, (uint)blockCount);
            Span<ulong> dqBufs = stackalloc ulong[] { srcBuf.Handle, dst.Handle };
            Dispatch(dqKernel, dqBufs, dqPc, GroupCount(dispatchCount, LocalX1D));
            return FinishCast(src, want, dst, cacheThis);
        }
        if (src.DType.IsQuantized && want == DType.F32)
        {
            // quant -> F16 -> F32 (the dequant kernels' native output is F16, matching CUDA's CastOnGpu).
            (VulkanBuffer midQ, VulkanBuffer? midQOwned) = CastIfNeeded(src, srcBuf, DType.F16);
            VulkanKernel castKernel = GetKernel("cast_f16_f32", storageBufferCount: 2, _default1DSpec);
            Span<byte> castPc = stackalloc byte[4]; BinaryWriteUInt(castPc, 0, (uint)elements);
            Span<ulong> castBufs = stackalloc ulong[] { midQ.Handle, dst.Handle };
            Dispatch(castKernel, castBufs, castPc, GroupCount(elements, LocalX1D));
            if (midQOwned is not null) _xfer.FreeDevice(midQOwned);
            return FinishCast(src, want, dst, cacheThis);
        }

        string shader;
        if (src.DType == DType.F32 && want == DType.F16) shader = "cast_f32_f16";
        else if (src.DType == DType.F16 && want == DType.F32) shader = "cast_f16_f32";
        else if (src.DType == DType.BF16 && want == DType.F32) shader = "cast_bf16_f32";
        else if (src.DType == DType.F32 && want == DType.BF16) shader = "cast_f32_bf16";
        else if (src.DType == DType.BF16 && want == DType.F16)
        {
            // BF16 -> F16 = BF16 -> F32 -> F16 (two-stage; no direct kernel, BF16/F16 share no exponent
            // range shortcut worth a dedicated shader for how rarely this specific pair is hit).
            (VulkanBuffer mid, _) = CastIfNeeded(src, srcBuf, DType.F32);
            VulkanKernel kk = GetKernel("cast_f32_f16", storageBufferCount: 2, _default1DSpec);
            Span<byte> pc2 = stackalloc byte[4]; BinaryWriteUInt(pc2, 0, (uint)elements);
            Span<ulong> bufs2 = stackalloc ulong[] { mid.Handle, dst.Handle };
            Dispatch(kk, bufs2, pc2, GroupCount(elements, LocalX1D));
            _xfer.FreeDevice(mid);
            return FinishCast(src, want, dst, cacheThis);
        }
        else if (src.DType == DType.F8E4M3 && want == DType.F32)
        {
            // FP8 -> F32 = FP8 -> F16 -> F32 (two-stage)
            (VulkanBuffer mid, _) = CastIfNeeded(src, srcBuf, DType.F16);
            // Now cast mid (F16) to F32
            VulkanKernel kk = GetKernel("cast_f16_f32", storageBufferCount: 2, _default1DSpec);
            Span<byte> pc2 = stackalloc byte[4]; BinaryWriteUInt(pc2, 0, (uint)elements);
            Span<ulong> bufs2 = stackalloc ulong[] { mid.Handle, dst.Handle };
            Dispatch(kk, bufs2, pc2, GroupCount(elements, LocalX1D));
            _xfer.FreeDevice(mid);
            return FinishCast(src, want, dst, cacheThis);
        }
        else
        {
            dst.Dispose();
            throw new NotSupportedException($"Vulkan cast {src.DType.Name} -> {want.Name} not implemented.");
        }

        VulkanKernel kernel = GetKernel(shader, storageBufferCount: 2, _default1DSpec);

        Span<byte> push = stackalloc byte[4];
        BinaryWriteUInt(push, 0, (uint)elements);
        Span<ulong> b = stackalloc ulong[] { srcBuf.Handle, dst.Handle };
        Dispatch(kernel, b, push, GroupCount(elements, LocalX1D));
        return FinishCast(src, want, dst, cacheThis);
    }

    /// <summary>Tail of <see cref="CastIfNeeded"/>: promotes <paramref name="dst"/> to the cache, else returns it as an owned temporary.</summary>
    private (VulkanBuffer buf, VulkanBuffer? owned) FinishCast(Tensor src, DType want, VulkanBuffer dst, bool cacheThis)
    {
        if (cacheThis)
        {
            _xfer.StoreWeightCast(src, want, dst);
            return (dst, null);
        }
        return (dst, dst);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void BinaryWriteUInt(Span<byte> buf, int offset, uint value)
    {
        buf[offset + 0] = (byte)(value & 0xFF);
        buf[offset + 1] = (byte)((value >> 8) & 0xFF);
        buf[offset + 2] = (byte)((value >> 16) & 0xFF);
        buf[offset + 3] = (byte)((value >> 24) & 0xFF);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void BinaryWriteFloat(Span<byte> buf, int offset, float value)
    {
        uint u = *(uint*)&value;
        BinaryWriteUInt(buf, offset, u);
    }

    #endregion

    #region Linear algebra

    /// <summary>Matrix multiply: output[M,N] = a[M,K] @ b[K,N]. Falls back to FP32 kernel if either input is FP32.</summary>
    public void MatMul(Tensor output, Tensor a, Tensor b)
    {
        using OpScope _ = EnterOp();
        DispatchMatmul(output, a, b, transposeA: false, transposeB: false, bias: null);
    }

    /// <summary>Opt-in switch for <see cref="Linear"/>'s INT8 dot-product GEMM path (see <see cref="TryDispatchInt8Linear"/>). Defaults from <c>HARTSYINFERENCE_VK_INT8=1</c> at construction, same as <c>CudaBackend.EnableW8A8</c>; settable afterward so tests/tooling can toggle it without an env var and a fresh process.</summary>
    public bool EnableInt8Linear { get; set; }

    public void Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias)
    {
        using OpScope _ = EnterOp();
        if (TryDispatchInt8Linear(output, input, weight, bias)) return;
        // input [M, K], weight [N, K] → output [M, N]   ⇒  C = A @ B^T  with A=input, B=weight
        DispatchMatmul(output, input, weight, transposeA: false, transposeB: true, bias: bias);
    }

    /// <summary>Opt-in INT8 dot-product GEMM path for <see cref="Linear"/> (<c>HARTSYINFERENCE_VK_INT8=1</c>), wiring the already-validated <see cref="MatMulInt8"/>/<see cref="Int8Quantizer"/> pair (bit-exact on the 3060 per <c>docs/Research/VULKAN_OPTIMIZATION.md</c>) into the normal model-code call path — the explicit open item both that doc and <c>ROADMAP.md</c> tracked as "wire the INT8 quantizer into Vulkan model loading." Re-quantizes BOTH weight and activation on EVERY call via the CPU-side <see cref="Int8Quantizer.RowwiseSymmetric"/> — correct and wired end-to-end, but not yet perf-optimal: caching the weight's quantized form across calls (weights don't change between calls, only activations do) is the natural follow-up and is intentionally NOT done here, to keep this pass bounded — a persistent per-weight INT8 cache needs its own lifecycle wiring (freed alongside <see cref="FreeWeights"/>) that deserves its own review, not a rushed addition here. Narrowly scoped to the plain 2-D F32 case (K%4==0, F32 in/out) on a device exposing the integer dot-product feature; anything else (F16, batched, non-4-divisible K, feature unavailable, opted out) falls through to the normal GEMM path completely unchanged.</summary>
    private unsafe bool TryDispatchInt8Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias)
    {
        if (!EnableInt8Linear || !Vk.HasInt8DotProduct) return false;
        if (output.Shape.Rank != 2 || input.Shape.Rank != 2 || weight.Shape.Rank != 2) return false;
        if (output.DType != DType.F32 || input.DType != DType.F32 || weight.DType != DType.F32) return false;

        int M = (int)output.Shape[0], N = (int)output.Shape[1], K = (int)input.Shape[1];
        if ((K & 3) != 0) return false;
        if ((int)weight.Shape[0] != N || (int)weight.Shape[1] != K) return false;
        if (bias is not null && bias.DType != DType.F32) return false;

        // Both quantizer calls read .DataPointer directly (CPU-side) — a real, documented D2H sync if
        // either operand happened to be GPU-resident. Activations from a prior GPU op commonly are; this
        // is the known cost of this opt-in path until the weight-side cache lands (see doc comment above).
        (Tensor inQuant, Tensor inScale) = Int8Quantizer.RowwiseSymmetric(input);
        (Tensor wQuant, Tensor wScale) = Int8Quantizer.RowwiseSymmetric(weight);
        try
        {
            MatMulInt8(output, inQuant, wQuant, inScale, wScale);
            if (bias is not null)
            {
                float* outP = (float*)output.DataPointer;
                float* biasP = (float*)bias.DataPointer;
                for (int m = 0; m < M; m++)
                    for (int n = 0; n < N; n++)
                        outP[m * N + n] += biasP[n];
            }
            return true;
        }
        finally
        {
            inQuant.Dispose(); inScale.Dispose(); wQuant.Dispose(); wScale.Dispose();
        }
    }

    public void BatchedMatMul(Tensor output, Tensor a, Tensor b)
    {
        long batch = a.Shape[0];
        long M = a.Shape[1];
        long K = a.Shape[2];
        bool bIs2D = b.Shape.Rank == 2;
        long N = bIs2D ? b.Shape[1] : b.Shape[2];

        // Fall back to per-batch matmul. Works correctly though slower than a single
        // batched dispatch — Phase-4 optimization will fuse this.
        for (long bi = 0; bi < batch; bi++)
        {
            // Slice through views/sub-tensors is non-trivial without TensorView ops here.
            // Simplest correct approach: dispatch one matmul per batch using offset push constants.
            DispatchMatmulBatched(output, a, b, bi, bIs2D);
        }
    }

    private void DispatchMatmulBatched(Tensor output, Tensor a, Tensor b, long batchIndex, bool bIs2D)
    {
        // For Phase-3.5 we promote each batch slice to a virtual matmul by bumping the offset
        // baked into push-constants, but keep the same dispatch shape. Implemented by reusing
        // DispatchMatmul on whole tensors — falls back to general kernel.
        // (A truly batched kernel is a Phase-4 optimization.)
        // The matmul shader uses lda/ldb/ldc; we set them so the tile loads from the right
        // slice. Easiest: reuse non-batched path by treating the batched slice as a separate
        // (M, N, K) GEMM. Since strides are baked in lda/ldb/ldc and we don't yet expose
        // explicit offset push constants, we delegate to DispatchMatmul on the entire tensor,
        // and rely on the kernel's bounds checks. This produces correct results when batch=1.
        if (a.Shape[0] != 1)
            throw new NotImplementedException("VulkanBackend.BatchedMatMul: batch > 1 needs a per-slice dispatch. Use the CPU backend or split the batch in the caller until the v2 path lands.");
        DispatchMatmul(output, a, b, transposeA: false, transposeB: false, bias: null);
    }

    /// <summary>Resolves the GEMM compute dtype: FP8 outputs compute in F16; F16 falls back to F32 without device support.</summary>
    // Deliberately does NOT promote F32 outputs to F16 to reach coopmat: tried this (2026-07-31), reverted the
    // same session. Measured payoff was ~0 (16/2112 GEMMs on a real Krea2 run — see the coopmat engagement
    // counters below) because the real blocker turns out to be shape, not dtype (see DispatchMatmul's M/N/K
    // comment), so there was no throughput win to weigh against the real, unverified precision cost: F16 has
    // a 5-bit exponent (max ~65504) vs F32/TF32's 8-bit range, and this path would have applied to FP8-
    // dequantized-weight GEMMs specifically, exactly where activation magnitudes are least predictable — never
    // actually exercised under that condition before it was reverted. If the shape blocker (below) gets fixed
    // and F32-output coopmat becomes worth revisiting, re-derive this from a real e2e SSIM/pixel-diff gate,
    // not from "CUDA already runs GEMMs at reduced precision by default" alone — that's necessary but not
    // sufficient (TF32's wider exponent range is not the same guarantee as F16's).
    private DType ResolveGemmDtype(DType outputDType)
    {
        DType gemmDtype = outputDType;
        if (gemmDtype.IsFp8) gemmDtype = DType.F16;
        if (gemmDtype == DType.F16 && !Capabilities.SupportsF16) gemmDtype = DType.F32;
        return gemmDtype;
    }

    private void DispatchMatmul(Tensor output, Tensor a, Tensor b, bool transposeA, bool transposeB, Tensor? bias)
    {
        // Resolve M, N, K from the OPERAND shapes (mirrors CudaBackend.LinearImpl: n/k come from the
        // weight, m from element-count / k) — NOT from output.Shape's rank structure. A caller may
        // legitimately shape the output as [B, S, heads, headDim] (e.g. Krea2Attention's Q/K/V, split
        // this way so RmsNorm/RoPE can normalize over headDim without a reshape) where the true GEMM
        // column count (heads·headDim) spans TWO trailing dims, not just the last one. Naively flattening
        // "all-but-last dims into M, last dim into N" silently computes the wrong-shaped GEMM in that case
        // (M too large, N too small) — reading input rows past its actual extent (out-of-bounds VRAM) and
        // using only a slice of the weight matrix, producing near-zero/garbage output for most rows.
        // N is fully determined by B's column count regardless of how output happens to be shaped, so
        // deriving M as ElementCount/N is always correct and a strict generalization of the old rank==2
        // and "last dim is the real N" cases (both keep the same M/N here).
        int N = transposeB ? (int)b.Shape[0] : (int)b.Shape[b.Shape.Rank - 1];
        int M = (int)(output.ElementCount / N);
        int K = transposeA ? (int)a.Shape[0] : (int)a.Shape[a.Shape.Rank - 1];

        // TryDispatchCoopmat below hard-requires M/N/K all multiples of 16 (its own doc comment: "spec
        // handles partial fragments IF the host pads the buffer" — this codebase doesn't pad, so it needs
        // exact multiples). Measured on a real Krea2 run (2026-07-31, the coopmat engagement counters
        // below): 0.8% (16/2112) of this model's GEMMs reach coopmat at all, and the 16 that do aren't the
        // expensive ones — every per-block QKV/FFN/out-proj Linear operates on the joint [txtSeq+imgSeq]
        // sequence, and txtSeq comes straight from the tokenized+encoded prompt length (Krea2Transformer.cs:
        // `txtSeq = (int)encoderHidden.Shape[1]`) with no padding to a fixed length. Root-caused on this
        // exact prompt: imgSeq=4096 (a multiple of 16, as expected for a square patchified latent), but
        // txtSeq=13 → jointSeq=4109, 3 short of the next multiple of 16 — and since txtSeq is
        // prompt-length-dependent, essentially no real prompt will land on a multiple of 16 by chance. This
        // was chased down a dtype path first (an F32-output Linear could never reach coopmat's OUTPUT_F32
        // support because gemmDtype was derived from output.DType with no F32→F16 promotion) — that fix was
        // real and correct but bought ~0 wall-clock (see ResolveGemmDtype's comment for why it was reverted
        // the same session) BECAUSE this shape gate blocks the same GEMMs regardless of dtype.
        //
        // A host-side M-padding fix (allocate a scratch A/output buffer padded to the next multiple of 16,
        // device-to-device copy the real M rows in, dispatch coopmat against the padded shape, let the
        // caller's tensor size naturally ignore the padding tail) was ATTEMPTED AND REVERTED 2026-07-31.
        // It passed every test thrown at it — a from-scratch CPU-reference correctness check, a 200-iteration
        // no-sync leak/stress test at Krea2's exact real M/K/N scale, and two full real-CLI runs (2-step,
        // 4-step) with correct-looking output and coopmat engagement climbing to 48-75% — but a subsequent
        // full 8-step real Krea2 run failed with `Vulkan error -4 (ErrorDeviceLost): vkQueueSubmit2` — a
        // genuine GPU driver-level fault/reset, not a logical bug or an allocation failure (which would throw
        // ErrorOutOfDeviceMemory, not lose the device). This happened on an otherwise-clean 4090 (an earlier,
        // separate 40-minute apparent "hang" on the SAME fix turned out to be caused by an unrelated external
        // process — a ComfyUI backend — silently consuming ~12.7GB of VRAM on this shared box; that was ruled
        // out as the sole explanation once the device-loss reproduced on a verified-clean GPU). No root cause
        // was found before reverting — the leading suspects are (a) RecordCopyAndBarrier's barrier not
        // actually bridging the copy and the coopmat dispatch's read if VulkanCommandStream.AcquireRecording
        // ever splits them across two separately-submitted command buffers (same-queue submission order does
        // NOT by itself guarantee memory-visibility ordering across separate vkQueueSubmit2 calls — only an
        // explicit barrier within the SAME command buffer does), or (b) a genuine out-of-bounds access from a
        // size/offset miscalculation that only manifests under the real model's specific op-interleaving
        // (SDPA's per-head shared-buffer reuse, Concat, CfgEulerStep's address-preserving in-place update)
        // rather than a simple repeated-Linear loop. Debugging this properly needs Vulkan validation layers
        // or compute-sanitizer-class tooling (already used elsewhere in this repo for exactly this bug class
        // — see TROUBLESHOOTING.md), not more blind full-CLI-run attempts. See benchmarks/scoreboards/
        // VULKAN.md and docs/Checklists/ROADMAP.md §3 for the full writeup; the ggml/llama.cpp shared-memory-
        // staged-clamp alternative (see TryDispatchCoopmat's neighborhood) sidesteps this whole class of risk
        // by never allocating a separate scratch buffer at all, and is the recommended next attempt.
        //
        // Pick GEMM dtype to MATCH the output's storage dtype — otherwise the matmul kernel writes
        // a smaller element type (e.g. F16) into an F32-sized buffer and the model reads garbage
        // when it interprets the bytes as F32. The model's choice of output dtype dictates the
        // pipeline. F8 inputs always need to be cast (no F8 matmul kernel); cast all the way to
        // the output dtype, not just to F16.
        DType gemmDtype = ResolveGemmDtype(output.DType);

        // Coopmat2 fast path — tried BEFORE coopmat1, opt-in only (see EnableCoopMat2's doc comment for
        // why it isn't unconditional yet). Self-contained (does its own buffer acquisition/allocation), so
        // this must run before the coopmat1/tiled path's own outBuf is allocated below, or a successful
        // coopmat2 dispatch would leak that unused allocation. Gated on !transposeA && transposeB HERE
        // (not just inside TryDispatchCoopMat2, which throws on any other combination) because
        // DispatchMatmul is also the shared entry point for MatMul/BatchedMatMul, which call with
        // transposeB=false — must fall through to coopmat1/tiled cleanly for those, not throw.
        if (EnableCoopMat2 && gemmDtype == DType.F16 && !transposeA && transposeB)
        {
            long t0 = _profiler.IsEnabled ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            bool coopmat2Ok = TryDispatchCoopMat2(output, a, b, transposeA, transposeB, bias);
            if (_profiler.IsEnabled) _coopmat2GemmTicks += System.Diagnostics.Stopwatch.GetTimestamp() - t0;
            if (coopmat2Ok)
            {
                _coopmat2GemmCount++;
                return;
            }
        }

        VulkanBuffer aBuf = GetBuffer(a);
        VulkanBuffer bBuf = GetBuffer(b);
        (VulkanBuffer aRes, VulkanBuffer? aOwned) = CastIfNeeded(a, aBuf, gemmDtype);
        (VulkanBuffer bRes, VulkanBuffer? bOwned) = CastIfNeeded(b, bBuf, gemmDtype);

        VulkanBuffer? biasOwned = null;
        VulkanBuffer? biasRes = null;
        VulkanBuffer? biasRaw = null;
        if (bias is not null)
        {
            biasRaw = GetBuffer(bias);
            (biasRes, biasOwned) = CastIfNeeded(bias, biasRaw, output.DType);
        }

        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);

        // Coopmat fast path — VK_KHR_cooperative_matrix tensor-core-style 16x16x16 ops on FP16.
        // Only applicable when (a) the device exposes coopmat, (b) GEMM dtype is FP16, and
        // (c) M, N, K are all multiples of 16 (the fragment size). Bias is folded as a separate
        // BroadcastAdd dispatch after the matmul. Falls through to the tiled path otherwise.
        long tCoopmat0 = _profiler.IsEnabled ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        bool coopmatOk = TryDispatchCoopmat(output, aRes, bRes, M, N, K, transposeA, transposeB,
                               gemmDtype, outBuf, bias, biasRes, biasRaw);
        if (_profiler.IsEnabled) _coopmatGemmTicks += System.Diagnostics.Stopwatch.GetTimestamp() - tCoopmat0;
        if (coopmatOk)
        {
            _coopmatGemmCount++;
            if (aOwned is not null) _xfer.FreeDevice(aOwned);
            if (bOwned is not null) _xfer.FreeDevice(bOwned);
            if (biasOwned is not null) _xfer.FreeDevice(biasOwned);
            return;
        }

        _tiledGemmCount++;
        long tTiled0 = _profiler.IsEnabled ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        try
        {
            // Tile shape selected from M, N, K — see PickMatmulTile. Big tiles (128x128) win on
            // Flux/SDXL Linear shapes (M, N >= 128); small tiles (32x32) avoid wasted dispatch
            // threads on tiny GEMMs.
            (uint BM, uint BN, uint BK, uint TM, uint TN) = PickMatmulTile(M, N, K);
            uint localX = BN / TN;
            uint localY = BM / TM;

            string shader = "matmul_tiled" + DtypeSuffix(gemmDtype);
            ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
            {
                SpecConstant.UInt(0, localX),
                SpecConstant.UInt(1, localY),
                SpecConstant.UInt(2, 1),
                SpecConstant.UInt(10, BM),
                SpecConstant.UInt(11, BN),
                SpecConstant.UInt(12, BK),
                SpecConstant.UInt(13, TM),
                SpecConstant.UInt(14, TN),
                SpecConstant.Bool(15, transposeA),
                SpecConstant.Bool(16, transposeB),
                SpecConstant.Bool(17, bias is not null),
                SpecConstant.UInt(18, 0u),    // no fused activation in v1
                SpecConstant.Bool(19, false),
            };

            VulkanKernel k = GetKernel(shader, storageBufferCount: 5, spec);

            Span<byte> pc = stackalloc byte[11 * 4];
            BinaryWriteUInt(pc, 0, (uint)M);
            BinaryWriteUInt(pc, 4, (uint)N);
            BinaryWriteUInt(pc, 8, (uint)K);
            BinaryWriteUInt(pc, 12, (uint)(transposeA ? M : K));
            BinaryWriteUInt(pc, 16, (uint)(transposeB ? K : N));
            BinaryWriteUInt(pc, 20, (uint)N);
            // FP8 scale: only fold weight (b) scale into alpha when both are FP8 we already cast,
            // or when only one operand was FP8. For non-FP8 inputs, scale factors default to 1.0.
            BinaryWriteFloat(pc, 24, 1.0f);
            BinaryWriteFloat(pc, 28, 0.0f);
            BinaryWriteUInt(pc, 32, 0u);   // aOffset
            BinaryWriteUInt(pc, 36, 0u);   // bOffset
            BinaryWriteUInt(pc, 40, 0u);   // cOffset

            // Bias slot 3 must be valid even when not used; bind out as a placeholder.
            ulong biasHandle = (biasRes ?? outBuf).Handle;
            ulong residualHandle = outBuf.Handle;
            Span<ulong> bufs = stackalloc ulong[] { aRes.Handle, bRes.Handle, outBuf.Handle, biasHandle, residualHandle };

            uint groupsX = (uint)((N + BN - 1) / BN);
            uint groupsY = (uint)((M + BM - 1) / BM);
            Dispatch(k, bufs, pc, groupsX, groupsY, 1);

            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan DispatchMatmul tiled-path dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
        finally
        {
            // a/b/bias buffers from GetBuffer are tracked by VulkanGpuTransferHelper as transients;
            // they're freed during the next flush. Cast buffers are caller-owned and freed here.
            if (aOwned is not null) _xfer.FreeDevice(aOwned);
            if (bOwned is not null) _xfer.FreeDevice(bOwned);
            if (biasOwned is not null) _xfer.FreeDevice(biasOwned);
            if (_profiler.IsEnabled) _tiledGemmTicks += System.Diagnostics.Stopwatch.GetTimestamp() - tTiled0;
        }
    }

    /// <summary>INT8 quantized matmul via the cross-vendor integer dot-product extension.</summary>
    /// <remarks><c>output[m,n] = (a[M,K] @ b[N,K]^T) * scaleA[m] * scaleB[n]</c>.</remarks>
    // b follows the Linear weight convention ([N,K], row n contiguous along K). Scales are per row:
    // scaleA is F32 length M (per token/activation row), scaleB is F32 length N (per output channel) — the
    // standard scheme for accurate INT8 inference. The int32 accumulation is exact; only the dequant rounds.
    // Requires VulkanCapabilities.HasInt8DotProduct and K a multiple of 4. Shared-memory tiled.
    public void MatMulInt8(Tensor output, Tensor a, Tensor b, Tensor scaleA, Tensor scaleB)
    {
        using OpScope _ = EnterOp();
        if (!Vk.HasInt8DotProduct)
            throw new NotSupportedException("MatMulInt8 requires the integer dot-product feature; this device does not expose it.");
        if (a.DType != DType.I8 || b.DType != DType.I8)
            throw new ArgumentException($"MatMulInt8 requires I8 operands; got a={a.DType.Name}, b={b.DType.Name}.");
        if (output.DType != DType.F32)
            throw new ArgumentException($"MatMulInt8 output must be F32; got {output.DType.Name}.");
        if (scaleA.DType != DType.F32 || scaleB.DType != DType.F32)
            throw new ArgumentException($"MatMulInt8 scales must be F32; got scaleA={scaleA.DType.Name}, scaleB={scaleB.DType.Name}.");

        int M = (int)a.Shape[0];
        int K = (int)a.Shape[a.Shape.Rank - 1];
        int N = (int)b.Shape[0];
        if ((int)b.Shape[b.Shape.Rank - 1] != K)
            throw new ArgumentException($"MatMulInt8 inner-dim mismatch: a.K={K}, b.K={(int)b.Shape[b.Shape.Rank - 1]}.");
        if ((K & 3) != 0)
            throw new ArgumentException($"MatMulInt8 requires K % 4 == 0 (dotPacked4x8 packs 4 int8 per word); got K={K}.");
        if (scaleA.ElementCount != M)
            throw new ArgumentException($"MatMulInt8 scaleA must have M={M} elements; got {scaleA.ElementCount}.");
        if (scaleB.ElementCount != N)
            throw new ArgumentException($"MatMulInt8 scaleB must have N={N} elements; got {scaleB.ElementCount}.");

        VulkanBuffer aBuf = GetBuffer(a);
        VulkanBuffer bBuf = GetBuffer(b);
        VulkanBuffer saBuf = GetBuffer(scaleA);
        VulkanBuffer sbBuf = GetBuffer(scaleB);
        ulong outBytes = (ulong)((long)M * N * sizeof(float));
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            // Shared-memory tiled: BM x BN output block per workgroup, TM x TN per invocation.
            // local = (BN/TN, BM/TM) = (16, 16) = 256 threads. BKP is the K tile in packed-int32
            // units (real K per tile = BKP*4). Tiles are fixed; small shapes just bounds-check out.
            const uint BM = 64, BN = 64, BKP = 8, TM = 4, TN = 4;
            VulkanKernel k = GetKernel("matmul_int8", storageBufferCount: 5, new SpecConstant[]
            {
                SpecConstant.UInt(0, BN / TN), SpecConstant.UInt(1, BM / TM), SpecConstant.UInt(2, 1),
                SpecConstant.UInt(10, BM), SpecConstant.UInt(11, BN), SpecConstant.UInt(12, BKP),
                SpecConstant.UInt(13, TM), SpecConstant.UInt(14, TN),
            });

            Span<byte> pc = stackalloc byte[3 * 4];
            BinaryWriteUInt(pc, 0, (uint)M);
            BinaryWriteUInt(pc, 4, (uint)N);
            BinaryWriteUInt(pc, 8, (uint)K);

            Span<ulong> bufs = stackalloc ulong[] { aBuf.Handle, bBuf.Handle, outBuf.Handle, saBuf.Handle, sbBuf.Handle };
            uint groupsX = (uint)(((long)N + BN - 1) / BN);
            uint groupsY = (uint)(((long)M + BM - 1) / BM);
            Dispatch(k, bufs, pc, groupsX, groupsY, 1);

            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan MatMulInt8 dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    #endregion

    #region Convolution

    public void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor? bias, int strideH, int strideW, int padH, int padW)
    {
        using OpScope _ = EnterOp();
        int batch = (int)input.Shape[0];
        int inCh = (int)input.Shape[1];
        int inH = (int)input.Shape[2];
        int inW = (int)input.Shape[3];
        int outCh = (int)weight.Shape[0];
        int kH = (int)weight.Shape[2];
        int kW = (int)weight.Shape[3];
        int outH = (inH + 2 * padH - kH) / strideH + 1;
        int outW = (inW + 2 * padW - kW) / strideW + 1;

        // GEMM dtype must match output's storage dtype (see DispatchMatmul note).
        DType gemmDtype = ResolveGemmDtype(output.DType);

        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer wBuf = GetBuffer(weight);
        (VulkanBuffer inRes, VulkanBuffer? inOwned) = CastIfNeeded(input, inBuf, gemmDtype);
        (VulkanBuffer wRes, VulkanBuffer? wOwned) = CastIfNeeded(weight, wBuf, gemmDtype);

        // im2col: rows = N*Cin*kH*kW, cols = N*outH*outW (per the shader's flat layout). Until
        // per-batch base offsets land in the matmul kernel (Phase 4) we restrict to batch=1.
        if (batch != 1)
            throw new NotImplementedException("VulkanBackend.Conv2D: batch>1 requires per-batch base offsets in the matmul kernel — Phase 4 work item.");

        int gemmK = inCh * kH * kW;
        int fullN = outH * outW;

        // Tile over output positions (fullN) so peak im2col memory is bounded regardless of spatial
        // resolution — a real Krea2-on-Vulkan finding (2026-07-30): the untiled path materialized one
        // [gemmK, fullN] column matrix per Conv2D call, which at 1024x1024 VAE-decode resolution with
        // ~192 input channels needs ~7 GB for a SINGLE allocation and OOM'd even with the transformer's
        // weights already freed (QwenImageVaeDecoder.Decode -> QwenImageResample.Forward -> Conv2D).
        // colOffset=0/tileCols=fullN (the tileN==fullN case below) reproduces the prior untiled
        // dispatch exactly, so small convs (the overwhelming majority of call sites) pay zero extra
        // allocations or dispatches — this only kicks in above Conv2DMaxColTileBytes.
        long maxTileN = Math.Max(1L, (long)(Conv2DMaxColTileBytes / ((ulong)gemmK * (ulong)gemmDtype.SizeInBytes)));
        long tileN = Math.Min(fullN, maxTileN);
        long tileColElements = (long)gemmK * tileN;
        // The im2col/matmul GLSL address the column/output buffers with 32-bit indices. Above
        // int.MaxValue elements the index would wrap, silently corrupting output. Shrinking tileN
        // can't help when gemmK alone exceeds this range — fail loudly instead.
        if (tileColElements > int.MaxValue)
            throw new NotSupportedException(
                $"Vulkan Conv2D im2col tile needs {tileColElements} elements (Cin={inCh}, k={kH}x{kW}), " +
                "exceeding the shader's 32-bit index range even at the minimum 1-column tile. Use the CUDA backend.");
        ulong colBytes = (ulong)(tileColElements * gemmDtype.SizeInBytes);
        VulkanBuffer colBuf = _xfer.AllocateDevice(colBytes);

        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);

        try
        {
            (uint BM, uint BN, uint BK, uint TM, uint TN) = PickMatmulTile(outCh, (int)tileN, gemmK);
            uint localX = BN / TN;
            uint localY = BM / TM;
            string matmulShader = "matmul_tiled" + DtypeSuffix(gemmDtype);
            ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
            {
                SpecConstant.UInt(0, localX),
                SpecConstant.UInt(1, localY),
                SpecConstant.UInt(2, 1),
                SpecConstant.UInt(10, BM),
                SpecConstant.UInt(11, BN),
                SpecConstant.UInt(12, BK),
                SpecConstant.UInt(13, TM),
                SpecConstant.UInt(14, TN),
                SpecConstant.Bool(15, false),
                SpecConstant.Bool(16, false),
                SpecConstant.Bool(17, false),
                SpecConstant.UInt(18, 0u),
                SpecConstant.Bool(19, false),
            };
            VulkanKernel matmulKernel = GetKernel(matmulShader, 5, spec);
            string im2colShader = "im2col" + DtypeSuffix(gemmDtype);
            VulkanKernel im2colKernel = GetKernel(im2colShader, 2, _default1DSpec);

            Span<byte> im2colPc = stackalloc byte[14 * 4];
            Span<ulong> im2colBufs = stackalloc ulong[] { inRes.Handle, colBuf.Handle };
            Span<byte> matmulPc = stackalloc byte[11 * 4];
            Span<ulong> matmulBufs = stackalloc ulong[] { wRes.Handle, colBuf.Handle, outBuf.Handle, outBuf.Handle, outBuf.Handle };

            for (long nStart = 0; nStart < fullN; nStart += tileN)
            {
                long thisTileN = Math.Min(tileN, fullN - nStart);

                // Step 1: im2col (tile)
                {
                    BinaryWriteUInt(im2colPc, 0, (uint)batch);
                    BinaryWriteUInt(im2colPc, 4, (uint)inCh);
                    BinaryWriteUInt(im2colPc, 8, (uint)inH);
                    BinaryWriteUInt(im2colPc, 12, (uint)inW);
                    BinaryWriteUInt(im2colPc, 16, (uint)kH);
                    BinaryWriteUInt(im2colPc, 20, (uint)kW);
                    BinaryWriteUInt(im2colPc, 24, (uint)padH);
                    BinaryWriteUInt(im2colPc, 28, (uint)padW);
                    BinaryWriteUInt(im2colPc, 32, (uint)strideH);
                    BinaryWriteUInt(im2colPc, 36, (uint)strideW);
                    BinaryWriteUInt(im2colPc, 40, (uint)outH);
                    BinaryWriteUInt(im2colPc, 44, (uint)outW);
                    BinaryWriteUInt(im2colPc, 48, (uint)nStart);
                    BinaryWriteUInt(im2colPc, 52, (uint)thisTileN);
                    Dispatch(im2colKernel, im2colBufs, im2colPc, GroupCount(gemmK * thisTileN, LocalX1D));
                }

                // Step 2: matmul (tile) — weight[Cout, gemmK] @ col[gemmK, thisTileN] written into
                // the REAL [Cout, fullN] output buffer at column offset nStart (ldc=fullN, cOffset=nStart).
                {
                    int M = outCh, K = gemmK, N = (int)thisTileN;
                    BinaryWriteUInt(matmulPc, 0, (uint)M);
                    BinaryWriteUInt(matmulPc, 4, (uint)N);
                    BinaryWriteUInt(matmulPc, 8, (uint)K);
                    BinaryWriteUInt(matmulPc, 12, (uint)K);
                    BinaryWriteUInt(matmulPc, 16, (uint)N);
                    BinaryWriteUInt(matmulPc, 20, (uint)fullN);
                    BinaryWriteFloat(matmulPc, 24, 1.0f);
                    BinaryWriteFloat(matmulPc, 28, 0.0f);
                    BinaryWriteUInt(matmulPc, 32, 0u);
                    BinaryWriteUInt(matmulPc, 36, 0u);
                    BinaryWriteUInt(matmulPc, 40, (uint)nStart);

                    uint groupsX = (uint)(((uint)N + BN - 1) / BN);
                    uint groupsY = (uint)((M + BM - 1) / BM);
                    Dispatch(matmulKernel, matmulBufs, matmulPc, groupsX, groupsY, 1);
                }
            }

            // Step 3: optional bias add
            if (bias is not null)
            {
                VulkanBuffer biasRaw = GetBuffer(bias);
                (VulkanBuffer biasRes, VulkanBuffer? biasOwned) = CastIfNeeded(bias, biasRaw, output.DType);
                try
                {
                    string shader = "col2bias_add" + DtypeSuffix(output.DType);
                    VulkanKernel k = GetKernel(shader, 2, _default1DSpec);
                    Span<byte> pc = stackalloc byte[3 * 4];
                    BinaryWriteUInt(pc, 0, (uint)outCh);
                    BinaryWriteUInt(pc, 4, (uint)(outH * outW));
                    BinaryWriteUInt(pc, 8, (uint)(batch * outCh * outH * outW));
                    Span<ulong> bufs = stackalloc ulong[] { outBuf.Handle, biasRes.Handle };
                    Dispatch(k, bufs, pc, GroupCount(batch * outCh * outH * outW, LocalX1D));
                }
                finally
                {
                    if (biasOwned is not null) _xfer.FreeDevice(biasOwned);
                }
            }

            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan Conv2D dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
        finally
        {
            if (inOwned is not null) _xfer.FreeDevice(inOwned);
            if (wOwned is not null) _xfer.FreeDevice(wOwned);
            _xfer.FreeDevice(colBuf);
        }
    }

    #endregion

    #region Normalization

    private void DispatchPerRowNorm(
        string shader,
        int storageBufs,
        Tensor output,
        Tensor input,
        Tensor? weight,
        Tensor? bias,
        float eps,
        int normDim,
        int totalRows)
    {
        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer? wBuf = weight is null ? null : GetBuffer(weight);
        VulkanBuffer? bBuf = bias is null ? null : GetBuffer(bias);

        // Norm weight/bias are FP32 in the shader signature regardless of activation dtype.
        // Cast if user-supplied weight/bias are FP16.
        VulkanBuffer? wOwned = null, bOwned = null;
        VulkanBuffer? wEff = wBuf, bEff = bBuf;
        if (weight is not null && weight.DType != DType.F32) (wEff, wOwned) = CastIfNeeded(weight, wBuf!, DType.F32);
        if (bias is not null && bias.DType != DType.F32) (bEff, bOwned) = CastIfNeeded(bias, bBuf!, DType.F32);

        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            uint local = 256;
            ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
            {
                SpecConstant.UInt(0, local), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)
            };
            VulkanKernel k = GetKernel(shader, storageBufs, spec);

            Span<byte> pc = stackalloc byte[3 * 4];
            BinaryWriteUInt(pc, 0, (uint)normDim);
            BinaryWriteUInt(pc, 4, (uint)totalRows);
            BinaryWriteFloat(pc, 8, eps);

            // Build buffer list per shader expectation:
            //   3 SSBO: x, weight, y          (rmsnorm)
            //   4 SSBO: x, weight, bias, y    (layernorm, groupnorm, groupnorm_silu)
            if (storageBufs == 3)
            {
                Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, wEff!.Handle, outBuf.Handle };
                Dispatch(k, bufs, pc, (uint)totalRows, 1, 1);
            }
            else
            {
                Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, wEff!.Handle, bEff!.Handle, outBuf.Handle };
                Dispatch(k, bufs, pc, (uint)totalRows, 1, 1);
            }

            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan DispatchPerRowNorm dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
        finally
        {
            if (wOwned is not null) _xfer.FreeDevice(wOwned);
            if (bOwned is not null) _xfer.FreeDevice(bOwned);
        }
    }

    public void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps)
    {
        DispatchGroupNorm(output, input, weight, bias, groups, eps, fused: false);
    }

    public void GroupNormSilu(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps)
    {
        DispatchGroupNorm(output, input, weight, bias, groups, eps, fused: true);
    }

    private void DispatchGroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps, bool fused)
    {
        int N = (int)input.Shape[0];
        int C = (int)input.Shape[1];
        int H = input.Shape.Rank > 2 ? (int)input.Shape[2] : 1;
        int W = input.Shape.Rank > 3 ? (int)input.Shape[3] : 1;

        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer wBuf = GetBuffer(weight);
        VulkanBuffer bBuf = GetBuffer(bias);

        VulkanBuffer? wOwned = null, bOwned = null;
        VulkanBuffer wEff = wBuf, bEff = bBuf;
        if (weight.DType != DType.F32) { (wEff, wOwned) = CastIfNeeded(weight, wBuf, DType.F32); }
        if (bias.DType != DType.F32) { (bEff, bOwned) = CastIfNeeded(bias, bBuf, DType.F32); }

        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            string shader = (fused ? "groupnorm_silu" : "groupnorm") + DtypeSuffix(input.DType);
            uint local = 256;
            ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
            {
                SpecConstant.UInt(0, local), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)
            };
            VulkanKernel k = GetKernel(shader, 4, spec);

            Span<byte> pc = stackalloc byte[6 * 4];
            BinaryWriteUInt(pc, 0, (uint)N);
            BinaryWriteUInt(pc, 4, (uint)C);
            BinaryWriteUInt(pc, 8, (uint)H);
            BinaryWriteUInt(pc, 12, (uint)W);
            BinaryWriteUInt(pc, 16, (uint)groups);
            BinaryWriteFloat(pc, 20, eps);

            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, wEff.Handle, bEff.Handle, outBuf.Handle };
            uint groupsX = (uint)(N * groups);
            Dispatch(k, bufs, pc, groupsX, 1, 1);

            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan DispatchGroupNorm dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
        finally
        {
            if (wOwned is not null) _xfer.FreeDevice(wOwned);
            if (bOwned is not null) _xfer.FreeDevice(bOwned);
        }
    }

    public void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps)
    {
        int normDim = (int)input.Shape[input.Shape.Rank - 1];
        int totalRows = (int)(input.ElementCount / normDim);
        string shader = "layernorm" + DtypeSuffix(input.DType);
        DispatchPerRowNorm(shader, 4, output, input, weight, bias, eps, normDim, totalRows);
    }

    public void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps)
    {
        int normDim = (int)input.Shape[input.Shape.Rank - 1];
        int totalRows = (int)(input.ElementCount / normDim);
        string shader = "rmsnorm" + DtypeSuffix(input.DType);
        DispatchPerRowNorm(shader, 3, output, input, weight, null, eps, normDim, totalRows);
    }

    /// <summary>Wan2.2 VAE channel-wise RMS norm (mirrors <c>CudaBackend.WanRmsNormChannel</c>). No override existed at all before this — every call fell through to <c>IBackend</c>'s CPU-loop default, which reads <c>input.DataPointer</c>/writes <c>output.DataPointer</c> directly: a full D2H sync, a single-threaded scalar reduction over C with a cache-hostile stride-<c>spatial</c> access pattern (each channel step is a full row apart), then an H2D re-upload for whatever consumes the result. Found via the Krea2 VAE decode profiling pass (2026-07-31): the decoder's one call site (<c>QwenImageVaeDecoder</c>'s final head norm) runs at the FULL 1024×1024 output resolution — <c>[1,96,1024,1024]</c>, ~402 MB — the worst possible shape for this fallthrough, and a real contributor to Krea2's ~2300× VAE-decode gap vs CUDA (which has always had a real kernel for this op). F32 only, matching both references (CUDA and the CPU default read/write <c>float*</c> unconditionally); anything else falls back to the CPU default.</summary>
    public void WanRmsNormChannel(Tensor output, Tensor input, Tensor? gamma, float eps)
    {
        using OpScope _op = EnterOp();
        if (input.DType != DType.F32 || output.DType != DType.F32 || (gamma is not null && gamma.DType != DType.F32))
        {
            ((IBackend)this).WanRmsNormChannel(output, input, gamma, eps);
            return;
        }
        int c = (int)input.Shape[1];
        long spatial = input.ElementCount / ((long)input.Shape[0] * c);
        long numPos = input.Shape[0] * spatial;
        float sqrtC = MathF.Sqrt(c);

        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer gammaBuf = gamma is not null ? GetBuffer(gamma) : inBuf;   // dummy when hasGamma=0 — never read
        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            VulkanKernel k = GetKernel("wan_rms_norm_channel", 3, _default1DSpec);
            Span<byte> pc = stackalloc byte[6 * 4];
            BinaryWriteUInt(pc, 0, (uint)c);
            BinaryWriteUInt(pc, 4, (uint)spatial);
            BinaryWriteFloat(pc, 8, eps);
            BinaryWriteFloat(pc, 12, sqrtC);
            BinaryWriteUInt(pc, 16, gamma is not null ? 1u : 0u);
            BinaryWriteUInt(pc, 20, (uint)numPos);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, gammaBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(numPos, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan WanRmsNormChannel dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    /// <summary>DiT adaLN modulation: <c>out = in*scale + (shift ?? 0)</c>, broadcasting scale/shift <c>[B,C]</c> over input/output <c>[B,seqLen,C]</c>. No override existed at all before this — every call fell through to IBackend's F32-only CPU default, which throws on an F16 activation (the shape Krea2's DiT actually uses — found via a real Krea2-on-Vulkan run, not a synthetic test).</summary>
    public void AffineBroadcastLastDim(Tensor output, Tensor input, Tensor scale, Tensor? shift)
    {
        using OpScope _op = EnterOp();
        if ((input.DType != DType.F32 && input.DType != DType.F16) || output.DType != input.DType
            || scale.DType != DType.F32 || (shift is not null && shift.DType != DType.F32))
        {
            IBackend.AffineBroadcastLastDimReference(output, input, scale, shift);
            return;
        }
        int rank = input.Shape.Rank;
        int dim = (int)input.Shape[rank - 1];
        int seqLen = rank >= 2 ? (int)input.Shape[rank - 2] : 1;
        long total = input.ElementCount;

        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer scaleBuf = GetBuffer(scale);
        VulkanBuffer? shiftBuf = shift is not null ? GetBuffer(shift) : null;
        ulong outBytes = (ulong)(total * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            string shader = "affine_broadcast_last_dim" + (shift is null ? "_noshift" : "") + DtypeSuffix(input.DType);
            int bufferCount = shift is null ? 3 : 4;
            VulkanKernel k = GetKernel(shader, bufferCount, _default1DSpec);
            Span<byte> pc = stackalloc byte[3 * 4];
            BinaryWriteUInt(pc, 0, (uint)dim);
            BinaryWriteUInt(pc, 4, (uint)seqLen);
            BinaryWriteUInt(pc, 8, (uint)total);
            Span<ulong> bufs = shift is null
                ? stackalloc ulong[] { inBuf.Handle, scaleBuf.Handle, outBuf.Handle }
                : stackalloc ulong[] { inBuf.Handle, scaleBuf.Handle, shiftBuf!.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(total, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan AffineBroadcastLastDim dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    public void GatedResidualLastDim(Tensor output, Tensor residual, Tensor value, Tensor gate)
    {
        using OpScope _op = EnterOp();
        if ((value.DType != DType.F32 && value.DType != DType.F16)
            || residual.DType != value.DType || output.DType != value.DType || gate.DType != DType.F32)
        {
            ((IBackend)this).GatedResidualLastDim(output, residual, value, gate);
            return;
        }
        int rank = value.Shape.Rank;
        int dim = (int)value.Shape[rank - 1];
        int seqLen = rank >= 2 ? (int)value.Shape[rank - 2] : 1;
        long total = value.ElementCount;

        VulkanBuffer resBuf = GetBuffer(residual);
        VulkanBuffer valBuf = GetBuffer(value);
        VulkanBuffer gateBuf = GetBuffer(gate);
        ulong outBytes = (ulong)(total * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            string shader = "gated_residual_last_dim" + DtypeSuffix(value.DType);
            VulkanKernel k = GetKernel(shader, 4, _default1DSpec);
            Span<byte> pc = stackalloc byte[3 * 4];
            BinaryWriteUInt(pc, 0, (uint)dim);
            BinaryWriteUInt(pc, 4, (uint)seqLen);
            BinaryWriteUInt(pc, 8, (uint)total);
            Span<ulong> bufs = stackalloc ulong[] { resBuf.Handle, valBuf.Handle, gateBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(total, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan GatedResidualLastDim dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    public void AdaInstanceNorm1d(Tensor output, Tensor input, Tensor gamma, Tensor beta, float eps)
    {
        throw new NotImplementedException("VulkanBackend.AdaInstanceNorm1d not yet implemented. Use the CPU backend for Kokoro / StyleTTS 2 prosody and decoder paths.");
    }

    public void LeakyRelu(Tensor output, Tensor input, float slope)
    {
        throw new NotImplementedException("VulkanBackend.LeakyRelu not yet implemented. Use the CPU backend for Kokoro / StyleTTS 2.");
    }

    #endregion

    #region Attention

    /// <summary>Max head dim <see cref="DispatchFlashAttention"/>'s shared-memory KV tile can hold (MAX_D in sdpa_flash.comp.glsl). Real model head dims (64-128) fit; the rare >128 case (some SDXL variants use 160) falls back to the materialized 3-pass path rather than corrupting memory.</summary>
    private const int FlashMaxHeadDim = 128;

    public void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale, bool allowF16 = false)
    {
        using OpScope _op = EnterOp();
        if (query.Shape.Rank != 4)
            throw new NotImplementedException("VulkanBackend SDPA expects [B, H, S, D] inputs.");

        // A mask whose query axis is 1 stores one [Skv] row broadcast over every query (a bias that depends only
        // on the key — Wan-Animate-2's log_scale band). Every shader below indexes it as [Sq, Skv], so expand it
        // here; handing them the short buffer would read past the allocation and go unnoticed.
        long sqRows = query.Shape[2], skvRows = key.Shape[2];
        using Tensor? expandedMask = mask is not null && MaskQueryRows(mask) == 1 && sqRows > 1
            ? ExpandKeyOnlyMask(mask, sqRows, skvRows) : null;
        mask = expandedMask ?? mask;

        int hq = (int)query.Shape[1], hkv = (int)key.Shape[1], headDim = (int)query.Shape[3];
        if (headDim <= FlashMaxHeadDim && hkv > 0 && hq % hkv == 0)
        {
            DispatchFlashAttention(output, query, key, value, mask, scale,
                causal: false, qOffset: 0, slidingWindow: 0, skv: (int)key.Shape[2], kvGroup: hq / hkv);
            return;
        }
        // Head dim exceeds the flash kernel's shared-memory tile, or Q/K head counts aren't an integer
        // GQA ratio — fall back to the materialized (but still correct) 3-pass path.
        ScaledDotProductAttentionNaive(output, query, key, value, mask, scale, allowF16);
    }

    /// <summary>Fused online-softmax SDPA — never materializes the [Sq,Skv] score matrix. Shared by both <see cref="ScaledDotProductAttention"/> (causal=false, kvGroup/skv derived from tensor shapes) and <see cref="FlashAttention"/> (explicit kvLen/kvGroup — the KV-cache buffer may be over-allocated beyond the currently-valid length, and causal/qOffset/slidingWindow are real for decode).</summary>
    private void DispatchFlashAttention(
        Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask,
        float scale, bool causal, int qOffset, int slidingWindow, int skv, int kvGroup)
    {
        int batch = (int)query.Shape[0], hq = (int)query.Shape[1], sq = (int)query.Shape[2], headDim = (int)query.Shape[3];

        DType dtype = ResolveGemmDtype(output.DType);
        VulkanBuffer qBuf = GetBuffer(query);
        VulkanBuffer kBuf = GetBuffer(key);
        VulkanBuffer vBuf = GetBuffer(value);
        (VulkanBuffer qRes, VulkanBuffer? qOwned) = CastIfNeeded(query, qBuf, dtype);
        (VulkanBuffer kRes, VulkanBuffer? kOwned) = CastIfNeeded(key, kBuf, dtype);
        (VulkanBuffer vRes, VulkanBuffer? vOwned) = CastIfNeeded(value, vBuf, dtype);

        VulkanBuffer? maskBuf = null;
        if (mask is not null)
        {
            if (mask.DType != DType.F32)
                throw new NotSupportedException($"sdpa_flash mask must be F32, got {mask.DType.Name}");
            maskBuf = GetBuffer(mask);
        }

        ulong outBytes = (ulong)((long)batch * hq * sq * headDim * dtype.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            string shader = (mask is null ? "sdpa_flash" : "sdpa_flash_mask") + DtypeSuffix(dtype);
            int bufferCount = mask is null ? 4 : 5;
            // Unlike every other kernel here, sdpa_flash hardcodes local_size_x = TILE (one thread per
            // KV-tile key, not a generically-retunable spec constant) — no spec constants to pass.
            VulkanKernel k = GetKernel(shader, bufferCount, ReadOnlySpan<SpecConstant>.Empty);

            // kvCapacity is the K/V buffer's ACTUAL per-head stride (key.Shape[2]) — distinct from skv
            // (the number of valid positions to attend over) whenever the caller passes an
            // over-allocated KV-cache buffer (FlashAttention's kvLen < the buffer's real capacity).
            int kvCapacity = (int)key.Shape[2];
            Span<byte> pc = stackalloc byte[10 * 4];
            BinaryWriteUInt(pc, 0, (uint)hq);
            BinaryWriteUInt(pc, 4, (uint)(hq / kvGroup));
            BinaryWriteUInt(pc, 8, (uint)sq);
            BinaryWriteUInt(pc, 12, (uint)skv);
            BinaryWriteUInt(pc, 16, (uint)kvCapacity);
            BinaryWriteUInt(pc, 20, (uint)headDim);
            BinaryWriteFloat(pc, 24, scale);
            BinaryWriteUInt(pc, 28, causal ? 1u : 0u);
            BinaryWriteUInt(pc, 32, (uint)qOffset);
            BinaryWriteUInt(pc, 36, (uint)slidingWindow);

            Span<ulong> bufs = mask is null
                ? stackalloc ulong[] { qRes.Handle, kRes.Handle, vRes.Handle, outBuf.Handle }
                : stackalloc ulong[] { qRes.Handle, kRes.Handle, vRes.Handle, maskBuf!.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, (uint)sq, (uint)hq, (uint)batch);

            if (output.DType == dtype)
            {
                CacheOutput(output, outBuf);
            }
            else
            {
                ulong outBytesFinal = (ulong)(output.ElementCount * output.DType.SizeInBytes);
                VulkanBuffer finalBuf = _xfer.AllocateDevice(outBytesFinal);
                try
                {
                    string castShader;
                    if (dtype == DType.F16 && output.DType == DType.F32) castShader = "cast_f16_f32";
                    else if (dtype == DType.F32 && output.DType == DType.F16) castShader = "cast_f32_f16";
                    else throw new NotSupportedException($"sdpa_flash dtype mismatch {dtype.Name}->{output.DType.Name}");

                    VulkanKernel castK = GetKernel(castShader, 2, _default1DSpec);
                    Span<byte> castPc = stackalloc byte[4]; BinaryWriteUInt(castPc, 0, (uint)output.ElementCount);
                    Span<ulong> castBufs = stackalloc ulong[] { outBuf.Handle, finalBuf.Handle };
                    Dispatch(castK, castBufs, castPc, GroupCount(output.ElementCount, LocalX1D));
                    CacheOutput(output, finalBuf);
                }
                catch (Exception ex)
                {
                    Logs.Error("Vulkan sdpa_flash output-dtype cast dispatch failed", ex);
                    finalBuf.Dispose();
                    throw;
                }
                _xfer.FreeDevice(outBuf);
            }
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan sdpa_flash dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
        finally
        {
            if (qOwned is not null) _xfer.FreeDevice(qOwned);
            if (kOwned is not null) _xfer.FreeDevice(kOwned);
            if (vOwned is not null) _xfer.FreeDevice(vOwned);
        }
    }

    /// <summary>True — the flash kernel above gives Vulkan the GPU-resident autoregressive decode toolkit this flag advertises (e.g. unblocks Zonos onto the fast path instead of CUDA-only host glue).</summary>
    public bool FlashDecodeSupported => true;

    // ── Step-graph capture (see VulkanStepGraph's doc comment for the full design) ─────────────────────────
    // Requires push descriptors (VK_KHR_push_descriptor) — without them there is no descriptor-binding
    // mechanism a replayed command buffer can safely use (a pool-allocated set can be invalidated by a later
    // pool reset). Gated on Vk.HasPushDescriptor, NOT on the process's eager-mode HARTSYINFERENCE_VK_PUSH_DESCRIPTORS
    // default — capture always pushes regardless of that setting (see Dispatch's capture branch).
    public bool StepGraphSupported => Vk.HasPushDescriptor;

    public bool StepGraphReady => _stepGraph?.IsReady == true && !_capturingStepGraph;

    /// <summary>Owner token for the single step-graph slot (see IBackend.StepGraphOwner) — shared across models the way CudaBackend's is, so alternating models under KEEP_MODELS correctly force a recapture.</summary>
    public object? StepGraphOwner { get; set; }

    public void StepGraphBegin()
    {
        // The capture buffer is a SEPARATE VkCommandBuffer/submission from the normal stream — Vulkan gives no
        // automatic memory-visibility ordering between independent submissions to the same queue without an
        // explicit wait. Krea2Transformer.ForwardPatched always issues at least one normal-mode CopyInto (the
        // per-step temb/tembMod refresh) immediately before this call; without flushing+waiting here, that
        // write can still be sitting unsubmitted (or submitted-but-not-completed) when the capture buffer's
        // dispatches read it, reading stale/zero data instead — silently wrong, not a crash.
        _stream.WaitIdleHost();
        _stepGraph ??= new VulkanStepGraph(_vkDevice.Handle, _vkDevice.ComputeQueue, Vk.ComputeQueueFamilyIndex);
        _stepGraph.BeginCapture();
        _capturingStepGraph = true;
        _xfer.CapturingStepGraph = true;
    }

    public void StepGraphEndAndLaunch()
    {
        if (!_capturingStepGraph || _stepGraph is null) return;
        _capturingStepGraph = false;
        _xfer.CapturingStepGraph = false;
        _stepGraph.EndCaptureAndLaunch();
    }

    public void StepGraphLaunch()
    {
        if (_stepGraph is null || !_stepGraph.IsReady)
            throw new InvalidOperationException("VulkanBackend.StepGraphLaunch called with no captured graph.");
        // Same cross-submission visibility requirement as StepGraphBegin — the caller's pre-launch CopyInto
        // refresh (normal stream) must be complete before the captured buffer's dispatches read it.
        _stream.WaitIdleHost();
        _stepGraph.Launch();
    }

    public void StepGraphReset()
    {
        if (_capturingStepGraph)
        {
            _capturingStepGraph = false;
            _xfer.CapturingStepGraph = false;
        }
        _stepGraph?.Reset();
        // Free everything the capture (or an aborted attempt at one) retained — safe only now that the
        // graph itself can no longer be replayed (Reset() above dropped IsReady).
        _xfer.ReleaseStepGraphRetained();
    }

    /// <summary>Fused online-softmax attention (see <see cref="DispatchFlashAttention"/>) for the common case. Falls back to the CPU numeric reference — the same one <c>IBackend</c>'s own default uses — for softcap/sink/ALiBi (Gemma-2/GPT-OSS/MPT-class models) and head dims the shared-memory tile can't hold: a documented "flash-lite" scope boundary, not silently wrong output.</summary>
    public unsafe void FlashAttention(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup, bool causal, int qOffset, float scale, float softcap = 0f, Tensor? sink = null, int slidingWindow = 0, Tensor? alibiSlopes = null)
    {
        int headDim = (int)query.Shape[query.Shape.Rank - 1];
        if (softcap == 0f && sink is null && alibiSlopes is null && headDim <= FlashMaxHeadDim)
        {
            using OpScope _op = EnterOp();
            DispatchFlashAttention(output, query, key, value, mask: null, scale, causal, qOffset, slidingWindow, skv: kvLen, kvGroup);
            return;
        }
        AttentionReference.FlashAttention(output, query, key, value, kvLen, kvGroup, causal, qOffset, scale, softcap, sink, slidingWindow, alibiSlopes);
    }

    /// <summary>Query rows the additive mask stores; 1 means it depends only on the key.</summary>
    private static long MaskQueryRows(Tensor mask)
        => mask.Shape.Rank switch { 2 => mask.Shape[0], 3 => mask.Shape[1], _ => mask.Shape[2] };

    /// <summary>Materializes the <c>[blocks·Sq, Skv]</c> duplicate of a key-only mask row, which is the layout the mask_add dispatch indexes.</summary>
    private static unsafe Tensor ExpandKeyOnlyMask(Tensor mask, long sq, long skv)
    {
        long blocks = mask.ElementCount / skv;
        Tensor full = new Tensor(new TensorShape(blocks * sq, skv), DType.F32);
        float* src = (float*)mask.DataPointer, dst = (float*)full.DataPointer;
        for (long blk = 0; blk < blocks; blk++)
            for (long q = 0; q < sq; q++)
                Buffer.MemoryCopy(src + blk * skv, dst + (blk * sq + q) * skv, skv * 4, skv * 4);
        return full;
    }

    /// <summary>Naive 3-pass SDPA: Q*K^T, mask add, softmax, *V — dispatched once per (B*H) head. FlashAttention-style fusion is a Phase-4 optimization.</summary>
    private void ScaledDotProductAttentionNaive(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale, bool allowF16 = false)
    {
        if (query.Shape.Rank != 4)
            throw new NotImplementedException("VulkanBackend SDPA expects [B, H, S, D] inputs.");

        long B  = query.Shape[0];
        long H  = query.Shape[1];
        long Sq = query.Shape[2];
        long D  = query.Shape[3];
        long Skv = key.Shape[2];
        long totalHeads = B * H;

        // Resolve dtype: must match output's storage dtype so the SDPA matmul writes match the
        // model's expected element size. FP8 inputs cast to F16 first.
        DType dtype = ResolveGemmDtype(output.DType);

        VulkanBuffer qBuf = GetBuffer(query);
        VulkanBuffer kBuf = GetBuffer(key);
        VulkanBuffer vBuf = GetBuffer(value);

        (VulkanBuffer qRes, VulkanBuffer? qOwned) = CastIfNeeded(query, qBuf, dtype);
        (VulkanBuffer kRes, VulkanBuffer? kOwned) = CastIfNeeded(key, kBuf, dtype);
        (VulkanBuffer vRes, VulkanBuffer? vOwned) = CastIfNeeded(value, vBuf, dtype);

        // Mask is always FP32 in the model code (CLIP causal mask). Keep it as F32 — the mask_add
        // kernel widens the score before adding regardless of score dtype.
        VulkanBuffer? maskBuf = null;
        long maskBroadcastSize = 0;
        long maskTotalSize = 0;
        if (mask is not null)
        {
            if (mask.DType != DType.F32)
                throw new NotImplementedException($"SDPA mask must be F32, got {mask.DType.Name}");
            maskBuf = GetBuffer(mask);
            // Mask shape may be [Sq, Skv] or [1, 1, Sq, Skv]; we only care about (per-head) elements.
            maskBroadcastSize = (long)Sq * Skv;
            maskTotalSize = mask.ElementCount;
        }

        // Query tiling for the plain (no-mask) case: the full [B, H, Sq, Skv] score matrix OOMs at Wan-Video
        // full res (24 × 14040² ≈ 19 GB). Tile the query axis into Br-row chunks so only [totalHeads, Br, Skv]
        // is materialized; each tile reuses the same matmul/softmax dispatches (numerically identical to the
        // untiled path — mirrors CudaBackend.SdpaTiledF32NoMask, which is bit-exact for a single tile). Masked
        // callers keep Br = Sq (untiled), so the mask-offset math below is unchanged for them.
        long Br = Sq;
        ulong fullScoreBytes = (ulong)(totalHeads * Sq * Skv) * (ulong)dtype.SizeInBytes;
        bool forceTile = Environment.GetEnvironmentVariable("HARTSY_SDPA_FORCE_TILED") == "1";
        if (mask is null && (forceTile || fullScoreBytes > (1UL << 30)))   // > 1 GiB score matrix → tile
        {
            long perRow = totalHeads * Skv * dtype.SizeInBytes;            // bytes per query row across all heads
            Br = Math.Max(1, (512L * 1024 * 1024) / Math.Max(1, perRow));  // ~512 MB score-tile budget
            if (Br > Sq) Br = Sq;
            if (int.TryParse(Environment.GetEnvironmentVariable("HARTSY_SDPA_TILE"), out int envBr) && envBr > 0)
                Br = Math.Min(envBr, Sq);
        }

        // Scratch: scores/probs sized for a single [totalHeads, Br, Skv] query tile (Br == Sq ⇒ the original
        // full-matrix behavior). Separate probsBuf avoids readonly+writeonly aliasing in the softmax shader.
        ulong scoresElems = (ulong)(totalHeads * Br * Skv);
        VulkanBuffer scoresBuf = _xfer.AllocateDevice(scoresElems * (ulong)dtype.SizeInBytes);
        VulkanBuffer probsBuf = _xfer.AllocateDevice(scoresElems * (ulong)dtype.SizeInBytes);

        // Output buffer (dtype)
        ulong outElems = (ulong)(B * H * Sq * D);
        ulong outBytesDtype = outElems * (ulong)dtype.SizeInBytes;
        VulkanBuffer outBufLocal = _xfer.AllocateDevice(outBytesDtype);

        try
        {
            for (long q0 = 0; q0 < Sq; q0 += Br)
            {
                long curBr = Math.Min(Br, Sq - q0);

                // Per-head dispatch over this query tile.
                for (long bh = 0; bh < totalHeads; bh++)
                {
                    uint qOff = (uint)(bh * Sq * D + q0 * D);
                    uint kOff = (uint)(bh * Skv * D);
                    uint sOff = (uint)(bh * curBr * Skv);   // packed within the tile buffer
                    uint vOff = (uint)(bh * Skv * D);
                    uint outOff = (uint)(bh * Sq * D + q0 * D);

                    // 1) scores = Q_tile @ K^T * scale  →  shape (curBr, Skv)
                    DispatchMatmulWithOffsets(
                        qRes.Handle, kRes.Handle, scoresBuf.Handle,
                        curBr, Skv, D,
                        transposeA: false, transposeB: true,
                        alpha: scale, beta: 0.0f,
                        aOffset: qOff, bOffset: kOff, cOffset: sOff,
                        dtype: dtype);

                    // 1b) optional mask add: scores += mask  (broadcast across heads). Only reached when
                    // mask is non-null, which forces Br == Sq (untiled), so curBr == Sq and this matches the
                    // original per-head offsets exactly.
                    if (maskBuf is not null)
                    {
                        // The mask may be either [Sq, Skv] or [B, H, Sq, Skv]. We use modulo so a
                        // [1, 1, Sq, Skv] broadcast cycles back to offset 0 for every head.
                        uint maskOff = (uint)((bh * maskBroadcastSize) % maskTotalSize);
                        DispatchMaskAdd(scoresBuf.Handle, maskBuf.Handle, dtype,
                            (uint)maskBroadcastSize, scoreOffset: sOff, maskOffset: maskOff);
                    }

                    // 2) softmax along last dim per row of scores  →  probsBuf (separate output)
                    DispatchSoftmaxRows(scoresBuf.Handle, probsBuf.Handle, dtype, (int)Skv, (int)curBr, sOff, sOff);

                    // 3) out_tile = probs @ V  →  shape (curBr, D)
                    DispatchMatmulWithOffsets(
                        probsBuf.Handle, vRes.Handle, outBufLocal.Handle,
                        curBr, D, Skv,
                        transposeA: false, transposeB: false,
                        alpha: 1.0f, beta: 0.0f,
                        aOffset: sOff, bOffset: vOff, cOffset: outOff,
                        dtype: dtype);
                }
            }

            // Cast result back to output dtype if needed (should match if model uses consistent dtype).
            if (output.DType == dtype)
            {
                CacheOutput(output, outBufLocal);
            }
            else
            {
                // Cast outBufLocal (dtype) to output.DType
                ulong outBytesFinal = (ulong)(output.ElementCount * output.DType.SizeInBytes);
                VulkanBuffer finalBuf = _xfer.AllocateDevice(outBytesFinal);
                try
                {
                    string shader;
                    if (dtype == DType.F16 && output.DType == DType.F32) shader = "cast_f16_f32";
                    else if (dtype == DType.F32 && output.DType == DType.F16) shader = "cast_f32_f16";
                    else throw new NotSupportedException($"SDPA dtype mismatch {dtype.Name}->{output.DType.Name}");

                    VulkanKernel k = GetKernel(shader, 2, _default1DSpec);
                    Span<byte> pc = stackalloc byte[4]; BinaryWriteUInt(pc, 0, (uint)output.ElementCount);
                    Span<ulong> bufs = stackalloc ulong[] { outBufLocal.Handle, finalBuf.Handle };
                    Dispatch(k, bufs, pc, GroupCount(output.ElementCount, LocalX1D));
                    CacheOutput(output, finalBuf);
                }
                catch (Exception ex)
                {
                    Logs.Error("Vulkan SDPA output-dtype cast dispatch failed", ex);
                    finalBuf.Dispose();
                    throw;
                }
                _xfer.FreeDevice(outBufLocal);
            }
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan ScaledDotProductAttention dispatch failed", ex);
            outBufLocal.Dispose();
            throw;
        }
        finally
        {
            _xfer.FreeDevice(scoresBuf);
            _xfer.FreeDevice(probsBuf);
            // q/k/v/mask are transients (or cached) and drained by the next flush.
            if (qOwned is not null) _xfer.FreeDevice(qOwned);
            if (kOwned is not null) _xfer.FreeDevice(kOwned);
            if (vOwned is not null) _xfer.FreeDevice(vOwned);
        }
    }

    private void DispatchMaskAdd(ulong scoresHandle, ulong maskHandle, DType dtype,
        uint count, uint scoreOffset, uint maskOffset)
    {
        string shader = "mask_add" + DtypeSuffix(dtype);
        VulkanKernel k = GetKernel(shader, 2, _default1DSpec);
        Span<byte> pc = stackalloc byte[3 * 4];
        BinaryWriteUInt(pc, 0, count);
        BinaryWriteUInt(pc, 4, scoreOffset);
        BinaryWriteUInt(pc, 8, maskOffset);
        Span<ulong> bufs = stackalloc ulong[] { scoresHandle, maskHandle };
        Dispatch(k, bufs, pc, GroupCount(count, LocalX1D));
    }

    /// <summary>Dispatch a tiled matmul on raw buffer handles with explicit element offsets. Used by SDPA's per-head loop.</summary>
    private void DispatchMatmulWithOffsets(
        ulong aHandle, ulong bHandle, ulong cHandle,
        long M, long N, long K,
        bool transposeA, bool transposeB,
        float alpha, float beta,
        uint aOffset, uint bOffset, uint cOffset,
        DType dtype)
    {
        (uint BM, uint BN, uint BK, uint TM, uint TN) = PickMatmulTile(M, N, K);
        uint localX = BN / TN;
        uint localY = BM / TM;

        string shader = "matmul_tiled" + DtypeSuffix(dtype);
        ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
        {
            SpecConstant.UInt(0, localX),
            SpecConstant.UInt(1, localY),
            SpecConstant.UInt(2, 1),
            SpecConstant.UInt(10, BM),
            SpecConstant.UInt(11, BN),
            SpecConstant.UInt(12, BK),
            SpecConstant.UInt(13, TM),
            SpecConstant.UInt(14, TN),
            SpecConstant.Bool(15, transposeA),
            SpecConstant.Bool(16, transposeB),
            SpecConstant.Bool(17, false),
            SpecConstant.UInt(18, 0u),
            SpecConstant.Bool(19, false),
        };
        VulkanKernel k = GetKernel(shader, 5, spec);

        Span<byte> pc = stackalloc byte[11 * 4];
        BinaryWriteUInt(pc, 0, (uint)M);
        BinaryWriteUInt(pc, 4, (uint)N);
        BinaryWriteUInt(pc, 8, (uint)K);
        BinaryWriteUInt(pc, 12, (uint)(transposeA ? M : K));
        BinaryWriteUInt(pc, 16, (uint)(transposeB ? K : N));
        BinaryWriteUInt(pc, 20, (uint)N);
        BinaryWriteFloat(pc, 24, alpha);
        BinaryWriteFloat(pc, 28, beta);
        BinaryWriteUInt(pc, 32, aOffset);
        BinaryWriteUInt(pc, 36, bOffset);
        BinaryWriteUInt(pc, 40, cOffset);

        Span<ulong> bufs = stackalloc ulong[] { aHandle, bHandle, cHandle, cHandle, cHandle };
        uint groupsX = (uint)((N + BN - 1) / BN);
        uint groupsY = (uint)((M + BM - 1) / BM);
        Dispatch(k, bufs, pc, groupsX, groupsY, 1);
    }

    /// <summary>Dispatch softmax over a contiguous block of rows starting at the given element offsets in src/dst buffers.</summary>
    private void DispatchSoftmaxRows(ulong srcHandle, ulong dstHandle, DType dtype, int N, int rows, uint srcOffset, uint dstOffset)
    {
        string shader = "softmax" + DtypeSuffix(dtype);
        ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
        {
            SpecConstant.UInt(0, 256), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)
        };
        VulkanKernel k = GetKernel(shader, 2, spec);
        // softmax kernel uses one push-const block — but its baseOffset applies to BOTH src and dst.
        // When src and dst differ, we dispatch in a way that passes one offset; we need src/dst at the
        // SAME offset within their respective buffers for each call (which is the case in SDPA).
        Span<byte> pc = stackalloc byte[3 * 4];
        BinaryWriteUInt(pc, 0, (uint)N);
        BinaryWriteUInt(pc, 4, (uint)rows);
        BinaryWriteUInt(pc, 8, srcOffset);   // baseOffset = srcOffset; we require dstOffset == srcOffset
        if (srcOffset != dstOffset)
            throw new InvalidOperationException("DispatchSoftmaxRows requires srcOffset == dstOffset");
        Span<ulong> bufs = stackalloc ulong[] { srcHandle, dstHandle };
        Dispatch(k, bufs, pc, (uint)rows, 1, 1);
    }

    #endregion

    #region Activations

    public void Gelu(Tensor output, Tensor input) => DispatchElementwise(5u /* gelu_tanh */, output, input, null, scalar: 0, minVal: 0, maxVal: 0);
    public void Silu(Tensor output, Tensor input) => DispatchElementwise(3u, output, input, null, scalar: 0, minVal: 0, maxVal: 0);

    public void Sigmoid(Tensor output, Tensor input)
    {
        // Op 6 in the existing elementwise.comp.glsl — already baked into elementwise_f32.spv.
        DispatchElementwise(6u, output, input, null, scalar: 0, minVal: 0, maxVal: 0);
    }

    public void Tanh(Tensor output, Tensor input)
    {
        // Op 8 — requires elementwise.comp.glsl recompile (source updated; SPIR-V will be
        // regenerated by the build pipeline). Until then this dispatches to an op code
        // that doesn't exist in the current .spv and will produce zeros — the test mock
        // path catches that, and CpuBackend remains the production-safe fallback.
        DispatchElementwise(8u, output, input, null, scalar: 0, minVal: 0, maxVal: 0);
    }

    public void Elu(Tensor output, Tensor input, float alpha)
    {
        // Op 9 — alpha goes through the existing `scalar` push-constant slot.
        DispatchElementwise(9u, output, input, null, scalar: alpha, minVal: 0, maxVal: 0);
    }

    public void Snake(Tensor output, Tensor input, Tensor alpha, Tensor? beta)
    {
        using OpScope _op = EnterOp();
        int batch = (int)input.Shape[0], channels = (int)input.Shape[1], timeDim = (int)input.Shape[2];
        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer alphaBuf = GetBuffer(alpha);
        VulkanBuffer? betaBuf = beta is null ? null : GetBuffer(beta);
        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            // USE_BETA gates a #if-compiled binding, not a spec constant — vanilla and snake-beta
            // are distinct SPIR-V modules (snake vs snake_beta) with different buffer counts.
            string shader = (beta is null ? "snake" : "snake_beta") + DtypeSuffix(output.DType);
            int bufferCount = beta is null ? 3 : 4;
            VulkanKernel k = GetKernel(shader, bufferCount, _default1DSpec);
            Span<byte> pc = stackalloc byte[3 * 4];
            BinaryWriteUInt(pc, 0, (uint)batch);
            BinaryWriteUInt(pc, 4, (uint)channels);
            BinaryWriteUInt(pc, 8, (uint)timeDim);
            Span<ulong> bufs = beta is null
                ? stackalloc ulong[] { inBuf.Handle, alphaBuf.Handle, outBuf.Handle }
                : stackalloc ulong[] { inBuf.Handle, alphaBuf.Handle, betaBuf!.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(output.ElementCount, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan Snake dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    public void Conv1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups)
    {
        using OpScope _op = EnterOp();
        int batch = (int)input.Shape[0], cIn = (int)input.Shape[1], tIn = (int)input.Shape[2];
        int cOut = (int)weight.Shape[0], kernel = (int)weight.Shape[2];
        int tOut = (int)output.Shape[2];
        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer wBuf = GetBuffer(weight);
        // HAS_BIAS is a real spec constant here (constant_id=10), but the descriptor set layout is
        // still fixed at 4 sequential bindings regardless — bind the input buffer as a never-read
        // placeholder when there's no bias, same convention as Conv2dDepthwise.
        VulkanBuffer biasBuf = bias is null ? inBuf : GetBuffer(bias);
        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            string shader = "conv1d" + DtypeSuffix(output.DType);
            SpecConstant[] spec = { _default1DSpec[0], _default1DSpec[1], _default1DSpec[2], SpecConstant.Bool(10, bias is not null) };
            VulkanKernel k = GetKernel(shader, 4, spec);
            Span<byte> pc = stackalloc byte[10 * 4];
            BinaryWriteUInt(pc, 0, (uint)batch);
            BinaryWriteUInt(pc, 4, (uint)cIn);
            BinaryWriteUInt(pc, 8, (uint)cOut);
            BinaryWriteUInt(pc, 12, (uint)tIn);
            BinaryWriteUInt(pc, 16, (uint)tOut);
            BinaryWriteUInt(pc, 20, (uint)kernel);
            BinaryWriteUInt(pc, 24, (uint)stride);
            BinaryWriteUInt(pc, 28, unchecked((uint)padLeft));
            BinaryWriteUInt(pc, 32, (uint)dilation);
            BinaryWriteUInt(pc, 36, (uint)groups);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, wBuf.Handle, biasBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(output.ElementCount, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan Conv1d dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    public void ConvTranspose1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups)
    {
        // The shipped conv_transpose1d.comp.glsl has no groups field at all — its inner loop sums
        // over the full cIn range assuming dense [cIn, cOut, K] weights. Depthwise (groups>1, e.g.
        // BigVGAN anti-aliased upsampling) would silently read the wrong weight slice if dispatched
        // through this path, so fail loudly instead of producing wrong output.
        if (groups != 1)
            throw new NotSupportedException(
                $"Vulkan ConvTranspose1d: groups={groups} not supported — the shipped shader only handles dense (groups=1) weights. " +
                "Use the CPU backend for depthwise transposed conv (e.g. BigVGAN upsampling).");
        using OpScope _op = EnterOp();
        int batch = (int)input.Shape[0], cIn = (int)input.Shape[1], tIn = (int)input.Shape[2];
        int cOut = (int)weight.Shape[1], kernel = (int)weight.Shape[2];
        int tOut = (int)output.Shape[2];
        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer wBuf = GetBuffer(weight);
        VulkanBuffer biasBuf = bias is null ? inBuf : GetBuffer(bias);
        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            string shader = "conv_transpose1d" + DtypeSuffix(output.DType);
            SpecConstant[] spec = { _default1DSpec[0], _default1DSpec[1], _default1DSpec[2], SpecConstant.Bool(10, bias is not null) };
            VulkanKernel k = GetKernel(shader, 4, spec);
            Span<byte> pc = stackalloc byte[9 * 4];
            BinaryWriteUInt(pc, 0, (uint)batch);
            BinaryWriteUInt(pc, 4, (uint)cIn);
            BinaryWriteUInt(pc, 8, (uint)cOut);
            BinaryWriteUInt(pc, 12, (uint)tIn);
            BinaryWriteUInt(pc, 16, (uint)tOut);
            BinaryWriteUInt(pc, 20, (uint)kernel);
            BinaryWriteUInt(pc, 24, (uint)stride);
            BinaryWriteUInt(pc, 28, unchecked((uint)padLeft));
            BinaryWriteUInt(pc, 32, (uint)dilation);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, wBuf.Handle, biasBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(output.ElementCount, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan ConvTranspose1d dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    #endregion

    #region Element-wise

    public void Add(Tensor output, Tensor a, Tensor b) => DispatchElementwise(0u, output, a, b, scalar: 0, minVal: 0, maxVal: 0);
    public void Mul(Tensor output, Tensor a, Tensor b) => DispatchElementwise(1u, output, a, b, scalar: 0, minVal: 0, maxVal: 0);
    public void Scale(Tensor output, Tensor input, float scalar) => DispatchElementwise(2u, output, input, null, scalar, minVal: 0, maxVal: 0);
    public void Clamp(Tensor output, Tensor input, float min, float max)
        => DispatchElementwise(7u, output, input, null, scalar: 0, minVal: min, maxVal: max);

    /// <summary>Regression gate for a real Krea2-on-Vulkan bug (2026-07-30): no <c>VulkanBackend</c> override existed, so every call fell through to <c>IBackend</c>'s CPU-loop default — found capture-illegal via a real <c>HARTSY_DIT_GRAPH=1</c> Krea2 run (<c>DiTUtils.Modulate</c>'s <c>AddScalar(scale, +1)</c>, called TWICE per block × 28 blocks per forward pass — the (1+scale) modulation convention every DiT block uses) and, independent of graph mode, a D2H sync 56 times per denoise step regardless.</summary>
    public void AddScalar(Tensor output, Tensor input, float scalar) => DispatchElementwise(10u, output, input, null, scalar, minVal: 0, maxVal: 0);

    private void DispatchElementwise(uint op, Tensor output, Tensor a, Tensor? b, float scalar, float minVal, float maxVal)
    {
        VulkanBuffer aBuf = GetBuffer(a);
        VulkanBuffer? bBuf = b is null ? null : GetBuffer(b);

        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            string shader = "elementwise" + DtypeSuffix(output.DType);
            ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
            {
                SpecConstant.UInt(0, LocalX1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1),
                SpecConstant.UInt(10, op),
            };
            VulkanKernel k = GetKernel(shader, 3, spec);

            Span<byte> pc = stackalloc byte[4 * 4];
            BinaryWriteUInt(pc, 0, (uint)output.ElementCount);
            BinaryWriteFloat(pc, 4, scalar);
            BinaryWriteFloat(pc, 8, minVal);
            BinaryWriteFloat(pc, 12, maxVal);

            Span<ulong> bufs = stackalloc ulong[] { aBuf.Handle, bBuf?.Handle ?? aBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(output.ElementCount, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan DispatchElementwise dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    #endregion

    #region Transpose / Permute / GeGlu / BroadcastAdd

    public void Transpose2D(Tensor output, Tensor input, int d1, int d2)
    {
        int rank = input.Shape.Rank;
        int B = (int)(input.ElementCount / (d1 * d2));
        VulkanBuffer inBuf = GetBuffer(input);
        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            string shader = "transpose" + DtypeSuffix(input.DType);
            VulkanKernel k = GetKernel(shader, 2, _default1DSpec);
            Span<byte> pc = stackalloc byte[3 * 4];
            BinaryWriteUInt(pc, 0, (uint)B);
            BinaryWriteUInt(pc, 4, (uint)d1);
            BinaryWriteUInt(pc, 8, (uint)d2);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(output.ElementCount, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan Transpose2D dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    public void Permute0213(Tensor output, Tensor input, int s, int h, int d)
    {
        int B = (int)(input.ElementCount / ((long)s * h * d));
        VulkanBuffer inBuf = GetBuffer(input);
        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            string shader = "permute_0213" + DtypeSuffix(input.DType);
            VulkanKernel k = GetKernel(shader, 2, _default1DSpec);
            Span<byte> pc = stackalloc byte[4 * 4];
            BinaryWriteUInt(pc, 0, (uint)B);
            BinaryWriteUInt(pc, 4, (uint)s);
            BinaryWriteUInt(pc, 8, (uint)h);
            BinaryWriteUInt(pc, 12, (uint)d);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(output.ElementCount, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan Permute0213 dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    public void GeGlu(Tensor output, Tensor input)
    {
        long lastDim = input.Shape[input.Shape.Rank - 1];
        long D = lastDim / 2;
        long outerCount = input.ElementCount / lastDim;

        VulkanBuffer inBuf = GetBuffer(input);
        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            string shader = "geglu" + DtypeSuffix(input.DType);
            VulkanKernel k = GetKernel(shader, 2, _default1DSpec);
            Span<byte> pc = stackalloc byte[2 * 4];
            BinaryWriteUInt(pc, 0, (uint)outerCount);
            BinaryWriteUInt(pc, 4, (uint)D);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(outerCount * D, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan GeGlu dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    public void BroadcastAdd(Tensor hidden, Tensor bias, int channels, int spatial)
    {
        VulkanBuffer hBuf = GetBuffer(hidden);
        VulkanBuffer bBuf = GetBuffer(bias);

        // bias dtype must match hidden — cast if not
        VulkanBuffer? bOwned = null;
        VulkanBuffer bEff = bBuf;
        if (bias.DType != hidden.DType) (bEff, bOwned) = CastIfNeeded(bias, bBuf, hidden.DType);

        try
        {
            string shader = "broadcast_add" + DtypeSuffix(hidden.DType);
            VulkanKernel k = GetKernel(shader, 2, _default1DSpec);
            Span<byte> pc = stackalloc byte[3 * 4];
            BinaryWriteUInt(pc, 0, (uint)channels);
            BinaryWriteUInt(pc, 4, (uint)spatial);
            BinaryWriteUInt(pc, 8, (uint)hidden.ElementCount);
            Span<ulong> bufs = stackalloc ulong[] { hBuf.Handle, bEff.Handle };
            Dispatch(k, bufs, pc, GroupCount(hidden.ElementCount, LocalX1D));
            // hidden's GPU contents just changed — re-cache so CPU readback (lazy-sync callback)
            // and subsequent GetBuffer hits return the post-add values, not the pre-upload CPU bytes.
            // CacheActivation handles idempotent re-cache for in-place ops; see PHASE_3_DEVIATIONS #17.
            CacheOutput(hidden, hBuf);
        }
        finally
        {
            if (bOwned is not null) _xfer.FreeDevice(bOwned);
        }
    }

    #endregion

    #region Shape ops

    /// <summary>Device-resident concat along <paramref name="dim"/>: one <c>vkCmdCopyBuffer</c> (multi-region when <c>dim</c> isn't the leading axis) per input, straight into <paramref name="output"/>'s buffer at the right byte offset — no compute shader needed, concatenation with contiguous inner strides is pure data movement. Overrides <c>IBackend</c>'s CPU-loop default (found capture-illegal via a real <c>HARTSY_DIT_GRAPH=1</c> Krea2 run: `ForwardCore`'s `Concat(joint, [txt, img], dim: 1)` — the text+image sequence join every DiT forward pass — read both inputs' <c>DataPointer</c> directly, which is capture-illegal and, outside capture, forced a D2H sync on every forward regardless of graph mode). Capture-aware, matching <see cref="CopyInto"/>'s pattern.</summary>
    public unsafe void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim)
    {
        long elemSize = output.DType.SizeInBytes;
        long innerStride = 1;
        for (int d = dim + 1; d < output.Shape.Rank; d++) innerStride *= output.Shape[d];
        long outerStride = 1;
        for (int d = 0; d < dim; d++) outerStride *= output.Shape[d];

        // Resolve every input's buffer BEFORE acquiring the recording command buffer: GetBuffer's cache-miss
        // path (Upload's non-ReBAR staging copy) calls _stream.SubmitAndAdvance() internally, which would end
        // any command buffer already acquired — issuing vkCmdCopyBuffer against a stale, already-submitted
        // handle afterward (caught by validation: "was recorded, but vkBeginCommandBuffer() was not called").
        VulkanBuffer[] srcBufs = new VulkanBuffer[inputs.Length];
        for (int i = 0; i < inputs.Length; i++) srcBufs[i] = GetBuffer(inputs[i]);

        VulkanBuffer outBuf = _capturingStepGraph
            ? (_xfer.TryGetCached(output, out VulkanBuffer? existingOut) && existingOut is not null ? existingOut : _xfer.AllocateDevice((ulong)(output.ElementCount * elemSize)))
            : _xfer.AllocateDevice((ulong)(output.ElementCount * elemSize));
        nint cb = _capturingStepGraph ? _stepGraph!.RecordingBuffer : _stream.AcquireRecording();

        long curDim = 0;
        Span<VkBufferCopy> regions = stackalloc VkBufferCopy[(int)outerStride];
        for (int i = 0; i < inputs.Length; i++)
        {
            Tensor t = inputs[i];
            VulkanBuffer srcBuf = srcBufs[i];
            long tSize = t.Shape[dim] * innerStride * elemSize;
            for (long o = 0; o < outerStride; o++)
            {
                long srcOffset = o * tSize;
                long dstOffset = (o * output.Shape[dim] * innerStride + curDim * innerStride) * elemSize;
                regions[(int)o] = new VkBufferCopy { srcOffset = (ulong)srcOffset, dstOffset = (ulong)dstOffset, size = (ulong)tSize };
            }
            fixed (VkBufferCopy* pRegions = regions)
                VulkanApi.vkCmdCopyBuffer(cb, srcBuf.Handle, outBuf.Handle, (uint)outerStride, (nint)pRegions);
            curDim += t.Shape[dim];
        }

        if (_capturingStepGraph)
        {
            VulkanCommandStream.RecordGlobalComputeBarrierOn(cb);
            CacheOutput(output, outBuf);
            return;
        }

        _stream.RecordGlobalComputeBarrier();
        _dispatchesSinceSubmit++;
        _dispatchesThisOp++;
        if (_dispatchesSinceSubmit >= FlushThreshold && _opNestingDepth == 0) DrainAndFlush();
        CacheOutput(output, outBuf);
    }

    public unsafe void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim)
    {
        SplitGeometry geometry = SplitContract.Validate(outputs, input, dim);
        // CPU fallback
        long elemSize = geometry.ElementSize;
        byte* inPtr = (byte*)input.DataPointer;
        long innerStride = geometry.Inner;
        long outerStride = geometry.Outer;

        long curDimOffset = 0;
        foreach (Tensor t in outputs)
        {
            byte* tPtr = (byte*)t.DataPointer;
            for (long o = 0; o < outerStride; o++)
            {
                long src = (o * input.Shape[dim] + curDimOffset) * innerStride * elemSize;
                long dst = o * t.Shape[dim] * innerStride * elemSize;
                long sz = t.Shape[dim] * innerStride * elemSize;
                Buffer.MemoryCopy(inPtr + src, tPtr + dst, sz, sz);
            }
            curDimOffset += t.Shape[dim];
        }
    }

    #endregion

    #region Sampling

    public void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW)
        => DispatchUpsample("upsample_nearest2d", output, input, scaleH, scaleW);

    public void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW)
        => DispatchUpsample("upsample_bilinear2d", output, input, scaleH, scaleW);

    private void DispatchUpsample(string baseName, Tensor output, Tensor input, int scaleH, int scaleW)
    {
        int N = (int)input.Shape[0], C = (int)input.Shape[1], H = (int)input.Shape[2], W = (int)input.Shape[3];
        VulkanBuffer inBuf = GetBuffer(input);
        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            string shader = baseName + DtypeSuffix(input.DType);
            VulkanKernel k = GetKernel(shader, 2, _default1DSpec);
            Span<byte> pc = stackalloc byte[6 * 4];
            BinaryWriteUInt(pc, 0, (uint)N);
            BinaryWriteUInt(pc, 4, (uint)C);
            BinaryWriteUInt(pc, 8, (uint)H);
            BinaryWriteUInt(pc, 12, (uint)W);
            BinaryWriteUInt(pc, 16, (uint)scaleH);
            BinaryWriteUInt(pc, 20, (uint)scaleW);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(output.ElementCount, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan DispatchUpsample dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    public void MaxPool2D(Tensor output, Tensor input, int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        if (output.DType != DType.F32 && output.DType != DType.F16)
            throw new NotSupportedException($"Vulkan MaxPool2D supports F32/F16 output — got {output.DType}.");
        using OpScope _op = EnterOp();
        int N = (int)input.Shape[0], C = (int)input.Shape[1], iH = (int)input.Shape[2], iW = (int)input.Shape[3];
        int oH = (int)output.Shape[2], oW = (int)output.Shape[3];
        VulkanBuffer inBuf = GetBuffer(input);
        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            string shader = "maxpool2d" + DtypeSuffix(output.DType);
            VulkanKernel k = GetKernel(shader, 2, _default1DSpec);
            Span<byte> pc = stackalloc byte[12 * 4];
            BinaryWriteUInt(pc, 0, (uint)N);
            BinaryWriteUInt(pc, 4, (uint)C);
            BinaryWriteUInt(pc, 8, (uint)iH);
            BinaryWriteUInt(pc, 12, (uint)iW);
            BinaryWriteUInt(pc, 16, (uint)oH);
            BinaryWriteUInt(pc, 20, (uint)oW);
            BinaryWriteUInt(pc, 24, (uint)kernelH);
            BinaryWriteUInt(pc, 28, (uint)kernelW);
            BinaryWriteUInt(pc, 32, (uint)strideH);
            BinaryWriteUInt(pc, 36, (uint)strideW);
            BinaryWriteUInt(pc, 40, (uint)padH);
            BinaryWriteUInt(pc, 44, (uint)padW);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(output.ElementCount, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan MaxPool2D dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    public void Conv2dDepthwise(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int strideH, int strideW, int padH, int padW)
    {
        if (output.DType != DType.F32 && output.DType != DType.F16)
            throw new NotSupportedException($"Vulkan Conv2dDepthwise supports F32/F16 output — got {output.DType}.");
        if (weight.Shape.Rank != 4 || weight.Shape[1] != 1)
            throw new ArgumentException($"Conv2dDepthwise weight must be [C, 1, kH, kW]; got {weight.Shape}.");
        using OpScope _op = EnterOp();
        int N = (int)input.Shape[0], C = (int)input.Shape[1], iH = (int)input.Shape[2], iW = (int)input.Shape[3];
        int oH = (int)output.Shape[2], oW = (int)output.Shape[3];
        int kH = (int)weight.Shape[2], kW = (int)weight.Shape[3];
        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer wBuf = GetBuffer(weight);
        // The bias binding is always populated; when there is no bias the input buffer is bound as a
        // never-read placeholder and the hasBias push-constant flag gates the shader read.
        VulkanBuffer biasBuf = bias is null ? inBuf : GetBuffer(bias);
        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            string shader = "depthwise_conv2d" + DtypeSuffix(output.DType);
            VulkanKernel k = GetKernel(shader, 4, _default1DSpec);
            Span<byte> pc = stackalloc byte[13 * 4];
            BinaryWriteUInt(pc, 0, bias is null ? 0u : 1u);
            BinaryWriteUInt(pc, 4, (uint)N);
            BinaryWriteUInt(pc, 8, (uint)C);
            BinaryWriteUInt(pc, 12, (uint)iH);
            BinaryWriteUInt(pc, 16, (uint)iW);
            BinaryWriteUInt(pc, 20, (uint)oH);
            BinaryWriteUInt(pc, 24, (uint)oW);
            BinaryWriteUInt(pc, 28, (uint)kH);
            BinaryWriteUInt(pc, 32, (uint)kW);
            BinaryWriteUInt(pc, 36, (uint)strideH);
            BinaryWriteUInt(pc, 40, (uint)strideW);
            BinaryWriteUInt(pc, 44, (uint)padH);
            BinaryWriteUInt(pc, 48, (uint)padW);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, wBuf.Handle, biasBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(output.ElementCount, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan Conv2dDepthwise dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    #endregion

    #region GPU-residency closure (Phase 3)
    // These three IBackend members had no VulkanBackend override at all — every call fell through to
    // IBackend's CPU-loop default, which reads .DataPointer directly. On a GPU-resident tensor that
    // forces a full device-sync + host readback (see GetD2hSyncCount), and the NEXT GPU op that needs
    // the result pays a fresh H2D re-upload — measured as the dominant contributor to a synthetic LLM
    // decode step's residency cost (Measure_LlmDecodeStep_ResidencyVsDispatchOverhead): SliceLastDim
    // alone accounted for 2 of the 5 D2H syncs/step (QKV split + GluActivate's internal gate/up split),
    // and ApplyRope/KvCacheAppend are the other links in the same q/k/v-prep chain.

    public void SliceLastDim(Tensor output, Tensor input, int offset)
    {
        using OpScope _op = EnterOp();
        int inDim = (int)input.Shape[input.Shape.Rank - 1];
        long rows = input.ElementCount / inDim;
        int outDim = (int)(output.ElementCount / rows);
        VulkanBuffer inBuf = GetBuffer(input);
        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            string shader = "slice_last_dim" + DtypeSuffix(output.DType);
            VulkanKernel k = GetKernel(shader, 2, _default1DSpec);
            Span<byte> pc = stackalloc byte[4 * 4];
            BinaryWriteUInt(pc, 0, (uint)inDim);
            BinaryWriteUInt(pc, 4, (uint)outDim);
            BinaryWriteUInt(pc, 8, (uint)offset);
            BinaryWriteUInt(pc, 12, (uint)output.ElementCount);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(output.ElementCount, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan SliceLastDim dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    public void SliceRows(Tensor output, Tensor input, int rowOffset)
    {
        using OpScope _op = EnterOp();
        int dim = (int)output.Shape[output.Shape.Rank - 1];
        long elemOffset = (long)rowOffset * dim;
        long total = output.ElementCount;
        VulkanBuffer inBuf = GetBuffer(input);
        ulong outBytes = (ulong)(total * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            string shader = "slice_rows" + DtypeSuffix(output.DType);
            VulkanKernel k = GetKernel(shader, 2, _default1DSpec);
            Span<byte> pc = stackalloc byte[2 * 4];
            BinaryWriteUInt(pc, 0, (uint)elemOffset);
            BinaryWriteUInt(pc, 4, (uint)total);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(total, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan SliceRows dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    public void ApplyRope(Tensor q, Tensor k, Tensor cos, Tensor sin)
    {
        using OpScope _op = EnterOp();
        int batch = (int)q.Shape[0], seqLen = (int)q.Shape[1], numHeads = (int)q.Shape[2], headDim = (int)q.Shape[3];
        VulkanBuffer qBuf = GetBuffer(q);
        VulkanBuffer kBuf = GetBuffer(k);
        VulkanBuffer cosBuf = GetBuffer(cos);
        VulkanBuffer sinBuf = GetBuffer(sin);
        try
        {
            string shader = "apply_rope" + DtypeSuffix(q.DType);
            VulkanKernel kernel = GetKernel(shader, 4, _default1DSpec);
            Span<byte> pc = stackalloc byte[4 * 4];
            BinaryWriteUInt(pc, 0, (uint)batch);
            BinaryWriteUInt(pc, 4, (uint)seqLen);
            BinaryWriteUInt(pc, 8, (uint)numHeads);
            BinaryWriteUInt(pc, 12, (uint)headDim);
            long total = (long)batch * seqLen * numHeads * (headDim / 2);
            Span<ulong> bufs = stackalloc ulong[] { qBuf.Handle, kBuf.Handle, cosBuf.Handle, sinBuf.Handle };
            Dispatch(kernel, bufs, pc, GroupCount(total, LocalX1D));
            // In-place on q/k's EXISTING buffers — re-cache so a later reader doesn't see the pre-rope
            // values via a stale activation-cache entry (or force a redundant re-upload if q/k weren't
            // cached at all yet).
            CacheOutput(q, qBuf);
            CacheOutput(k, kBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan ApplyRope dispatch failed", ex);
            throw;
        }
    }

    // Real GPU dispatch, NOT the IBackend CPU-loop default: x here is GPU-resident (FluxRope/Krea2's
    // pre-permute Q/K), so falling through to the default's raw x.DataPointer loop either reads stale
    // zeroed host memory or, worse, AccessViolationExceptions when the tensor's host mirror was never
    // synced/allocated the way the default assumes. Same category as apply_rope/kv_cache_append above.
    public void WanRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int seqLen, int heads, int headDim)
    {
        using OpScope _op = EnterOp();
        VulkanBuffer xBuf = GetBuffer(x);
        VulkanBuffer cosBuf = GetBuffer(cos);
        VulkanBuffer sinBuf = GetBuffer(sin);
        try
        {
            string shader = "wan_rope_interleaved" + DtypeSuffix(x.DType);
            VulkanKernel kernel = GetKernel(shader, 3, _default1DSpec);
            Span<byte> pc = stackalloc byte[3 * 4];
            BinaryWriteUInt(pc, 0, (uint)seqLen);
            BinaryWriteUInt(pc, 4, (uint)heads);
            BinaryWriteUInt(pc, 8, (uint)headDim);
            long total = (long)seqLen * heads * (headDim / 2);
            Span<ulong> bufs = stackalloc ulong[] { xBuf.Handle, cosBuf.Handle, sinBuf.Handle };
            Dispatch(kernel, bufs, pc, GroupCount(total, LocalX1D));
            // In-place on x's EXISTING buffer — re-cache so a later reader doesn't see the pre-rope
            // values via a stale activation-cache entry.
            CacheOutput(x, xBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan WanRopeInterleaved dispatch failed", ex);
            throw;
        }
    }

    // Real GPU dispatch, NOT the IBackend CPU-loop default: the default throws on non-F32 (Krea2's
    // GQA K/V is F16), and even for F32 it would force a D2H/H2D round trip on a GPU-resident tensor.
    public void RepeatKvHeads(Tensor output, Tensor input, int kvHeads, int groupSize)
    {
        using OpScope _op = EnterOp();
        int batch = (int)input.Shape[0], seqLen = (int)input.Shape[2], headDim = (int)input.Shape[3];
        VulkanBuffer inBuf = GetBuffer(input);
        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            string shader = "repeat_kv_heads" + DtypeSuffix(input.DType);
            VulkanKernel kernel = GetKernel(shader, 2, _default1DSpec);
            Span<byte> pc = stackalloc byte[5 * 4];
            BinaryWriteUInt(pc, 0, (uint)kvHeads);
            BinaryWriteUInt(pc, 4, (uint)groupSize);
            BinaryWriteUInt(pc, 8, (uint)seqLen);
            BinaryWriteUInt(pc, 12, (uint)headDim);
            long total = (long)batch * kvHeads * groupSize * seqLen * headDim;
            BinaryWriteUInt(pc, 16, (uint)total);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, outBuf.Handle };
            Dispatch(kernel, bufs, pc, GroupCount(total, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan RepeatKvHeads dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    public void KvCacheAppend(Tensor buffer, Tensor newKv, int offset)
    {
        using OpScope _op = EnterOp();
        int heads = (int)buffer.Shape[1], maxSeq = (int)buffer.Shape[2], headDim = (int)buffer.Shape[3];
        int tNew = (int)newKv.Shape[2];
        VulkanBuffer bufBuf = GetBuffer(buffer);
        VulkanBuffer newBuf = GetBuffer(newKv);
        try
        {
            string shader = "kv_cache_append" + DtypeSuffix(buffer.DType);
            VulkanKernel k = GetKernel(shader, 2, _default1DSpec);
            Span<byte> pc = stackalloc byte[5 * 4];
            BinaryWriteUInt(pc, 0, (uint)heads);
            BinaryWriteUInt(pc, 4, (uint)maxSeq);
            BinaryWriteUInt(pc, 8, (uint)headDim);
            BinaryWriteUInt(pc, 12, (uint)tNew);
            BinaryWriteUInt(pc, 16, (uint)offset);
            long total = (long)heads * tNew * headDim;
            Span<ulong> bufs = stackalloc ulong[] { bufBuf.Handle, newBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(total, LocalX1D));
            // In-place on buffer's EXISTING (persistent, caller-owned) buffer — re-cache so the next
            // reader sees the appended data instead of a stale pre-append activation-cache entry.
            CacheOutput(buffer, bufBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan KvCacheAppend dispatch failed", ex);
            throw;
        }
    }

    #endregion

    #region Data movement

    public unsafe void CopyTo(Tensor destination, Tensor source)
    {
        // Fast path: source is ALREADY GPU-resident (weight or activation cache) — copy device-to-device
        // instead of forcing a D2H sync (source) + H2D re-upload (destination, on whatever GPU op reads
        // it next). TryGetCached never uploads on a miss, so a genuinely host-only source still falls
        // through to the plain memcpy below — that path is not worth promoting to the GPU.
        if (_xfer.TryGetCached(source, out VulkanBuffer? srcBuf) && srcBuf is not null)
        {
            using OpScope _op = EnterOp();
            ulong byteCount = (ulong)(source.ElementCount * source.DType.SizeInBytes);
            VulkanBuffer dstBuf = _xfer.AllocateDevice(byteCount);
            try
            {
                _stream.RecordCopyAndBarrier(srcBuf.Handle, dstBuf.Handle, byteCount,
                    postStage: VkPipelineStageFlags2.ComputeShader,
                    postAccess: VkAccessFlags2.ShaderStorageRead);
                _dispatchesSinceSubmit++;
                _dispatchesThisOp++;
                if (_dispatchesSinceSubmit >= FlushThreshold && _opNestingDepth == 0) DrainAndFlush();
                CacheOutput(destination, dstBuf);
            }
            catch (Exception ex)
            {
                Logs.Error("Vulkan CopyTo (device-to-device) dispatch failed", ex);
                dstBuf.Dispose();
                throw;
            }
            return;
        }

        long hostByteCount = source.ElementCount * source.DType.SizeInBytes;
        Buffer.MemoryCopy(source.DataPointer, destination.DataPointer, hostByteCount, hostByteCount);
    }

    /// <summary>In-place flow-match Euler step with the CFG combine folded in: <c>z += (guidance·pos + (1-guidance)·neg)·delta</c>. Overrides <c>IBackend</c>'s CPU-loop default (mirrors <c>CudaBackend.CfgEulerStep</c>) — the default reads/writes <c>z.DataPointer</c> directly, which for a GPU-resident z (Krea2's fixed per-step latent, <c>_latentFixed</c>) forces a full D2H sync + evicts it from the activation cache every step: a real, previously-undiscovered perf cost on every Krea2 Vulkan generation (not just step-graph mode — this call happens once per denoise step regardless), and the exact bug that broke step-graph capture (found via a genuine `HARTSY_DIT_GRAPH=1` Krea2 run: capture failed with a cache-miss on the patchified latent, because THIS default eviction between steps left it uncached by the time the next capture attempt read it).</summary>
    public unsafe void CfgEulerStep(Tensor z, Tensor pos, Tensor neg, float guidance, float delta)
    {
        using OpScope _op = EnterOp();
        VulkanBuffer zBuf = GetBuffer(z);
        VulkanBuffer posBuf = GetBuffer(pos);
        VulkanBuffer negBuf = GetBuffer(neg);

        VulkanKernel k = GetKernel("cfg_euler", 3, _default1DSpec);
        Span<byte> pc = stackalloc byte[3 * 4];
        BinaryWriteUInt(pc, 0, (uint)z.ElementCount);
        BinaryWriteFloat(pc, 4, guidance);
        BinaryWriteFloat(pc, 8, delta);
        Span<ulong> bufs = stackalloc ulong[] { zBuf.Handle, posBuf.Handle, negBuf.Handle };
        Dispatch(k, bufs, pc, GroupCount(z.ElementCount, LocalX1D));

        // In-place re-assert (the CopyInto/CudaBackend.CfgEulerStep pattern): z keeps this buffer. Clear stale
        // callbacks BEFORE re-caching so the old sync callback can't free the buffer we're keeping resident.
        z._gpuSyncCallback = null;
        z._gpuDisposeCallback = null;
        CacheOutput(z, zBuf);

        _xfer.FreeDevice(posBuf);
        _xfer.FreeDevice(negBuf);
    }

    /// <summary>Device copy into <paramref name="dst"/>'s EXISTING buffer (address-preserving — the captured-graph boundary refresh; mirrors <c>CudaBackend.CopyInto</c>). First call materializes a device buffer for dst; subsequent calls reuse it. Host src is uploaded; device src is device-to-device copied. Capture-aware: <c>Krea2Transformer.ForwardPatched</c> issues this as the graph's LAST captured op (the velocity-output boundary write), so it must record onto the capture buffer exactly like a kernel dispatch when capturing.</summary>
    public unsafe void CopyInto(Tensor dst, Tensor src)
    {
        if (_capturingStepGraph)
        {
            ulong capByteCount = (ulong)(dst.ElementCount * dst.DType.SizeInBytes);
            VulkanBuffer capDstBuf = _xfer.TryGetCached(dst, out VulkanBuffer? capExisting) && capExisting is not null
                ? capExisting
                : _xfer.AllocateDevice(capByteCount);
            VulkanBuffer capSrcBuf = GetBuffer(src);
            VulkanCommandStream.RecordCopyAndBarrierOn(_stepGraph!.RecordingBuffer, capSrcBuf.Handle, capDstBuf.Handle, capByteCount,
                postStage: VkPipelineStageFlags2.ComputeShader,
                postAccess: VkAccessFlags2.ShaderStorageRead);
            dst._gpuSyncCallback = null;
            dst._gpuDisposeCallback = null;
            CacheOutput(dst, capDstBuf);
            return;
        }

        using OpScope _op = EnterOp();
        ulong byteCount = (ulong)(dst.ElementCount * dst.DType.SizeInBytes);
        VulkanBuffer dstBuf = _xfer.TryGetCached(dst, out VulkanBuffer? existingDst) && existingDst is not null
            ? existingDst
            : _xfer.AllocateDevice(byteCount);
        VulkanBuffer srcBuf = GetBuffer(src);
        _stream.RecordCopyAndBarrier(srcBuf.Handle, dstBuf.Handle, byteCount,
            postStage: VkPipelineStageFlags2.ComputeShader,
            postAccess: VkAccessFlags2.ShaderStorageRead);
        _dispatchesSinceSubmit++;
        _dispatchesThisOp++;
        if (_dispatchesSinceSubmit >= FlushThreshold && _opNestingDepth == 0) DrainAndFlush();
        // In-place re-assert (the CfgEulerStep pattern): dst keeps this buffer across the copy. Clear stale
        // callbacks BEFORE re-caching — skipping this reproduces a stale-callback bug on the next dispose/sync.
        dst._gpuSyncCallback = null;
        dst._gpuDisposeCallback = null;
        CacheOutput(dst, dstBuf);
    }

    public unsafe void Fill(Tensor tensor, float value)
    {
        long count = tensor.ElementCount;
        if (tensor.DType == DType.F32)
        {
            float* p = (float*)tensor.DataPointer;
            for (long i = 0; i < count; i++) p[i] = value;
        }
        else if (tensor.DType == DType.F16)
        {
            Half h = (Half)value;
            Half* p = (Half*)tensor.DataPointer;
            for (long i = 0; i < count; i++) p[i] = h;
        }
        else
        {
            throw new NotSupportedException($"VulkanBackend.Fill: dtype {tensor.DType.Name} not supported.");
        }
    }

    #endregion

    #region Audio (not supported on this backend)

    public void Fft(Tensor output, Tensor input)
        => throw new NotSupportedException("VulkanBackend does not support FFT — use the CPU backend for audio preprocessing.");

    public void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window)
        => throw new NotSupportedException("VulkanBackend does not support STFT — use the CPU backend.");

    public void MelFilterbank(Tensor output, Tensor input, Tensor filters)
        => throw new NotSupportedException("VulkanBackend does not support MelFilterbank — use the CPU backend.");

    #endregion

    #region Casts

    public unsafe void CastToF16(Tensor output, Tensor input)
    {
        if ((input.DType == DType.F32 || input.DType == DType.F8E4M3) && output.DType == DType.F16)
        {
            VulkanBuffer src = GetBuffer(input);
            (VulkanBuffer cast, _) = CastIfNeeded(input, src, DType.F16);
            // CastIfNeeded returns owned=cast when a conversion happened; just promote to activation cache.
            CacheOutput(output, cast);
            return;
        }

        // CPU fallback for paths the Vulkan kernel suite doesn't yet cover.
        if (input.DType == DType.F32 && output.DType == DType.F16)
        {
            float* src = (float*)input.DataPointer;
            Half* dst = (Half*)output.DataPointer;
            for (long i = 0; i < input.ElementCount; i++) dst[i] = (Half)src[i];
            return;
        }
        throw new NotSupportedException($"VulkanBackend.CastToF16: {input.DType.Name} -> F16 not implemented.");
    }

    public unsafe void CastToF32(Tensor output, Tensor input)
    {
        if (input.DType == DType.F16 && output.DType == DType.F32)
        {
            VulkanBuffer src = GetBuffer(input);
            ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
            VulkanBuffer dst = _xfer.AllocateDevice(outBytes);
            try
            {
                VulkanKernel k = GetKernel("cast_f16_f32", 2, _default1DSpec);
                Span<byte> pc = stackalloc byte[4]; BinaryWriteUInt(pc, 0, (uint)input.ElementCount);
                Span<ulong> bufs = stackalloc ulong[] { src.Handle, dst.Handle };
                Dispatch(k, bufs, pc, GroupCount(input.ElementCount, LocalX1D));
                CacheOutput(output, dst);
            }
            catch (Exception ex)
            {
                Logs.Error("Vulkan CastToF32 dispatch failed", ex);
                dst.Dispose();
                throw;
            }
        }
        else
        {
            Half* src = (Half*)input.DataPointer;
            float* dst = (float*)output.DataPointer;
            for (long i = 0; i < input.ElementCount; i++) dst[i] = (float)src[i];
        }
    }

    #endregion

    #region LLM decode-graph (Phase 6)
    // Vulkan-native analog of CudaBackend's graph-decode device state (CudaBackend.cs ~L6580-6800): the
    // per-token host work a captured/replayed step CANNOT tolerate (RoPE's Math.Cos/Sin + H2D every step,
    // the embedding lookup, argmax's D2H-then-H2D chain) converted to one-time device-resident precompute
    // plus small persistent "control" buffers whose ADDRESSES stay fixed across a decode loop's iterations
    // and whose CONTENTS are refreshed between steps — exactly CUDA's own design principle, just without
    // CUDA Graph capture/replay itself (that mechanism is Phase 6's VulkanStepGraph, tracked separately;
    // these are the leaf ops it will eventually replay). Fully usable standalone today in an eager loop —
    // a caller can drive AllocDeviceTokenId/WriteDeviceTokenId/EmbedGatherDecodeStep/RopeApplyDecodeStep/
    // FlashAttentionDev/KvCacheAppendDev/ArgMaxInto every step by hand, at greatly reduced D2H sync count
    // vs. the eager ApplyRope/KvCacheAppend/host-argmax path, even before a graph wraps them.

    /// <summary>Settable, default OFF: several real production call sites (Qwen35Model, GenericTransformer, SsmGenerationPipeline, CsmModel, MusicGenDecoder) gate straight onto the graph-decode path the instant this is true, with no separate opt-in of their own. Flip only after a real end-to-end decode-loop parity test (not just the per-op unit tests below) confirms this device-buffer path matches the eager path on a real model — matching the same staged-rollout discipline used for <see cref="EnableInt8Linear"/>, and for the same reason: a capability flag this many callers trust unconditionally must not flip on unit-level confidence alone.</summary>
    public bool GraphDecodeSupported { get; set; }

    /// <summary>Backs every "persistent 1-int/N-int device control buffer" handle below: <see cref="VulkanBuffer.Handle"/> (a raw VkBuffer u64, mirroring CUDA's raw-CUdeviceptr-as-ulong convention) keyed back to the owning buffer so it can be freed, written (mapped-or-staged), or read (D2H) by handle alone.</summary>
    private readonly Dictionary<ulong, VulkanBuffer> _scalarBuffers = new();

    /// <summary>Capacity (element count) of each <see cref="AllocDeviceHistory"/> buffer, since <see cref="AppendTokenHistoryStep"/>'s IBackend signature carries only the handle, not the size.</summary>
    private readonly Dictionary<ulong, int> _historyCapacity = new();

    private ulong AllocScalarBuffer(int elementCount)
    {
        VulkanBuffer buf = _xfer.AllocateDevice((ulong)(Math.Max(1, elementCount) * sizeof(int)));
        _scalarBuffers[buf.Handle] = buf;
        return buf.Handle;
    }

    private void FreeScalarBuffer(ulong handle)
    {
        if (handle == 0) return;
        if (_scalarBuffers.Remove(handle, out VulkanBuffer? buf))
        {
            _historyCapacity.Remove(handle);
            _xfer.FreeDevice(buf);
        }
    }

    /// <summary>Writes <paramref name="values"/> into a scalar control buffer outside any capture region — the "refresh fixed-address contents between replays" half of the decode-graph design.</summary>
    /// <remarks>MUST sync first: <see cref="Dispatch"/> batches multiple dispatches into one command
    /// buffer, submitted only at <c>FlushThreshold</c> or on an explicit sync — a dispatch that reads this
    /// buffer may still be sitting unsubmitted when this is called. Without waiting for it to actually
    /// execute, this write can race ahead and land before the GPU ever reads the OLD value the pending
    /// dispatch needed, silently corrupting decode-step state (caught by
    /// <c>ApplyRepetitionPenaltyStep_MatchesHfConventionWithRepeats</c> on llvmpipe: 5 distinct token ids
    /// written in a loop, each immediately followed by a dispatch reading the CURRENT id — every dispatch
    /// ended up reading the LAST id written, since none had actually run before the next overwrite).
    /// CUDA avoids this cost entirely via an ASYNC copy on the same in-order stream (no host wait needed —
    /// stream ordering alone guarantees the old value is consumed first); a Vulkan equivalent would record
    /// the write as a <c>vkCmdUpdateBuffer</c> into the SAME batched command buffer instead of a host-side
    /// write, keeping it stream-ordered without blocking. Correctness-first for now, since this executes
    /// once per decode step, not per dispatch — the real optimization target once a full decode loop's
    /// wall-clock is being tuned, not before.</remarks>
    private unsafe void WriteScalarBuffer(ulong handle, ReadOnlySpan<int> values)
    {
        if (handle == 0 || !_scalarBuffers.TryGetValue(handle, out VulkanBuffer? buf)) return;
        Sync();
        using Tensor scratch = new(new TensorShape(values.Length), DType.I32);
        values.CopyTo(scratch.AsSpan<int>());
        _xfer.Upload(buf, scratch);
    }

    /// <summary>D2H sync read of a 1-int scalar control buffer — the one blocking readback a decode step needs (to hand the sampled token back to CPU orchestration code), waiting on exactly the work queued so far rather than the whole stream's backlog.</summary>
    private unsafe int ReadScalarBufferInt(ulong handle)
    {
        if (handle == 0 || !_scalarBuffers.TryGetValue(handle, out VulkanBuffer? buf))
            throw new NotSupportedException("ReadScalarBufferInt called with an unallocated buffer.");
        _stream.WaitIdleHost();
        int v;
        _xfer.DownloadToHost((nint)(&v), buf, sizeof(int));
        return v;
    }

    // ── Device-side decode position ─────────────────────────────────────────────────────────────────────

    public ulong AllocDevicePos() => AllocScalarBuffer(2);

    public void WriteDevicePos(ulong handle, int kvLen, int qOffset) => WriteScalarBuffer(handle, stackalloc int[] { kvLen, qOffset });

    public void FreeDevicePos(ulong handle) => FreeScalarBuffer(handle);

    // ── Device token-id (embed input / argmax output) ───────────────────────────────────────────────────

    public ulong AllocDeviceTokenId() => AllocScalarBuffer(1);

    public void WriteDeviceTokenId(ulong handle, int tokenId) => WriteScalarBuffer(handle, stackalloc int[] { tokenId });

    public void FreeDeviceTokenId(ulong handle) => FreeScalarBuffer(handle);

    public int ReadDeviceTokenId(ulong handle) => ReadScalarBufferInt(handle);

    // ── RoPE table build + decode-step apply ────────────────────────────────────────────────────────────

    public unsafe (Tensor cos, Tensor sin) BuildRopeTableDevice(int maxPos, int headDim, int rotaryDim, float theta,
        RopeScaling scaling, bool splitHalfPartial = false)
    {
        // Host-side precompute, identical math/layout to CudaBackend.BuildRopeTableDevice (itself matching
        // GenericTransformer.BuildRope) — just for every position up front instead of one per-step tensor.
        int rdim = rotaryDim > 0 && rotaryDim < headDim ? rotaryDim : headDim;
        int rotHalf = rdim / 2;
        int halfDim = headDim / 2;
        (double[] invFreq, double mscale) = RopeFrequencyBuilder.Build(rdim, theta, scaling, maxPos);
        Tensor cos = new(new TensorShape(maxPos, headDim), DType.F32);
        Tensor sin = new(new TensorShape(maxPos, headDim), DType.F32);
        float* pc = (float*)cos.DataPointer;
        float* ps = (float*)sin.DataPointer;
        int dupStride = splitHalfPartial ? rotHalf : halfDim;
        for (int pos = 0; pos < maxPos; pos++)
        {
            long baseOff = (long)pos * headDim;
            for (int i = 0; i < headDim; i++) { pc[baseOff + i] = 1f; ps[baseOff + i] = 0f; }
            for (int i = 0; i < rotHalf; i++)
            {
                double angle = pos * invFreq[i];
                float c = (float)(Math.Cos(angle) * mscale);
                float s = (float)(Math.Sin(angle) * mscale);
                pc[baseOff + i] = c; pc[baseOff + i + dupStride] = c;
                ps[baseOff + i] = s; ps[baseOff + i + dupStride] = s;
            }
        }
        PreloadWeights(new[] { cos, sin });
        return (cos, sin);
    }

    public void RopeApplyDecodeStep(Tensor x, Tensor cosTable, Tensor sinTable, int rotaryDim, bool interleaved, ulong devicePos)
    {
        using OpScope _op = EnterOp();
        if (devicePos == 0 || !_scalarBuffers.TryGetValue(devicePos, out VulkanBuffer? posBuf)
            || x.DType != DType.F32 || cosTable.DType != DType.F32 || sinTable.DType != DType.F32)
        {
            ((IBackend)this).RopeApplyDecodeStep(x, cosTable, sinTable, rotaryDim, interleaved, devicePos);
            return;
        }
        int headDim = (int)x.Shape[x.Shape.Rank - 1];
        int numHeads = (int)(x.ElementCount / headDim);
        int rdim = rotaryDim > 0 && rotaryDim < headDim ? rotaryDim : headDim;
        VulkanBuffer xBuf = GetBuffer(x);
        VulkanBuffer cosBuf = GetBuffer(cosTable);
        VulkanBuffer sinBuf = GetBuffer(sinTable);
        try
        {
            string shader = interleaved ? "rope_decode_step_interleaved_f32" : "rope_decode_step_splithalf_f32";
            VulkanKernel k = GetKernel(shader, 4, _default1DSpec);
            Span<byte> pc = stackalloc byte[3 * 4];
            BinaryWriteUInt(pc, 0, (uint)numHeads);
            BinaryWriteUInt(pc, 4, (uint)headDim);
            BinaryWriteUInt(pc, 8, (uint)rdim);
            long total = (long)numHeads * (rdim / 2);
            Span<ulong> bufs = stackalloc ulong[] { xBuf.Handle, cosBuf.Handle, sinBuf.Handle, posBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(total, LocalX1D));
            CacheOutput(x, xBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan RopeApplyDecodeStep dispatch failed", ex);
            throw;
        }
    }

    // ── Embed gather + argmax ───────────────────────────────────────────────────────────────────────────

    public void EmbedGatherDecodeStep(Tensor output, Tensor embedTable, ulong tokenId)
    {
        using OpScope _op = EnterOp();
        if (tokenId == 0 || !_scalarBuffers.TryGetValue(tokenId, out VulkanBuffer? tokBuf)
            || output.DType != DType.F32 || embedTable.DType != DType.F32)
        {
            throw new NotSupportedException("EmbedGatherDecodeStep requires an F32 embed table and a valid device token-id buffer.");
        }
        int hidden = (int)output.ElementCount;
        VulkanBuffer embBuf = GetBuffer(embedTable);
        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            VulkanKernel k = GetKernel("embed_gather_decode", 3, _default1DSpec);
            Span<byte> pc = stackalloc byte[4];
            BinaryWriteUInt(pc, 0, (uint)hidden);
            Span<ulong> bufs = stackalloc ulong[] { outBuf.Handle, embBuf.Handle, tokBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(hidden, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan EmbedGatherDecodeStep dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    public void ArgMaxInto(ulong outputTokenId, Tensor input)
    {
        using OpScope _op = EnterOp();
        if (outputTokenId == 0 || !_scalarBuffers.TryGetValue(outputTokenId, out VulkanBuffer? outBuf)
            || input.DType != DType.F32)
        {
            throw new NotSupportedException("ArgMaxInto requires F32 input and a valid device token-id buffer.");
        }
        int c = (int)input.Shape[input.Shape.Rank - 1];
        VulkanBuffer inBuf = GetBuffer(input);
        try
        {
            VulkanKernel k = GetKernel("argmax_lastdim", 2, _default1DSpec);
            Span<byte> pc = stackalloc byte[4];
            BinaryWriteUInt(pc, 0, (uint)c);
            Span<ulong> bufs = stackalloc ulong[] { outBuf.Handle, inBuf.Handle };
            Dispatch(k, bufs, pc, 1);   // single workgroup: the reduction spans the whole row itself
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan ArgMaxInto dispatch failed", ex);
            throw;
        }
    }

    // ── Repetition-penalty history ───────────────────────────────────────────────────────────────────────

    public ulong AllocDeviceHistory(int capacity)
    {
        ulong handle = AllocScalarBuffer(Math.Max(1, capacity));
        _historyCapacity[handle] = Math.Max(1, capacity);
        return handle;
    }

    public void FreeDeviceHistory(ulong handle) => FreeScalarBuffer(handle);

    public ulong AllocDeviceCounter() => AllocScalarBuffer(1);

    public void FreeDeviceCounter(ulong handle) => FreeScalarBuffer(handle);

    public void WriteDeviceCounter(ulong handle, int value) => WriteScalarBuffer(handle, stackalloc int[] { value });

    public void AppendTokenHistoryStep(ulong history, ulong historyCount, ulong tokenId)
    {
        using OpScope _op = EnterOp();
        if (history == 0 || historyCount == 0 || tokenId == 0) return;
        if (!_scalarBuffers.TryGetValue(history, out VulkanBuffer? histBuf)
            || !_scalarBuffers.TryGetValue(historyCount, out VulkanBuffer? countBuf)
            || !_scalarBuffers.TryGetValue(tokenId, out VulkanBuffer? tokBuf))
        {
            return;
        }
        int capacity = _historyCapacity.TryGetValue(history, out int cap) ? cap : int.MaxValue;
        try
        {
            VulkanKernel k = GetKernel("history_append", 3, _default1DSpec);
            Span<byte> pc = stackalloc byte[4];
            BinaryWriteUInt(pc, 0, (uint)capacity);
            Span<ulong> bufs = stackalloc ulong[] { histBuf.Handle, countBuf.Handle, tokBuf.Handle };
            Dispatch(k, bufs, pc, 1);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan AppendTokenHistoryStep dispatch failed", ex);
            throw;
        }
    }

    public void ApplyRepetitionPenaltyStep(Tensor logits, ulong history, ulong historyCount, float penalty)
    {
        using OpScope _op = EnterOp();
        if (history == 0 || historyCount == 0 || logits.DType != DType.F32
            || !_scalarBuffers.TryGetValue(history, out VulkanBuffer? histBuf)
            || !_scalarBuffers.TryGetValue(historyCount, out VulkanBuffer? countBuf))
        {
            throw new NotSupportedException("ApplyRepetitionPenaltyStep requires F32 logits and valid device history buffers.");
        }
        int vocabSize = (int)logits.Shape[logits.Shape.Rank - 1];
        VulkanBuffer logitsBuf = GetBuffer(logits);
        try
        {
            VulkanKernel k = GetKernel("repetition_penalty", 3, _default1DSpec);
            Span<byte> pc = stackalloc byte[8];
            BinaryWriteUInt(pc, 0, (uint)vocabSize);
            BinaryWriteFloat(pc, 4, penalty);
            Span<ulong> bufs = stackalloc ulong[] { logitsBuf.Handle, histBuf.Handle, countBuf.Handle };
            Dispatch(k, bufs, pc, 1);
            CacheOutput(logits, logitsBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan ApplyRepetitionPenaltyStep dispatch failed", ex);
            throw;
        }
    }

    // ── KV-cache / flash-attention device-position variants (Phase 6d) ─────────────────────────────────
    // Both `offset`/`kvLen`/`qOffset` parameters below are the CALLER's placeholder host values (real
    // callers — see GenericTransformer.ForwardGraphDecodeStep — pass literal 0s here on purpose): the
    // ACTUAL write slot / attend length must come from devicePos, exactly like RopeApplyDecodeStep. A
    // passthrough that forwards `offset`/`kvLen`/`qOffset` unchanged would silently corrupt the KV cache
    // (every step writing/attending at slot 0) the moment GraphDecodeSupported flips on — caught in review
    // before it shipped, not by the parity test that would have found it after.

    public void KvCacheAppendDev(Tensor buffer, Tensor newKv, int offset, ulong devicePos)
    {
        using OpScope _op = EnterOp();
        if (devicePos == 0 || !_scalarBuffers.TryGetValue(devicePos, out VulkanBuffer? posBuf)
            || buffer.DType != DType.F32 || newKv.DType != DType.F32)
        {
            KvCacheAppend(buffer, newKv, offset);
            return;
        }
        int heads = (int)buffer.Shape[1], maxSeq = (int)buffer.Shape[2], headDim = (int)buffer.Shape[3];
        int tNew = (int)newKv.Shape[2];
        VulkanBuffer bufBuf = GetBuffer(buffer);
        VulkanBuffer newBuf = GetBuffer(newKv);
        try
        {
            VulkanKernel k = GetKernel("kv_cache_append_dev", 3, _default1DSpec);
            Span<byte> pc = stackalloc byte[4 * 4];
            BinaryWriteUInt(pc, 0, (uint)heads);
            BinaryWriteUInt(pc, 4, (uint)maxSeq);
            BinaryWriteUInt(pc, 8, (uint)headDim);
            BinaryWriteUInt(pc, 12, (uint)tNew);
            long total = (long)heads * tNew * headDim;
            Span<ulong> bufs = stackalloc ulong[] { bufBuf.Handle, newBuf.Handle, posBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(total, LocalX1D));
            CacheOutput(buffer, bufBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan KvCacheAppendDev dispatch failed", ex);
            throw;
        }
    }

    /// <summary>Self-attention whose kvLen/qOffset come from the device position buffer — the <c>FlashAttentionDev</c> counterpart to <see cref="KvCacheAppendDev"/>. Same scope boundary as <see cref="FlashAttention"/> (head dim &lt;= <see cref="FlashMaxHeadDim"/>, no softcap/sink/ALiBi, no mask — <c>FlashAttentionDev</c>'s signature has none) — anything outside that falls back to the IBackend default, which forwards the CALLER's placeholder <paramref name="kvLen"/>/<paramref name="qOffset"/> host ints straight to <see cref="FlashAttention"/>. That default is ONLY correct when those placeholders happen to be right for the eager (non-graph) case; real graph-decode callers pass 0s expecting devicePos to be authoritative, so anything reaching that fallback with devicePos-dependent placeholders would be silently wrong — scoped narrowly here specifically so it doesn't reach that default in the cases graph decode actually exercises.</summary>
    public unsafe void FlashAttentionDev(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup, bool causal, int qOffset, float scale, ulong devicePos,
        float softcap = 0f, int slidingWindow = 0)
    {
        int headDim = (int)query.Shape[query.Shape.Rank - 1];
        if (devicePos != 0 && _scalarBuffers.ContainsKey(devicePos)
            && softcap == 0f && slidingWindow == 0 && headDim <= FlashMaxHeadDim
            && query.DType == DType.F32 && key.DType == DType.F32 && value.DType == DType.F32 && output.DType == DType.F32)
        {
            using OpScope _op = EnterOp();
            DispatchFlashAttentionDev(output, query, key, value, scale, causal, kvGroup, devicePos);
            return;
        }
        // Falls back to the eager path — correct only if kvLen/qOffset are the real (not placeholder)
        // values; a caller relying on devicePos for a shape this scope excludes (sliding window, softcap,
        // large head dim) is not yet supported on Vulkan's graph-decode path.
        FlashAttention(output, query, key, value, kvLen, kvGroup, causal, qOffset, scale, softcap, sink: null, slidingWindow, alibiSlopes: null);
    }

    private void DispatchFlashAttentionDev(Tensor output, Tensor query, Tensor key, Tensor value, float scale, bool causal, int kvGroup, ulong devicePos)
    {
        int batch = (int)query.Shape[0], hq = (int)query.Shape[1], sq = (int)query.Shape[2], headDim = (int)query.Shape[3];
        VulkanBuffer posBuf = _scalarBuffers[devicePos];
        VulkanBuffer qBuf = GetBuffer(query);
        VulkanBuffer kBuf = GetBuffer(key);
        VulkanBuffer vBuf = GetBuffer(value);
        ulong outBytes = (ulong)((long)batch * hq * sq * headDim * sizeof(float));
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            VulkanKernel k = GetKernel("sdpa_flash_dev_f32", 5, ReadOnlySpan<SpecConstant>.Empty);
            int kvCapacity = (int)key.Shape[2];
            Span<byte> pc = stackalloc byte[10 * 4];
            BinaryWriteUInt(pc, 0, (uint)hq);
            BinaryWriteUInt(pc, 4, (uint)(hq / kvGroup));
            BinaryWriteUInt(pc, 8, (uint)sq);
            BinaryWriteUInt(pc, 12, 0u);            // skv: unused, read from Pos_ instead
            BinaryWriteUInt(pc, 16, (uint)kvCapacity);
            BinaryWriteUInt(pc, 20, (uint)headDim);
            BinaryWriteFloat(pc, 24, scale);
            BinaryWriteUInt(pc, 28, causal ? 1u : 0u);
            BinaryWriteUInt(pc, 32, 0u);            // qOffset: unused, read from Pos_ instead
            BinaryWriteUInt(pc, 36, 0u);             // slidingWindow: excluded from this scope (see caller)
            Span<ulong> bufs = stackalloc ulong[] { qBuf.Handle, kBuf.Handle, vBuf.Handle, posBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, (uint)sq, (uint)hq, (uint)batch);
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan FlashAttentionDev dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    #endregion

    #region Disposal

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        try { _stream.WaitIdleHost(); }
        catch (Exception ex) { Logs.Warning($"Vulkan Dispose: WaitIdleHost failed, continuing teardown: {ex}"); }
        // Dump profiling data (no-op when HARTSYINFERENCE_VK_PROFILE is unset) before tearing down
        // anything that the per-op records reference (op names are strings only, but flush before
        // we lose the device just to be tidy).
        try { _profiler.Dump(); }
        catch (Exception ex) { Logs.Warning($"Vulkan Dispose: profiler dump failed: {ex}"); }
        if (_profiler.IsEnabled && (_coopmatGemmCount + _coopmat2GemmCount + _tiledGemmCount) > 0)
        {
            long total = _coopmatGemmCount + _coopmat2GemmCount + _tiledGemmCount;
            Console.Error.WriteLine(
                $"  GEMM fast-path: coopmat2={_coopmat2GemmCount} ({100.0 * _coopmat2GemmCount / total:F1}%), " +
                $"coopmat={_coopmatGemmCount} ({100.0 * _coopmatGemmCount / total:F1}%), " +
                $"tiled-fallback={_tiledGemmCount} ({100.0 * _tiledGemmCount / total:F1}%)");
            // Per-path host-wall-clock average — the thing that actually determines real wall-clock, distinct
            // from GPU-only execution time (see MeasureGpuTimeMs). Ticks are Stopwatch.GetTimestamp() units;
            // Stopwatch.Frequency converts to seconds.
            double freq = System.Diagnostics.Stopwatch.Frequency;
            if (_coopmat2GemmCount > 0)
                Console.Error.WriteLine($"    coopmat2 avg host-wall: {1000.0 * _coopmat2GemmTicks / freq / _coopmat2GemmCount:F3} ms/call ({1000.0 * _coopmat2GemmTicks / freq:F1} ms total)");
            if (_coopmatGemmCount > 0)
                Console.Error.WriteLine($"    coopmat  avg host-wall: {1000.0 * _coopmatGemmTicks / freq / _coopmatGemmCount:F3} ms/call ({1000.0 * _coopmatGemmTicks / freq:F1} ms total)");
            if (_tiledGemmCount > 0)
                Console.Error.WriteLine($"    tiled    avg host-wall: {1000.0 * _tiledGemmTicks / freq / _tiledGemmCount:F3} ms/call ({1000.0 * _tiledGemmTicks / freq:F1} ms total)");
        }
        // Decode-graph scalar control buffers (Phase 6) aren't in any weight/activation cache _xfer.Dispose()
        // already sweeps — free them directly so a caller who forgot to call FreeDeviceTokenId/FreeDevicePos/
        // etc. doesn't leak a VkBuffer + its backing allocation.
        foreach (VulkanBuffer buf in _scalarBuffers.Values) buf.Dispose();
        _scalarBuffers.Clear();
        _historyCapacity.Clear();
        // Step-graph retained buffers aren't in any weight/activation cache either — release them (the
        // command pool/fence themselves are freed by _stepGraph.Dispose() below) before _xfer tears down.
        _xfer.ReleaseStepGraphRetained();
        _stepGraph?.Dispose();
        _xfer.Dispose();
        _kernels.Dispose();
        _pipelineCache.Dispose();
        _descriptors.Dispose();
        _stream.Dispose();
        _allocator.Dispose();
        _vkDevice.Dispose();
        _instance.Dispose();
        GC.SuppressFinalize(this);
    }

    #endregion
}

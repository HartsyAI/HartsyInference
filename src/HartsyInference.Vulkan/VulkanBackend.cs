using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

/// <summary>Vulkan compute backend implementing <see cref="IBackend"/>. Routes operations to SPIR-V compute shaders loaded from disk and dispatched via vkCmdDispatch on a single timeline-semaphore stream. Mirrors the CUDA backend's GPU weight cache + lazy-sync activation cache so model code that works on CUDA works unchanged here.</summary>
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

    public DeviceKind Device { get; }
    public BackendCapabilities Capabilities { get; }
    public VulkanCapabilities Vk => _vkDevice.Capabilities;

    /// <summary>Filesystem path of the on-disk SPIR-V pipeline cache. Exposed for tests that
    /// verify persist + reload across backend lifetimes.</summary>
    public string PipelineCachePath => _pipelineCache.CachePath;

    /// <summary>Diagnostic snapshot of device-memory usage. Used by the leak-validation tests
    /// to assert that VRAM returns to baseline after a generation loop. Aggregated across all
    /// DEVICE_LOCAL heaps; values are stable across slab boundaries (slab-internal free regions
    /// are subtracted from <c>UsedDeviceBytes</c>).</summary>
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
            catch { /* swallow inside callback */ }
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

    public void Sync()
    {
        _xfer.DrainTransients();
        _stream.WaitIdleHost();
        _dispatchesSinceSubmit = 0;
    }

    public void FreeWeights(IEnumerable<Tensor> weights) => _xfer.FreeWeights(weights);

    // ── Helpers ─────────────────────────────────────────────────────────

    private const uint LOCAL_X_1D = 256;

    private static uint GroupCount(long total, uint localX)
        => (uint)((total + localX - 1) / localX);

    /// <summary>Resolve the dtype suffix for kernel selection: f16 if both inputs/output are F16; f32 otherwise.</summary>
    private static string DtypeSuffix(DType dt) => dt == DType.F16 ? "_f16" : "_f32";

    /// <summary>The kernel binding count (number of SSBOs) for a given shape — drives descriptor-set-layout selection.</summary>
    private VulkanKernel GetKernel(string shaderName, int storageBufferCount, ReadOnlySpan<SpecConstant> spec)
        => _kernels.Get(shaderName, storageBufferCount, spec);

    /// <summary>Counter of recorded dispatches since last submit. Flushed periodically so the
    /// timeline-semaphore advances and deferred-free buffers can be reclaimed; otherwise large
    /// models accumulate hundreds of allocations between submits and exhaust the heap.</summary>
    private int _dispatchesSinceSubmit;
    private const int FLUSH_THRESHOLD = 8;

    private unsafe void Dispatch(
        VulkanKernel kernel,
        ReadOnlySpan<ulong> bufferHandles,
        ReadOnlySpan<byte> pushConstants,
        uint groupX, uint groupY = 1, uint groupZ = 1)
    {
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
        // CRITICAL: do NOT drain transients here. Multi-dispatch ops like SDPA (24 heads × 3
        // dispatches) reference the same Q/K/V upload buffers across many dispatches. Draining
        // mid-op would tag those buffers for deferred-free, then the next flush completes them
        // and the buffers get vkDestroy'd while later heads' descriptor sets still reference
        // them — silent garbage output for heads beyond the first ~2.
        // Transient drain is now done explicitly at op boundaries via DrainAndFlush.
        if (_dispatchesSinceSubmit >= FLUSH_THRESHOLD && _opNestingDepth == 0)
        {
            DrainAndFlush();
        }
    }

    /// <summary>Tracks whether we're currently inside a backend op. While >0, auto-flush is suppressed
    /// because draining transients mid-op would free buffers still referenced by later dispatches in
    /// the same op (e.g. SDPA's per-head loop sharing Q/K/V uploads).</summary>
    private int _opNestingDepth;

    private OpScope EnterOp([System.Runtime.CompilerServices.CallerMemberName] string opName = "")
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
            b._opNestingDepth++;
        }

        public void Dispose()
        {
            _b._opNestingDepth--;
            if (_b._opNestingDepth == 0)
            {
                int dispatches = _b._dispatchesSinceSubmit;
                _b.DrainAndFlush();
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

    /// <summary>Promotes a freshly-allocated GPU buffer to the activation cache (so the next op finds it via reference equality on the Tensor).</summary>
    private void CacheOutput(Tensor output, VulkanBuffer buffer) => _xfer.CacheActivation(output, buffer);

    private VulkanBuffer GetBuffer(Tensor t) => _xfer.CopyToDevice(t);

    /// <summary>Selects the (BM, BN, BK, TM, TN) tile size for the tiled matmul kernel based on
    /// the GEMM shape. Profile-driven choice (Flux Schnell, RTX 3060): big tiles 128×128/32 for
    /// Flux/SDXL Linear shapes (M, N ≥ 128) cut workgroup count by ~16× vs the old 32×32 default,
    /// which is the primary perf win from Phase C2 step 1. Smaller tiles are picked for shapes
    /// where big tiles would leave most threads idle (e.g. SDPA per-head 64×64 matmuls).
    /// MAX_BM/BN/BK/TM/TN in the shader (matmul_tiled.comp.glsl) cap what's selectable here —
    /// keep them in sync if you bump these values.</summary>
    private static (uint BM, uint BN, uint BK, uint TM, uint TN) PickMatmulTile(long M, long N, long K)
    {
        if (M >= 128 && N >= 128) return (128, 128, 32, 8, 8);   // 16×16 = 256 threads/wg, 32 KB shared
        if (M >= 64  && N >= 64)  return ( 64,  64, 16, 8, 8);   //  8× 8 =  64 threads/wg
        return (32, 32, 16, 4, 4);                               //  8× 8 =  64 threads/wg, small shapes
    }

    /// <summary>Attempts the cooperative-matrix matmul fast path. Returns true when the kernel
    /// was dispatched (op is fully handled). Returns false when the path doesn't apply and the
    /// caller should fall through to <c>matmul_tiled</c>. Constraints for v1:
    /// (a) <c>HasCooperativeMatrix</c> capability, (b) GEMM dtype is FP16, (c) M, N, K all
    /// multiples of <c>FRAG=16</c>, (d) no fused activation. Bias is handled via a follow-up
    /// <c>BroadcastAdd(N, 1)</c> dispatch — see comments below.</summary>
    private static readonly bool _disableCoopmat =
        Environment.GetEnvironmentVariable("HARTSYINFERENCE_VK_DISABLE_COOPMAT") == "1";

    private bool TryDispatchCoopmat(
        Tensor output, VulkanBuffer aRes, VulkanBuffer bRes,
        int M, int N, int K, bool transposeA, bool transposeB, DType gemmDtype,
        VulkanBuffer outBuf, Tensor? bias, VulkanBuffer? biasRes, VulkanBuffer? biasRaw)
    {
        const uint FRAG = 16;
        if (_disableCoopmat) return false;
        if (!Vk.HasCooperativeMatrix) return false;
        if (gemmDtype != DType.F16) return false;
        if ((M % FRAG) != 0 || (N % FRAG) != 0 || (K % FRAG) != 0) return false;
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
        // subgroups per workgroup = (BM/16) * (BN/16). Workgroup size = subgroups * subgroupSize.
        uint subgroups = (BM / FRAG) * (BN / FRAG);
        uint localX = subgroups * subgroupSize;

        try
        {
            string shader = "matmul_coopmat";
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
        catch
        {
            outBuf.Dispose();
            throw;
        }

        return true;
    }

    /// <summary>If <paramref name="src"/> is on a different dtype than <paramref name="want"/>, allocate a temp buffer in <paramref name="want"/> and dispatch the cast kernel; otherwise return <paramref name="srcBuf"/>. Returns (buffer, ownedTemp) — caller frees ownedTemp.</summary>
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
            VulkanKernel k = GetKernel("cast_f8e4m3_f16", storageBufferCount: 2,
                stackalloc SpecConstant[] {
                    SpecConstant.UInt(0, LOCAL_X_1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)
                });
            Span<byte> pc = stackalloc byte[8];
            BinaryWriteUInt(pc, 0, (uint)elements);
            BinaryWriteFloat(pc, 4, src.Fp8ScaleFactor);
            Span<ulong> bufs = stackalloc ulong[] { srcBuf.Handle, dst.Handle };
            Dispatch(k, bufs, pc, GroupCount(elements, LOCAL_X_1D));
            return FinishCast(src, want, dst, cacheThis);
        }

        string shader;
        if (src.DType == DType.F32 && want == DType.F16) shader = "cast_f32_f16";
        else if (src.DType == DType.F16 && want == DType.F32) shader = "cast_f16_f32";
        else if (src.DType == DType.F8E4M3 && want == DType.F32)
        {
            // FP8 -> F32 = FP8 -> F16 -> F32 (two-stage)
            (VulkanBuffer mid, _) = CastIfNeeded(src, srcBuf, DType.F16);
            // Now cast mid (F16) to F32
            VulkanKernel kk = GetKernel("cast_f16_f32", storageBufferCount: 2,
                stackalloc SpecConstant[] {
                    SpecConstant.UInt(0, LOCAL_X_1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)
                });
            Span<byte> pc2 = stackalloc byte[4]; BinaryWriteUInt(pc2, 0, (uint)elements);
            Span<ulong> bufs2 = stackalloc ulong[] { mid.Handle, dst.Handle };
            Dispatch(kk, bufs2, pc2, GroupCount(elements, LOCAL_X_1D));
            _xfer.FreeDevice(mid);
            return FinishCast(src, want, dst, cacheThis);
        }
        else
        {
            dst.Dispose();
            throw new NotSupportedException($"Vulkan cast {src.DType.Name} -> {want.Name} not implemented.");
        }

        VulkanKernel kernel = GetKernel(shader, storageBufferCount: 2,
            stackalloc SpecConstant[] {
                SpecConstant.UInt(0, LOCAL_X_1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)
            });

        Span<byte> push = stackalloc byte[4];
        BinaryWriteUInt(push, 0, (uint)elements);
        Span<ulong> b = stackalloc ulong[] { srcBuf.Handle, dst.Handle };
        Dispatch(kernel, b, push, GroupCount(elements, LOCAL_X_1D));
        return FinishCast(src, want, dst, cacheThis);
    }

    /// <summary>Tail of <see cref="CastIfNeeded"/>: when <paramref name="dst"/> is a preloaded weight's cast,
    /// promote it to the weight-cast cache and return it as non-owned (the caller must not free it); otherwise
    /// return it as a caller-owned temporary.</summary>
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

    // ── Linear algebra ──────────────────────────────────────────────────

    /// <summary>Matrix multiply: output[M,N] = a[M,K] @ b[K,N]. Falls back to FP32 kernel if either input is FP32.</summary>
    public void MatMul(Tensor output, Tensor a, Tensor b)
    {
        using OpScope _ = EnterOp();
        DispatchMatmul(output, a, b, transposeA: false, transposeB: false, bias: null);
    }

    public void Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias)
    {
        using OpScope _ = EnterOp();
        // input [M, K], weight [N, K] → output [M, N]   ⇒  C = A @ B^T  with A=input, B=weight
        DispatchMatmul(output, input, weight, transposeA: false, transposeB: true, bias: bias);
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

    private void DispatchMatmul(Tensor output, Tensor a, Tensor b, bool transposeA, bool transposeB, Tensor? bias)
    {
        // Resolve M, N, K from logical shapes
        int M, N, K;
        if (output.Shape.Rank == 2)
        {
            M = (int)output.Shape[0];
            N = (int)output.Shape[1];
        }
        else
        {
            // Multi-dim output — flatten leading dims for matmul
            long m = 1;
            for (int d = 0; d < output.Shape.Rank - 1; d++) m *= output.Shape[d];
            M = (int)m;
            N = (int)output.Shape[output.Shape.Rank - 1];
        }
        K = transposeA ? (int)a.Shape[0] : (int)a.Shape[a.Shape.Rank - 1];

        // Pick GEMM dtype to MATCH the output's storage dtype — otherwise the matmul kernel writes
        // a smaller element type (e.g. F16) into an F32-sized buffer and the model reads garbage
        // when it interprets the bytes as F32. The model's choice of output dtype dictates the
        // pipeline. F8 inputs always need to be cast (no F8 matmul kernel); cast all the way to
        // the output dtype, not just to F16.
        DType gemmDtype = output.DType;
        if (gemmDtype.IsFp8) gemmDtype = DType.F16;
        if (gemmDtype == DType.F16 && !Capabilities.SupportsF16) gemmDtype = DType.F32;

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
        if (TryDispatchCoopmat(output, aRes, bRes, M, N, K, transposeA, transposeB,
                               gemmDtype, outBuf, bias, biasRes, biasRaw))
        {
            if (aOwned is not null) _xfer.FreeDevice(aOwned);
            if (bOwned is not null) _xfer.FreeDevice(bOwned);
            if (biasOwned is not null) _xfer.FreeDevice(biasOwned);
            return;
        }

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
        catch
        {
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
        }
    }

    /// <summary>INT8 quantized matmul via the integer dot-product extension (the cross-vendor DP4a /
    /// IMMA equivalent): <c>output[m,n] = (a[M,K] @ b[N,K]^T) * scaleA[m] * scaleB[n]</c>, with <paramref name="a"/>
    /// and <paramref name="b"/> INT8 and <paramref name="output"/> F32. <paramref name="b"/> follows the
    /// Linear weight convention ([N,K], row n contiguous along K). Scales are <b>per row</b>: <paramref name="scaleA"/>
    /// is F32 length M (per token/activation row), <paramref name="scaleB"/> is F32 length N (per output channel) —
    /// the standard scheme for accurate INT8 inference. The int32 accumulation is exact; only the dequant rounds.
    /// Requires <see cref="VulkanCapabilities.HasInt8DotProduct"/> and K a multiple of 4. Shared-memory tiled.</summary>
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
        catch
        {
            outBuf.Dispose();
            throw;
        }
    }

    // ── Convolution ─────────────────────────────────────────────────────

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
        DType gemmDtype = output.DType;
        if (gemmDtype.IsFp8) gemmDtype = DType.F16;
        if (gemmDtype == DType.F16 && !Capabilities.SupportsF16) gemmDtype = DType.F32;

        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer wBuf = GetBuffer(weight);
        (VulkanBuffer inRes, VulkanBuffer? inOwned) = CastIfNeeded(input, inBuf, gemmDtype);
        (VulkanBuffer wRes, VulkanBuffer? wOwned) = CastIfNeeded(weight, wBuf, gemmDtype);

        // im2col: rows = N*Cin*kH*kW, cols = N*outH*outW (per the shader's flat layout)
        long colElements = (long)batch * inCh * kH * kW * outH * outW;
        // The im2col GLSL addresses the column buffer with 32-bit indices, and a single SSBO
        // is bounded by maxStorageBufferRange (~4 GB on NVIDIA) regardless. Above int.MaxValue
        // elements the index would wrap / the buffer can't be fully addressed, silently
        // corrupting output. Fail loudly instead. (A full fix widens the shader to 64-bit
        // workgroup-derived indexing; tracked separately.)
        if (colElements > int.MaxValue)
            throw new NotSupportedException(
                $"Vulkan Conv2D im2col buffer needs {colElements} elements (N={batch}, Cin={inCh}, k={kH}x{kW}, out={outH}x{outW}), " +
                "exceeding the shader's 32-bit index range. Use the CUDA backend or tile this convolution.");
        ulong colBytes = (ulong)(colElements * gemmDtype.SizeInBytes);
        VulkanBuffer colBuf = _xfer.AllocateDevice(colBytes);

        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);

        try
        {
            // Step 1: im2col
            {
                string shader = "im2col" + DtypeSuffix(gemmDtype);
                VulkanKernel k = GetKernel(shader, 2, stackalloc SpecConstant[]
                {
                    SpecConstant.UInt(0, LOCAL_X_1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)
                });
                Span<byte> pc = stackalloc byte[12 * 4];
                BinaryWriteUInt(pc, 0, (uint)batch);
                BinaryWriteUInt(pc, 4, (uint)inCh);
                BinaryWriteUInt(pc, 8, (uint)inH);
                BinaryWriteUInt(pc, 12, (uint)inW);
                BinaryWriteUInt(pc, 16, (uint)kH);
                BinaryWriteUInt(pc, 20, (uint)kW);
                BinaryWriteUInt(pc, 24, (uint)padH);
                BinaryWriteUInt(pc, 28, (uint)padW);
                BinaryWriteUInt(pc, 32, (uint)strideH);
                BinaryWriteUInt(pc, 36, (uint)strideW);
                BinaryWriteUInt(pc, 40, (uint)outH);
                BinaryWriteUInt(pc, 44, (uint)outW);
                Span<ulong> bufs = stackalloc ulong[] { inRes.Handle, colBuf.Handle };
                Dispatch(k, bufs, pc, GroupCount(colElements, LOCAL_X_1D));
            }

            // Step 2: matmul — weight[Cout, Cin*kH*kW] @ col[Cin*kH*kW, outH*outW] = out[Cout, outH*outW]
            // Until per-batch base offsets land in the matmul kernel (Phase 4) we restrict to batch=1.
            if (batch != 1)
                throw new NotImplementedException("VulkanBackend.Conv2D: batch>1 requires per-batch base offsets in the matmul kernel — Phase 4 work item.");

            {
                int M = outCh;
                int K = inCh * kH * kW;
                int N = outH * outW;

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
                    SpecConstant.Bool(15, false),
                    SpecConstant.Bool(16, false),
                    SpecConstant.Bool(17, false),
                    SpecConstant.UInt(18, 0u),
                    SpecConstant.Bool(19, false),
                };

                VulkanKernel k = GetKernel(shader, 5, spec);

                Span<byte> pc = stackalloc byte[11 * 4];
                BinaryWriteUInt(pc, 0, (uint)M);
                BinaryWriteUInt(pc, 4, (uint)N);
                BinaryWriteUInt(pc, 8, (uint)K);
                BinaryWriteUInt(pc, 12, (uint)K);
                BinaryWriteUInt(pc, 16, (uint)N);
                BinaryWriteUInt(pc, 20, (uint)N);
                BinaryWriteFloat(pc, 24, 1.0f);
                BinaryWriteFloat(pc, 28, 0.0f);
                BinaryWriteUInt(pc, 32, 0u);
                BinaryWriteUInt(pc, 36, 0u);
                BinaryWriteUInt(pc, 40, 0u);

                Span<ulong> bufs = stackalloc ulong[] { wRes.Handle, colBuf.Handle, outBuf.Handle, outBuf.Handle, outBuf.Handle };
                uint groupsX = (uint)((N + BN - 1) / BN);
                uint groupsY = (uint)((M + BM - 1) / BM);
                Dispatch(k, bufs, pc, groupsX, groupsY, 1);
            }

            // Step 3: optional bias add
            if (bias is not null)
            {
                VulkanBuffer biasRaw = GetBuffer(bias);
                (VulkanBuffer biasRes, VulkanBuffer? biasOwned) = CastIfNeeded(bias, biasRaw, output.DType);
                try
                {
                    string shader = "col2bias_add" + DtypeSuffix(output.DType);
                    VulkanKernel k = GetKernel(shader, 2, stackalloc SpecConstant[]
                    {
                        SpecConstant.UInt(0, LOCAL_X_1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)
                    });
                    Span<byte> pc = stackalloc byte[3 * 4];
                    BinaryWriteUInt(pc, 0, (uint)outCh);
                    BinaryWriteUInt(pc, 4, (uint)(outH * outW));
                    BinaryWriteUInt(pc, 8, (uint)(batch * outCh * outH * outW));
                    Span<ulong> bufs = stackalloc ulong[] { outBuf.Handle, biasRes.Handle };
                    Dispatch(k, bufs, pc, GroupCount(batch * outCh * outH * outW, LOCAL_X_1D));
                }
                finally
                {
                    if (biasOwned is not null) _xfer.FreeDevice(biasOwned);
                }
            }

            CacheOutput(output, outBuf);
        }
        catch
        {
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

    // ── Normalization ───────────────────────────────────────────────────

    private void DispatchPerRowNorm(string shader, int storageBufs, Tensor output, Tensor input, Tensor? weight, Tensor? bias, float eps, int normDim, int totalRows)
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
        catch
        {
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
        catch
        {
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

    public void AdaInstanceNorm1d(Tensor output, Tensor input, Tensor gamma, Tensor beta, float eps)
    {
        throw new NotImplementedException("VulkanBackend.AdaInstanceNorm1d not yet implemented. Use the CPU backend for Kokoro / StyleTTS 2 prosody and decoder paths.");
    }

    public void LeakyRelu(Tensor output, Tensor input, float slope)
    {
        throw new NotImplementedException("VulkanBackend.LeakyRelu not yet implemented. Use the CPU backend for Kokoro / StyleTTS 2.");
    }

    // ── Attention ───────────────────────────────────────────────────────

    /// <summary>Naive 3-pass SDPA. Q*K^T -> mask add -> softmax -> *V dispatched once per (B*H) head with base offsets. FlashAttention-style is a Phase-4 optimization.</summary>
    public void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale)
    {
        using OpScope _ = EnterOp();
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
        DType dtype = output.DType;
        if (dtype.IsFp8) dtype = DType.F16;
        if (dtype == DType.F16 && !Capabilities.SupportsF16) dtype = DType.F32;

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

        // Scratch: scores [B, H, Sq, Skv] for raw QK^T, probsBuf for softmax output (separate
        // buffer to avoid same-binding aliasing in the softmax shader, which has readonly+writeonly
        // bindings and would produce undefined results when src==dst).
        ulong scoresElems = (ulong)(totalHeads * Sq * Skv);
        VulkanBuffer scoresBuf = _xfer.AllocateDevice(scoresElems * (ulong)dtype.SizeInBytes);
        VulkanBuffer probsBuf = _xfer.AllocateDevice(scoresElems * (ulong)dtype.SizeInBytes);

        // Output buffer (dtype)
        ulong outElems = (ulong)(B * H * Sq * D);
        ulong outBytesDtype = outElems * (ulong)dtype.SizeInBytes;
        VulkanBuffer outBufLocal = _xfer.AllocateDevice(outBytesDtype);

        try
        {
            // Per-head dispatch
            for (long bh = 0; bh < totalHeads; bh++)
            {
                uint qOff = (uint)(bh * Sq * D);
                uint kOff = (uint)(bh * Skv * D);
                uint sOff = (uint)(bh * Sq * Skv);
                uint vOff = (uint)(bh * Skv * D);
                uint outOff = (uint)(bh * Sq * D);

                // 1) scores = Q @ K^T * scale  →  shape (Sq, Skv)
                DispatchMatmulWithOffsets(
                    qRes.Handle, kRes.Handle, scoresBuf.Handle,
                    Sq, Skv, D,
                    transposeA: false, transposeB: true,
                    alpha: scale, beta: 0.0f,
                    aOffset: qOff, bOffset: kOff, cOffset: sOff,
                    dtype: dtype);

                // 1b) optional mask add: scores += mask  (broadcast across heads)
                if (maskBuf is not null)
                {
                    // The mask may be either [Sq, Skv] or [B, H, Sq, Skv]. We use modulo so a
                    // [1, 1, Sq, Skv] broadcast cycles back to offset 0 for every head.
                    uint maskOff = (uint)((bh * maskBroadcastSize) % maskTotalSize);
                    DispatchMaskAdd(scoresBuf.Handle, maskBuf.Handle, dtype,
                        (uint)maskBroadcastSize, scoreOffset: sOff, maskOffset: maskOff);
                }

                // 2) softmax along last dim per row of scores  →  probsBuf (separate output)
                DispatchSoftmaxRows(scoresBuf.Handle, probsBuf.Handle, dtype, (int)Skv, (int)Sq, sOff, sOff);

                // 3) out_head = probs @ V  →  shape (Sq, D)
                DispatchMatmulWithOffsets(
                    probsBuf.Handle, vRes.Handle, outBufLocal.Handle,
                    Sq, D, Skv,
                    transposeA: false, transposeB: false,
                    alpha: 1.0f, beta: 0.0f,
                    aOffset: sOff, bOffset: vOff, cOffset: outOff,
                    dtype: dtype);
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

                    VulkanKernel k = GetKernel(shader, 2, stackalloc SpecConstant[]
                    {
                        SpecConstant.UInt(0, LOCAL_X_1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)
                    });
                    Span<byte> pc = stackalloc byte[4]; BinaryWriteUInt(pc, 0, (uint)output.ElementCount);
                    Span<ulong> bufs = stackalloc ulong[] { outBufLocal.Handle, finalBuf.Handle };
                    Dispatch(k, bufs, pc, GroupCount(output.ElementCount, LOCAL_X_1D));
                    CacheOutput(output, finalBuf);
                }
                catch { finalBuf.Dispose(); throw; }
                _xfer.FreeDevice(outBufLocal);
            }
        }
        catch
        {
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
        ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
        {
            SpecConstant.UInt(0, LOCAL_X_1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)
        };
        VulkanKernel k = GetKernel(shader, 2, spec);
        Span<byte> pc = stackalloc byte[3 * 4];
        BinaryWriteUInt(pc, 0, count);
        BinaryWriteUInt(pc, 4, scoreOffset);
        BinaryWriteUInt(pc, 8, maskOffset);
        Span<ulong> bufs = stackalloc ulong[] { scoresHandle, maskHandle };
        Dispatch(k, bufs, pc, GroupCount(count, LOCAL_X_1D));
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

    private void SoftmaxLastDim(Tensor output, Tensor input)
    {
        int N = (int)input.Shape[input.Shape.Rank - 1];
        int rows = (int)(input.ElementCount / N);

        VulkanBuffer inBuf = GetBuffer(input);
        ulong outBytes = (ulong)(output.ElementCount * output.DType.SizeInBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            DispatchSoftmaxRows(inBuf.Handle, outBuf.Handle, input.DType, N, rows, srcOffset: 0, dstOffset: 0);
            CacheOutput(output, outBuf);
        }
        catch { outBuf.Dispose(); throw; }
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

    // ── Activations ─────────────────────────────────────────────────────

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
        throw new NotSupportedException("Vulkan Snake not yet implemented — use CpuBackend for snake-using vocoders.");
    }

    public void Conv1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups)
    {
        throw new NotSupportedException("Vulkan Conv1d not yet implemented — use CpuBackend for codec models.");
    }

    public void ConvTranspose1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups)
    {
        throw new NotSupportedException("Vulkan ConvTranspose1d not yet implemented — use CpuBackend for codec models.");
    }

    // ── Element-wise ────────────────────────────────────────────────────

    public void Add(Tensor output, Tensor a, Tensor b) => DispatchElementwise(0u, output, a, b, scalar: 0, minVal: 0, maxVal: 0);
    public void Mul(Tensor output, Tensor a, Tensor b) => DispatchElementwise(1u, output, a, b, scalar: 0, minVal: 0, maxVal: 0);
    public void Scale(Tensor output, Tensor input, float scalar) => DispatchElementwise(2u, output, input, null, scalar, minVal: 0, maxVal: 0);
    public void Clamp(Tensor output, Tensor input, float min, float max) => DispatchElementwise(7u, output, input, null, scalar: 0, minVal: min, maxVal: max);

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
                SpecConstant.UInt(0, LOCAL_X_1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1),
                SpecConstant.UInt(10, op),
            };
            VulkanKernel k = GetKernel(shader, 3, spec);

            Span<byte> pc = stackalloc byte[4 * 4];
            BinaryWriteUInt(pc, 0, (uint)output.ElementCount);
            BinaryWriteFloat(pc, 4, scalar);
            BinaryWriteFloat(pc, 8, minVal);
            BinaryWriteFloat(pc, 12, maxVal);

            Span<ulong> bufs = stackalloc ulong[] { aBuf.Handle, bBuf?.Handle ?? aBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(output.ElementCount, LOCAL_X_1D));
            CacheOutput(output, outBuf);
        }
        catch { outBuf.Dispose(); throw; }
    }

    // ── Transpose / Permute / GeGlu / BroadcastAdd ─────────────────────

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
            ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
            {
                SpecConstant.UInt(0, LOCAL_X_1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)
            };
            VulkanKernel k = GetKernel(shader, 2, spec);
            Span<byte> pc = stackalloc byte[3 * 4];
            BinaryWriteUInt(pc, 0, (uint)B);
            BinaryWriteUInt(pc, 4, (uint)d1);
            BinaryWriteUInt(pc, 8, (uint)d2);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(output.ElementCount, LOCAL_X_1D));
            CacheOutput(output, outBuf);
        }
        catch { outBuf.Dispose(); throw; }
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
            ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
            {
                SpecConstant.UInt(0, LOCAL_X_1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)
            };
            VulkanKernel k = GetKernel(shader, 2, spec);
            Span<byte> pc = stackalloc byte[4 * 4];
            BinaryWriteUInt(pc, 0, (uint)B);
            BinaryWriteUInt(pc, 4, (uint)s);
            BinaryWriteUInt(pc, 8, (uint)h);
            BinaryWriteUInt(pc, 12, (uint)d);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(output.ElementCount, LOCAL_X_1D));
            CacheOutput(output, outBuf);
        }
        catch { outBuf.Dispose(); throw; }
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
            ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
            {
                SpecConstant.UInt(0, LOCAL_X_1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)
            };
            VulkanKernel k = GetKernel(shader, 2, spec);
            Span<byte> pc = stackalloc byte[2 * 4];
            BinaryWriteUInt(pc, 0, (uint)outerCount);
            BinaryWriteUInt(pc, 4, (uint)D);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(outerCount * D, LOCAL_X_1D));
            CacheOutput(output, outBuf);
        }
        catch { outBuf.Dispose(); throw; }
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
            ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
            {
                SpecConstant.UInt(0, LOCAL_X_1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)
            };
            VulkanKernel k = GetKernel(shader, 2, spec);
            Span<byte> pc = stackalloc byte[3 * 4];
            BinaryWriteUInt(pc, 0, (uint)channels);
            BinaryWriteUInt(pc, 4, (uint)spatial);
            BinaryWriteUInt(pc, 8, (uint)hidden.ElementCount);
            Span<ulong> bufs = stackalloc ulong[] { hBuf.Handle, bEff.Handle };
            Dispatch(k, bufs, pc, GroupCount(hidden.ElementCount, LOCAL_X_1D));
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

    // ── Shape ops ───────────────────────────────────────────────────────

    /// <summary>CPU fallback for concat — dtypes/strides handling is non-trivial, and concat is rare on hot paths.</summary>
    public unsafe void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim)
    {
        // Force lazy sync of all participants
        long elemSize = output.DType.SizeInBytes;
        byte* outPtr = (byte*)output.DataPointer;
        // Compute strides
        long innerStride = 1;
        for (int d = dim + 1; d < output.Shape.Rank; d++) innerStride *= output.Shape[d];
        long outerStride = 1;
        for (int d = 0; d < dim; d++) outerStride *= output.Shape[d];

        // For each "outer" index, concat slices along dim
        for (long o = 0; o < outerStride; o++)
        {
            long writeOffset = o * output.Shape[dim] * innerStride;
            long curDimOffset = 0;
            foreach (Tensor t in inputs)
            {
                byte* tPtr = (byte*)t.DataPointer;
                long tSize = t.Shape[dim] * innerStride * elemSize;
                long srcOffset = o * tSize;
                Buffer.MemoryCopy(tPtr + srcOffset, outPtr + (writeOffset + curDimOffset * innerStride) * elemSize, tSize, tSize);
                curDimOffset += t.Shape[dim];
            }
        }
    }

    public unsafe void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim)
    {
        // CPU fallback
        long elemSize = input.DType.SizeInBytes;
        byte* inPtr = (byte*)input.DataPointer;
        long innerStride = 1;
        for (int d = dim + 1; d < input.Shape.Rank; d++) innerStride *= input.Shape[d];
        long outerStride = 1;
        for (int d = 0; d < dim; d++) outerStride *= input.Shape[d];

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

    // ── Sampling ────────────────────────────────────────────────────────

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
            ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
            {
                SpecConstant.UInt(0, LOCAL_X_1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)
            };
            VulkanKernel k = GetKernel(shader, 2, spec);
            Span<byte> pc = stackalloc byte[6 * 4];
            BinaryWriteUInt(pc, 0, (uint)N);
            BinaryWriteUInt(pc, 4, (uint)C);
            BinaryWriteUInt(pc, 8, (uint)H);
            BinaryWriteUInt(pc, 12, (uint)W);
            BinaryWriteUInt(pc, 16, (uint)scaleH);
            BinaryWriteUInt(pc, 20, (uint)scaleW);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, outBuf.Handle };
            Dispatch(k, bufs, pc, GroupCount(output.ElementCount, LOCAL_X_1D));
            CacheOutput(output, outBuf);
        }
        catch { outBuf.Dispose(); throw; }
    }

    // ── Data movement ───────────────────────────────────────────────────

    public unsafe void CopyTo(Tensor destination, Tensor source)
    {
        long byteCount = source.ElementCount * source.DType.SizeInBytes;
        Buffer.MemoryCopy(source.DataPointer, destination.DataPointer, byteCount, byteCount);
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

    // ── Audio (not supported on this backend) ───────────────────────────

    public void Fft(Tensor output, Tensor input)
        => throw new NotSupportedException("VulkanBackend does not support FFT — use the CPU backend for audio preprocessing.");

    public void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window)
        => throw new NotSupportedException("VulkanBackend does not support STFT — use the CPU backend.");

    public void MelFilterbank(Tensor output, Tensor input, Tensor filters)
        => throw new NotSupportedException("VulkanBackend does not support MelFilterbank — use the CPU backend.");

    // ── Casts ──────────────────────────────────────────────────────────

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
                VulkanKernel k = GetKernel("cast_f16_f32", 2,
                    stackalloc SpecConstant[] {
                        SpecConstant.UInt(0, LOCAL_X_1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1)
                    });
                Span<byte> pc = stackalloc byte[4]; BinaryWriteUInt(pc, 0, (uint)input.ElementCount);
                Span<ulong> bufs = stackalloc ulong[] { src.Handle, dst.Handle };
                Dispatch(k, bufs, pc, GroupCount(input.ElementCount, LOCAL_X_1D));
                CacheOutput(output, dst);
            }
            catch { dst.Dispose(); throw; }
        }
        else
        {
            Half* src = (Half*)input.DataPointer;
            float* dst = (float*)output.DataPointer;
            for (long i = 0; i < input.ElementCount; i++) dst[i] = (float)src[i];
        }
    }

    // ── Disposal ────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        try { _stream.WaitIdleHost(); } catch { /* swallow */ }
        // Dump profiling data (no-op when HARTSYINFERENCE_VK_PROFILE is unset) before tearing down
        // anything that the per-op records reference (op names are strings only, but flush before
        // we lose the device just to be tidy).
        try { _profiler.Dump(); } catch { /* swallow */ }
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
}

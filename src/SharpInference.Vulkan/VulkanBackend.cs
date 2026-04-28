using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Vulkan;

/// <summary>
/// Vulkan compute backend implementing <see cref="IBackend"/>.
/// Routes operations to SPIR-V compute shaders loaded from disk and dispatched
/// via vkCmdDispatch on a single timeline-semaphore stream. Mirrors the CUDA
/// backend's GPU weight cache + lazy-sync activation cache so model code that
/// works on CUDA works unchanged here.
/// </summary>
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
    private readonly string _spvDir;
    private bool _disposed;

    public DeviceKind Device { get; }
    public BackendCapabilities Capabilities { get; }
    public VulkanCapabilities Vk => _vkDevice.Capabilities;

    /// <summary>Creates a Vulkan backend on the best discrete GPU. Validation layers enabled if SHARPINFERENCE_VK_VALIDATION=1.</summary>
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
        _descriptors = new VulkanDescriptorManager(_vkDevice.Handle);
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

        ulong setLayout = kernel.DescriptorSetLayout;
        ulong dstSet = _descriptors.AllocateSet(setLayout);
        _descriptors.WriteSet(dstSet, bufferHandles);

        ulong layout = kernel.PipelineLayout;
        VulkanApi.vkCmdBindDescriptorSets(cb, VkPipelineBindPoint.Compute, layout,
            0, 1, (nint)(&dstSet), 0, 0);

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

    private OpScope EnterOp() => new(this);
    private readonly struct OpScope : IDisposable
    {
        private readonly VulkanBackend _b;
        public OpScope(VulkanBackend b) { _b = b; b._opNestingDepth++; }
        public void Dispose()
        {
            _b._opNestingDepth--;
            if (_b._opNestingDepth == 0)
                _b.DrainAndFlush();
        }
    }

    private void DrainAndFlush()
    {
        _xfer.DrainTransients();
        _stream.WaitIdleHost();
        _dispatchesSinceSubmit = 0;
    }

    /// <summary>Promotes a freshly-allocated GPU buffer to the activation cache (so the next op finds it via reference equality on the Tensor).</summary>
    private void CacheOutput(Tensor output, VulkanBuffer buffer) => _xfer.CacheActivation(output, buffer);

    private VulkanBuffer GetBuffer(Tensor t) => _xfer.CopyToDevice(t);

    /// <summary>If <paramref name="src"/> is on a different dtype than <paramref name="want"/>, allocate a temp buffer in <paramref name="want"/> and dispatch the cast kernel; otherwise return <paramref name="srcBuf"/>. Returns (buffer, ownedTemp) — caller frees ownedTemp.</summary>
    private (VulkanBuffer buf, VulkanBuffer? owned) CastIfNeeded(Tensor src, VulkanBuffer srcBuf, DType want)
    {
        if (src.DType == want) return (srcBuf, null);

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
            return (dst, dst);
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
            return (dst, dst);
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

        try
        {
            // Tile shape: keep small but-correct defaults. Tunable per vendor in Phase 4.
            const uint BM = 32, BN = 32, BK = 16, TM = 4, TN = 4;
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

                const uint BM = 32, BN = 32, BK = 16, TM = 4, TN = 4;
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
        const uint BM = 32, BN = 32, BK = 16, TM = 4, TN = 4;
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
            // hidden mutated in place — clear cache callbacks before re-cache, see PHASE_3_DEVIATIONS #17.
            // Note: BroadcastAdd modifies hidden in-place; the previous activation cache entry on `hidden`
            // already points at hBuf. We don't need to re-cache since the buffer is unchanged.
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

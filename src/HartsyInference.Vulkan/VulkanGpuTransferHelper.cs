using HartsyInference.Core.Configuration;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Gpu;

namespace HartsyInference.Vulkan;

/// <summary>Vulkan's device-residency cache: which tensors are on the GPU, and the transfers that put them there.</summary>
/// <remarks>The caches, the lookup order, the tensor-binding lifecycle, the weight-cast cache and the offload policy
/// live in <see cref="GpuResidencyCache{TBuffer}"/> and are shared with every other GPU backend. What remains here is
/// what is genuinely Vulkan: the ReBAR-or-staging upload, the non-coherent flush/invalidate, deferred frees tied to
/// the command stream's timeline, and the step-graph retain list.</remarks>
public sealed class VulkanGpuTransferHelper : GpuResidencyCache<VulkanBuffer>
{
    /// <summary>Uncached upload buffers from cache misses, freed when the command stream next flushes.</summary>
    private readonly List<VulkanBuffer> _transientBuffers = new();

    private readonly nint _device;
    private readonly VulkanMemoryAllocator _allocator;
    private readonly VkPhysicalDeviceMemoryProperties _memProps;
    private readonly VulkanCapabilities _caps;
    private readonly VulkanCommandStream _stream;

    public VulkanGpuTransferHelper(
        nint device,
        VulkanMemoryAllocator allocator,
        in VkPhysicalDeviceMemoryProperties memProps,
        VulkanCapabilities caps,
        VulkanCommandStream stream)
    {
        _device = device;
        _allocator = allocator;
        _memProps = memProps;
        _caps = caps;
        _stream = stream;
        // A large fp8 or quantized model whose full cast set does not fit alongside its own raw weights needs this
        // off: found by a real Krea2 OOM on this backend, where every layer's F32 cast stayed resident forever on
        // top of the fp8 weights and exhausted VRAM partway through the text encoder alone.
        CacheWeightCasts = !EngineKnobs.VkNoWeightCastCache.Value;
    }

    // ── Step-graph capture ──────────────────────────────────────────────────────────────────────────────
    // A captured command buffer bakes device addresses at record time, through bound descriptors and any
    // push-descriptor writes. Every buffer a recorded dispatch referenced must therefore outlive the graph, not
    // just the next deferred-free tick — and the model's ordinary allocate/dispatch/dispose pattern runs unchanged
    // during capture, so each of those disposals has to be intercepted and redirected here instead.

    /// <summary>Whether a step graph is being recorded. Set by the backend around capture.</summary>
    public bool CapturingStepGraph { get; set; }

    private readonly List<VulkanBuffer> _stepGraphRetained = new();

    /// <summary>A host read is illegal while recording: it violates the capture contract, and servicing it would
    /// submit and drain the normal stream while the capture buffer is still open.</summary>
    protected override bool HostReadForbidden => CapturingStepGraph;

    /// <summary>A buffer released during capture is retained for the graph's lifetime rather than freed.</summary>
    protected override bool TryRetainOnRelease(VulkanBuffer buffer)
    {
        if (!CapturingStepGraph)
        {
            return false;
        }
        _stepGraphRetained.Add(buffer);
        return true;
    }

    /// <summary>A cache miss during capture would upload now and freeze this step's bytes into the graph, so every
    /// replay would reuse them. The pre-capture warmup passes exist to make every tensor resident first; a miss means
    /// that invariant broke, and the caller's capture fallback turns graph mode off for the session.</summary>
    protected override void OnMissDuringCapture(Tensor tensor)
    {
        if (!CapturingStepGraph)
        {
            return;
        }
        throw new InvalidOperationException(
            $"VulkanGpuTransferHelper: cache miss during step-graph capture for {tensor.Shape} {tensor.DType.Name} — "
            + "uploading now would freeze this step's data into the graph. The pre-capture warmup should have "
            + "made every tensor the capture touches resident.");
    }

    /// <summary>Frees every buffer retained during capture. Call only after the graph itself is torn down: these
    /// back the captured command buffer's baked-in bindings, and freeing them while it could still replay would
    /// leave it pointing at destroyed memory.</summary>
    public void ReleaseStepGraphRetained()
    {
        foreach (VulkanBuffer buffer in _stepGraphRetained)
        {
            buffer.Dispose();
        }
        _stepGraphRetained.Clear();
    }

    /// <summary>Buffers currently held for a capture, and their total size.</summary>
    /// <remarks>Diagnostic. Capture has a fundamentally higher peak-VRAM requirement than eager execution: it must
    /// hold every intermediate the whole recorded pass touched alive at once, instead of freeing and reusing
    /// block by block.</remarks>
    public (int count, long bytes) StepGraphRetainedStats()
    {
        long bytes = 0;
        foreach (VulkanBuffer buffer in _stepGraphRetained)
        {
            bytes += (long)buffer.Size;
        }
        return (_stepGraphRetained.Count, bytes);
    }

    // ── The operations the shared cache delegates ───────────────────────────────────────────────────────

    /// <inheritdoc/>
    protected override VulkanBuffer AllocateDevice(long bytes) => AllocateDevice((ulong)bytes);

    /// <summary>True inside a bulk release, which has already drained the stream and is not followed by a flush.</summary>
    private bool _releasingInBulk;

    /// <inheritdoc/>
    /// <remarks>Deferred against the timeline, because the stream may still hold recorded work referencing this
    /// buffer — except inside a bulk release, which drains first and then ends, so a deferred free tagged with the
    /// next submit's tick would strand every byte it was asked to reclaim. At teardown that is fatal rather than
    /// wasteful: the buffer's finalizer would call <c>vkDestroyBuffer</c> against an already-destroyed
    /// device.</remarks>
    protected override void FreeDevice(VulkanBuffer buffer, long bytes)
    {
        if (_releasingInBulk)
        {
            buffer.Dispose();
            return;
        }
        _stream.DeferredFree(buffer);
    }

    /// <summary>Called after a bulk release has drained the stream, so the backend's own submit accounting can
    /// account for a submit it did not make.</summary>
    /// <remarks>The backend batches submits by counting dispatches since the last one. A drain here submits
    /// whatever was recorded, and without this the count keeps those dispatches, so the next op crosses the flush
    /// threshold early. Harmless — a submit with nothing recorded is a no-op — but it makes the number a lie.</remarks>
    internal Action? StreamDrained { get; set; }

    /// <summary>Submits what is recorded, waits for it, and tells the backend the submit happened.</summary>
    /// <remarks>Every drain in this class goes through here for the same reason the backend routes its own through
    /// one helper: a drain always submits first, and a drain the batching count never hears about leaves that count
    /// holding dispatches already sent.</remarks>
    private void DrainStream()
    {
        _stream.WaitIdleHost();
        StreamDrained?.Invoke();
    }

    /// <summary>Runs a bulk release with the stream drained first and the frees taken immediately.</summary>
    private void ReleaseInBulk(Action release)
    {
        DrainStream();
        _releasingInBulk = true;
        try
        {
            release();
        }
        finally
        {
            _releasingInBulk = false;
        }
    }

    /// <inheritdoc/>
    protected override void Upload(VulkanBuffer destination, Tensor source, long bytes) => Upload(destination, source);

    /// <inheritdoc/>
    protected override void DownloadSynced(nint hostDestination, VulkanBuffer source, long bytes)
    {
        DrainStream();
        DownloadToHost(hostDestination, source, (ulong)bytes);
    }

    /// <inheritdoc/>
    /// <remarks>Vulkan has no current-device notion; the device is bound to this helper for its lifetime.</remarks>
    protected override void MakeCurrent()
    {
    }

    /// <inheritdoc/>
    /// <remarks>An uncached upload has to survive until every dispatch that referenced it has been recorded, which
    /// is what <see cref="DrainTransients"/> waits for; freeing it when the op ends would destroy it mid-op.</remarks>
    protected override void OnTransientUploaded(VulkanBuffer buffer, Tensor source) => _transientBuffers.Add(buffer);

    // ── The Vulkan-specific surface the backend calls ───────────────────────────────────────────────────

    /// <summary>Allocates a device-local buffer for an op output or temporary.</summary>
    /// <remarks>Where ReBAR is available the memory type is also asked to be host-visible, so the uploader can write
    /// straight through a mapping and skip a staging copy — which also spares the host-visible heap, much smaller
    /// than VRAM on most discrete cards.</remarks>
    public VulkanBuffer AllocateDevice(ulong byteSize)
    {
        VkMemoryPropertyFlags preferred = _caps.HasReBar
            ? VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent
            : VkMemoryPropertyFlags.None;
        return VulkanBufferFactory.Create(
            _device, _allocator, in _memProps,
            byteSize,
            VkBufferUsageFlags.StorageBuffer | VkBufferUsageFlags.TransferSrc | VkBufferUsageFlags.TransferDst,
            VkMemoryPropertyFlags.DeviceLocal,
            preferred);
    }

    /// <summary>Releases an op temporary, unless the cache has taken ownership of it since.</summary>
    public void FreeDevice(VulkanBuffer? buffer) => ReleaseIfNotCached(buffer, buffer is null ? 0 : (long)buffer.Size);

    /// <summary>Binds an op's output buffer to its tensor, so the next op reads it in place.</summary>
    public void CacheActivation(Tensor tensor, VulkanBuffer buffer) => CacheActivation(tensor, buffer, ByteSize(tensor));

    /// <summary>Stores a weight's dtype conversion, sized from the buffer rather than the tensor — a conversion's
    /// byte count is not the weight's, which is the whole point of making one.</summary>
    public void StoreWeightCast(Tensor weight, DType want, VulkanBuffer buffer)
        => StoreWeightCast(weight, want, buffer, (long)buffer.Size);

    /// <summary>Hands every uncached upload buffer to the deferred-free list. Called by the backend at a flush, by
    /// which point every dispatch that referenced them has been recorded into the active command buffer.</summary>
    /// <remarks>Never call this mid-op. A multi-dispatch op — attention runs three dispatches per head — reuses the
    /// same upload buffers across its dispatches, and draining between them tags buffers later dispatches still
    /// reference, which the next flush then destroys while their descriptor sets point at them.</remarks>
    public void DrainTransients()
    {
        foreach (VulkanBuffer transient in _transientBuffers)
        {
            if (CachedBuffers.Contains(transient) || TryRetainOnRelease(transient))
            {
                continue;
            }
            _stream.DeferredFree(transient);
        }
        _transientBuffers.Clear();
    }

    /// <summary>Uploads weights, then submits so the transfers are in flight before the first op needs them.</summary>
    public override void PreloadWeights(IEnumerable<Tensor> weights)
    {
        base.PreloadWeights(weights);
        _stream.SubmitAndAdvance();
    }

    /// <summary>Releases these weights and their conversions, after waiting for work that might still read them.</summary>
    public override void FreeWeights(IEnumerable<Tensor> weights)
    {
        Tensor[] materialized = [.. weights];
        ReleaseInBulk(() => base.FreeWeights(materialized));
    }

    /// <summary>Releases every unpinned activation. The phase-boundary counterpart to <see cref="FreeWeights"/>.</summary>
    public override void FreeActivations() => ReleaseInBulk(base.FreeActivations);

    /// <summary>Drops every cached buffer and any pending transients.</summary>
    /// <remarks>Waits for the device first, then lets the base clear the caches — which also neutralizes the tensor
    /// bindings. That ordering matters: a tensor finalized after the backend is disposed would otherwise run a
    /// callback closing over destroyed device state.</remarks>
    public override void FreeAllCached() => ReleaseInBulk(() =>
    {
        base.FreeAllCached();
        foreach (VulkanBuffer transient in _transientBuffers)
        {
            transient.Dispose();
        }
        _transientBuffers.Clear();
    });

    /// <summary>The shared occupancy summary plus the Vulkan-only rows.</summary>
    public override string DiagnosticsSummary()
    {
        (int retainedCount, long retainedBytes) = StepGraphRetainedStats();
        return base.DiagnosticsSummary()
            + $"\n  step-graph-retained: {retainedCount} buffers, {ByteFormat.MbF1(retainedBytes)} (capturing={CapturingStepGraph})"
            + $"\n  transient (in-flight, uncached) buffers: {_transientBuffers.Count}";
    }

    /// <summary>Cached bytes, hits and misses, for the backend's own stats line.</summary>
    public (long cachedBytes, long hits, long misses) GetStats()
    {
        long cachedBytes = 0;
        foreach (VulkanBuffer buffer in CachedBuffers)
        {
            cachedBytes += (long)buffer.Size;
        }
        return (cachedBytes, Hits, Misses);
    }

    // ── Transfers ───────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Uploads a tensor's bytes into a device-local buffer: through a mapping where ReBAR allows it, and a
    /// staging buffer otherwise.</summary>
    public unsafe void Upload(VulkanBuffer dst, Tensor src)
    {
        ulong size = (ulong)ByteSize(src);
        if (size == 0)
        {
            return;
        }

        if (dst.MappedPointer != 0)
        {
            // ReBAR / unified memory: write straight through, no staging buffer.
            Buffer.MemoryCopy(src.DataPointer, (void*)dst.MappedPointer, (long)dst.Size, (long)size);
            FlushIfNonCoherent(dst);
            return;
        }

        VulkanBuffer staging = VulkanBufferFactory.Create(
            _device, _allocator, in _memProps,
            size,
            VkBufferUsageFlags.TransferSrc,
            VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent);
        try
        {
            Buffer.MemoryCopy(src.DataPointer, (void*)staging.MappedPointer, (long)staging.Size, (long)size);
            // The destination may be a buffer a dispatch just wrote, or one an earlier staging copy wrote, and
            // neither is ordered against this one by the compute→compute barrier a dispatch leaves behind.
            _stream.RecordComputeToCopyBarrier();
            _stream.RecordCopyAndBarrier(staging.Handle, dst.Handle, size,
                postStage: VkPipelineStageFlags2.ComputeShader,
                // Write as well as read: an in-place op dispatches straight onto a buffer it just uploaded, and
                // a destination scope naming only reads leaves that write-after-write unordered.
                postAccess: VkAccessFlags2.ShaderStorageRead | VkAccessFlags2.ShaderStorageWrite);
            // Submit so the staging buffer can be released once the copy has run.
            ulong tick = _stream.SubmitAndAdvance();
            _stream.DeferredFreeAt(tick, staging);
        }
        catch
        {
            staging.Dispose();
            throw;
        }
    }

    /// <summary>Copies a device buffer back to host memory. Drives the lazy-sync callback the cache installs.</summary>
    public unsafe void DownloadToHost(nint cpuPtr, VulkanBuffer src, ulong size)
    {
        if (src.MappedPointer != 0)
        {
            InvalidateIfNonCoherent(src);
            Buffer.MemoryCopy((void*)src.MappedPointer, (void*)cpuPtr, (long)size, (long)size);
            return;
        }

        VulkanBuffer staging = VulkanBufferFactory.Create(
            _device, _allocator, in _memProps,
            size,
            VkBufferUsageFlags.TransferDst,
            VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent,
            VkMemoryPropertyFlags.HostCached);
        try
        {
            _stream.RecordComputeToCopyBarrier();
            _stream.RecordCopyAndBarrier(src.Handle, staging.Handle, size,
                postStage: VkPipelineStageFlags2.Host,
                postAccess: VkAccessFlags2.HostRead);
            ulong tick = _stream.SubmitAndAdvance();
            _stream.WaitTimeline(tick);
            Buffer.MemoryCopy((void*)staging.MappedPointer, (void*)cpuPtr, (long)size, (long)size);
        }
        finally
        {
            staging.Dispose();
        }
    }

    // Aligning the offset DOWN has to be paid for by extending the END, or the tail bytes fall outside the range
    // and the other side sees stale data at the end of any non-atom-aligned buffer.
    private VkMappedMemoryRange AtomAlignedRange(VulkanBuffer buffer)
    {
        ulong atom = _caps.NonCoherentAtomSize;
        ulong rawOffset = buffer.Allocation.Offset;
        ulong start = AlignDown(rawOffset, atom);
        ulong end = AlignUp(rawOffset + buffer.Size, atom);
        return new VkMappedMemoryRange
        {
            sType = VkStructureType.MappedMemoryRange,
            memory = buffer.Allocation.DeviceMemory,
            offset = start,
            size = end - start,
        };
    }

    private void FlushIfNonCoherent(VulkanBuffer buffer)
    {
        VkMemoryType mt = _memProps.GetMemoryType((int)buffer.Allocation.MemoryTypeIndex);
        if ((mt.propertyFlags & VkMemoryPropertyFlags.HostCoherent) != 0)
        {
            return;
        }
        VkMappedMemoryRange range = AtomAlignedRange(buffer);
        unsafe
        {
            VulkanApi.vkFlushMappedMemoryRanges(_device, 1, (nint)(&range)).ThrowOnError("vkFlushMappedMemoryRanges");
        }
    }

    private void InvalidateIfNonCoherent(VulkanBuffer buffer)
    {
        VkMemoryType mt = _memProps.GetMemoryType((int)buffer.Allocation.MemoryTypeIndex);
        if ((mt.propertyFlags & VkMemoryPropertyFlags.HostCoherent) != 0)
        {
            return;
        }
        VkMappedMemoryRange range = AtomAlignedRange(buffer);
        unsafe
        {
            VulkanApi.vkInvalidateMappedMemoryRanges(_device, 1, (nint)(&range)).ThrowOnError("vkInvalidateMappedMemoryRanges");
        }
    }

    private static ulong AlignUp(ulong v, ulong a) => (v + a - 1) & ~(a - 1);

    private static ulong AlignDown(ulong v, ulong a) => v & ~(a - 1);
}

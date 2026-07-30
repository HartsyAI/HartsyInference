using System.Runtime.CompilerServices;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

/// <summary>Vulkan analogue of <c>HartsyInference.Cuda.GpuTransferHelper</c>.</summary>
/// <remarks>Holds a weight cache (permanent until <see cref="FreeWeights"/> / <see cref="FreeAllCached"/>) and
/// an activation cache (set by <see cref="CacheActivation"/> after each op, consumed by the next op's
/// <see cref="CopyToDevice"/>), both keyed by Tensor reference. Lazy-sync callbacks on the Tensor mirror
/// CUDA so model code may freely access <c>DataPointer</c>.</remarks>
public sealed class VulkanGpuTransferHelper : IDisposable
{
    private readonly Dictionary<Tensor, VulkanBuffer> _weightCache = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<Tensor, VulkanBuffer> _activationCache = new(ReferenceEqualityComparer.Instance);
    /// <summary>Per-weight, per-target-dtype cast cache — casts a preloaded weight to the GEMM dtype once instead of on every Linear.</summary>
    private readonly Dictionary<Tensor, Dictionary<string, VulkanBuffer>> _weightCastCache = new(ReferenceEqualityComparer.Instance);
    /// <summary>Master kill-switch for weight dtype-cast caching (mirrors <c>CudaBackend.CacheWeightCasts</c>
    /// exactly, exposed on <see cref="VulkanBackend.CacheWeightCasts"/>). Defaults from
    /// <c>HARTSYINFERENCE_VK_NO_WEIGHT_CAST_CACHE=1</c>, settable afterward — a large FP8/quantized model
    /// whose full dtype-cast set (e.g. FP8→F32, 4x expansion) doesn't fit VRAM alongside its own raw
    /// weights needs this off (transient, recomputed-and-freed-per-call dequant) instead of caching every
    /// layer's cast forever. Found via a real OOM running Krea2 (13 GB fp8) on Vulkan: with no way to
    /// disable it, EVERY layer's F32-cast weight stayed resident permanently on top of the raw FP8
    /// weights, exhausting VRAM partway through just the 4B-param text encoder. See TROUBLESHOOTING.md.</summary>
    public bool CacheWeightCasts { get; set; } =
        Environment.GetEnvironmentVariable("HARTSYINFERENCE_VK_NO_WEIGHT_CAST_CACHE") != "1";
    private readonly HashSet<VulkanBuffer> _cachedBuffers = new();
    /// <summary>Uncached upload buffers from CopyToDevice cache-misses, drained when the command stream flushes.</summary>
    private readonly List<VulkanBuffer> _transientBuffers = new();

    private readonly nint _device;
    private readonly VulkanMemoryAllocator _allocator;
    private readonly VkPhysicalDeviceMemoryProperties _memProps;
    private readonly VulkanCapabilities _caps;
    private readonly VulkanCommandStream _stream;

    private long _cachedBytes;
    private long _hits;
    private long _misses;
    private long _d2hSyncs;

    /// <summary>Number of lazy D2H sync callbacks fired since the last <see cref="ResetSyncCount"/>. Each
    /// one is a full GPU stall plus a device-to-host copy; a GPU-resident hot loop should fire none.
    /// Mirrors <c>HartsyInference.Cuda.GpuTransferHelper.GetSyncCount</c>.</summary>
    public long GetSyncCount() => _d2hSyncs;

    /// <summary>Resets the D2H sync counter (call at the start of a region you want to measure for residency).</summary>
    public void ResetSyncCount() => _d2hSyncs = 0;

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
    }

    /// <summary>Peeks the weight/activation caches WITHOUT uploading on a miss — unlike
    /// <see cref="CopyToDevice"/>, does not touch <c>DataPointer</c> or allocate anything. Lets a caller
    /// (e.g. <see cref="VulkanBackend.CopyTo"/>) take a device-to-device fast path only when the source
    /// is ALREADY GPU-resident, instead of always forcing one.</summary>
    public bool TryGetCached(Tensor tensor, out VulkanBuffer? buffer)
    {
        if (_weightCache.TryGetValue(tensor, out buffer)) return true;
        if (_activationCache.TryGetValue(tensor, out buffer)) return true;
        buffer = null;
        return false;
    }

    /// <summary>Returns the GPU buffer for a tensor, using caches to avoid transfers. Priority: weight cache → activation cache → fresh H2D upload.</summary>
    public VulkanBuffer CopyToDevice(Tensor cpuTensor)
    {
        if (_weightCache.TryGetValue(cpuTensor, out VulkanBuffer? cachedWeight))
        { _hits++; return cachedWeight; }

        if (_activationCache.TryGetValue(cpuTensor, out VulkanBuffer? cachedAct))
        { _hits++; return cachedAct; }

        _misses++;
        ulong byteSize = (ulong)ByteSize(cpuTensor);
        VulkanBuffer dst = AllocateDevice(byteSize);
        Upload(dst, cpuTensor);
        // Record so it gets freed at the next stream flush; otherwise streamed weights leak.
        _transientBuffers.Add(dst);
        return dst;
    }

    /// <summary>Schedules every non-cached upload buffer for deferred-free. Called by the backend
    /// during periodic flushes — by then every dispatch that referenced these buffers has been
    /// recorded into the active command buffer, so tagging at _value+1 is safe.</summary>
    public void DrainTransients()
    {
        foreach (VulkanBuffer t in _transientBuffers)
        {
            if (!_cachedBuffers.Contains(t)) _stream.DeferredFree(t);
        }
        _transientBuffers.Clear();
    }

    /// <summary>Allocates a device-local buffer for an op output / temporary. When ReBAR is
    /// available we *prefer* a memory type that is also HOST_VISIBLE so the uploader can write
    /// directly via mapped memory and skip staging — saving an entire staging copy + freeing the
    /// host-visible heap (which is much smaller than VRAM on most discrete GPUs).</summary>
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

    /// <summary>Frees a device buffer asynchronously through the stream's deferred-free list. Skips cached buffers (weights + activations).</summary>
    public void FreeDevice(VulkanBuffer? buffer)
    {
        if (buffer is null) return;
        if (_cachedBuffers.Contains(buffer)) return;
        _stream.DeferredFree(buffer);
    }

    /// <summary>Caches an op's output buffer with the tensor — avoids D2H. Sets lazy-sync + dispose callbacks for CPU access.</summary>
    public unsafe void CacheActivation(Tensor tensor, VulkanBuffer buffer)
    {
        // Do NOT touch tensor.DataPointer here: that would force the lazy host buffer to allocate (and zero) for
        // every GPU-resident activation. The host buffer is allocated only if/when CPU code actually reads the
        // tensor, inside the sync callback below (via Tensor.EnsureHostBuffer).
        ulong byteSize = (ulong)ByteSize(tensor);

        // Old callbacks may close over an earlier GPU buffer (in-place ops on the same Tensor). Clear before re-cache.
        tensor._gpuSyncCallback = null;
        tensor._gpuDisposeCallback = null;

        _activationCache[tensor] = buffer;
        _cachedBuffers.Add(buffer);

        VulkanGpuTransferHelper self = this;
        Tensor capturedTensor = tensor;
        VulkanBuffer capturedBuffer = buffer;
        ulong capturedSize = byteSize;

        // Lazy sync: when CPU code reads the tensor, wait for the stream, copy back, free GPU buffer.
        tensor._gpuSyncCallback = () =>
        {
            if (self._activationCache.Remove(capturedTensor, out VulkanBuffer? cached))
            {
                self._d2hSyncs++;
                self._stream.WaitIdleHost();
                self.DownloadToHost((nint)capturedTensor.EnsureHostBuffer(), cached, capturedSize);
                self._cachedBuffers.Remove(cached);
                self._stream.DeferredFree(cached);
            }
        };

        tensor._gpuDisposeCallback = () =>
        {
            if (self._activationCache.Remove(capturedTensor, out VulkanBuffer? cached))
            {
                self._cachedBuffers.Remove(cached);
                self._stream.DeferredFree(cached);
            }
        };
    }

    /// <summary>Uploads a CPU tensor's bytes into a device-local Vulkan buffer. Uses the ReBAR fast path if the buffer's memory type is mappable.</summary>
    public unsafe void Upload(VulkanBuffer dst, Tensor src)
    {
        ulong size = (ulong)ByteSize(src);
        if (size == 0) return;

        if (dst.MappedPointer != 0)
        {
            // ReBAR / unified-memory fast path: write directly, no staging buffer.
            Buffer.MemoryCopy(src.DataPointer, (void*)dst.MappedPointer, (long)dst.Size, (long)size);
            FlushIfNonCoherent(dst);
            return;
        }

        // Stage via HOST_VISIBLE | HOST_COHERENT buffer
        VulkanBuffer staging = VulkanBufferFactory.Create(
            _device, _allocator, in _memProps,
            size,
            VkBufferUsageFlags.TransferSrc,
            VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent);
        try
        {
            Buffer.MemoryCopy(src.DataPointer, (void*)staging.MappedPointer, (long)staging.Size, (long)size);
            _stream.RecordCopyAndBarrier(staging.Handle, dst.Handle, size,
                postStage: VkPipelineStageFlags2.ComputeShader,
                postAccess: VkAccessFlags2.ShaderStorageRead);
            // Submit so we can free the staging buffer once the copy is done.
            ulong tick = _stream.SubmitAndAdvance();
            _stream.DeferredFreeAt(tick, staging);
        }
        catch
        {
            staging.Dispose();
            throw;
        }
    }

    /// <summary>Downloads a device-local buffer back to CPU memory. Used by the lazy-sync callback installed in <see cref="CacheActivation"/>.</summary>
    public unsafe void DownloadToHost(nint cpuPtr, VulkanBuffer src, ulong size)
    {
        if (src.MappedPointer != 0)
        {
            // Mapped — direct read after invalidate (if non-coherent)
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

    // Builds an atom-aligned VkMappedMemoryRange that fully covers the buffer's bytes.
    // Aligning the offset DOWN must be compensated by extending the END to cover the data;
    // computing size from buffer.Size alone (the old bug) can leave the tail bytes outside
    // the range, so the GPU/CPU sees stale data at the end of any non-atom-aligned buffer.
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
        if ((mt.propertyFlags & VkMemoryPropertyFlags.HostCoherent) != 0) return;

        VkMappedMemoryRange range = AtomAlignedRange(buffer);
        unsafe
        {
            VulkanApi.vkFlushMappedMemoryRanges(_device, 1, (nint)(&range))
                .ThrowOnError("vkFlushMappedMemoryRanges");
        }
    }

    private void InvalidateIfNonCoherent(VulkanBuffer buffer)
    {
        VkMemoryType mt = _memProps.GetMemoryType((int)buffer.Allocation.MemoryTypeIndex);
        if ((mt.propertyFlags & VkMemoryPropertyFlags.HostCoherent) != 0) return;

        VkMappedMemoryRange range = AtomAlignedRange(buffer);
        unsafe
        {
            VulkanApi.vkInvalidateMappedMemoryRanges(_device, 1, (nint)(&range))
                .ThrowOnError("vkInvalidateMappedMemoryRanges");
        }
    }

    private static ulong AlignUp(ulong v, ulong a) => (v + a - 1) & ~(a - 1);
    private static ulong AlignDown(ulong v, ulong a) => v & ~(a - 1);

    /// <summary>True when <paramref name="weight"/> is a preloaded weight whose dtype-cast result is worth
    /// caching (i.e. caching is enabled and the tensor lives in the permanent weight cache). Activations
    /// are never cached this way — they change every step, so the key would never hit.</summary>
    public bool ShouldCacheCast(Tensor weight) => CacheWeightCasts && _weightCache.ContainsKey(weight);

    /// <summary>Returns a previously-cached dtype-cast of <paramref name="weight"/> for <paramref name="want"/>, if present.</summary>
    public bool TryGetWeightCast(Tensor weight, DType want, out VulkanBuffer buffer)
    {
        buffer = null!;
        if (!CacheWeightCasts) return false;
        if (_weightCastCache.TryGetValue(weight, out Dictionary<string, VulkanBuffer>? inner)
            && inner.TryGetValue(want.Name, out VulkanBuffer? hit))
        {
            buffer = hit;
            _hits++;
            return true;
        }
        return false;
    }

    /// <summary>Caches the cast result <paramref name="buffer"/> for (<paramref name="weight"/>, <paramref name="want"/>). The buffer is promoted to a cached buffer so <see cref="FreeDevice"/> won't reclaim it.</summary>
    public void StoreWeightCast(Tensor weight, DType want, VulkanBuffer buffer)
    {
        if (!_weightCastCache.TryGetValue(weight, out Dictionary<string, VulkanBuffer>? inner))
        {
            inner = new Dictionary<string, VulkanBuffer>();
            _weightCastCache[weight] = inner;
        }
        inner[want.Name] = buffer;
        _cachedBuffers.Add(buffer);
        _cachedBytes += (long)buffer.Size;
    }

    /// <summary>Uploads a weight tensor to GPU and caches it permanently.</summary>
    public void PreloadWeight(Tensor weight)
    {
        if (_weightCache.ContainsKey(weight)) return;
        ulong byteSize = (ulong)ByteSize(weight);
        VulkanBuffer dst = AllocateDevice(byteSize);
        Upload(dst, weight);
        _weightCache[weight] = dst;
        _cachedBuffers.Add(dst);
        _cachedBytes += (long)byteSize;
    }

    /// <summary>Bulk preloads weights — called by VulkanBackend.PreloadWeights.</summary>
    public void PreloadWeights(IEnumerable<Tensor> weights)
    {
        foreach (Tensor w in weights) PreloadWeight(w);
        _stream.SubmitAndAdvance();
    }

    /// <summary>Frees specific weight tensors from the GPU cache.</summary>
    public void FreeWeights(IEnumerable<Tensor> weights)
    {
        _stream.WaitIdleHost();
        foreach (Tensor w in weights)
        {
            if (_weightCache.Remove(w, out VulkanBuffer? buf))
            {
                _cachedBuffers.Remove(buf);
                _cachedBytes -= (long)buf.Size;
                buf.Dispose();
            }
            if (_weightCastCache.Remove(w, out Dictionary<string, VulkanBuffer>? casts))
            {
                foreach (VulkanBuffer cast in casts.Values)
                {
                    _cachedBuffers.Remove(cast);
                    _cachedBytes -= (long)cast.Size;
                    cast.Dispose();
                }
            }
        }
    }

    /// <summary>Drops every cached buffer (weights + activations) and any pending transients.</summary>
    public void FreeAllCached()
    {
        _stream.WaitIdleHost();
        // Neutralize lazy-sync / dispose callbacks on cached tensors before tearing down the
        // buffers — otherwise a late Tensor finalizer (running after VulkanBackend.Dispose) would
        // invoke a lambda closing over disposed device/buffer state and crash with
        // "vkDestroyBuffer: Invalid device".
        foreach (Tensor t in _activationCache.Keys)
        {
            t._gpuSyncCallback = null;
            t._gpuDisposeCallback = null;
        }
        foreach (VulkanBuffer b in _cachedBuffers) b.Dispose();
        // Synchronously dispose any uncached upload buffers too — these would otherwise leak until
        // GC, at which point VulkanBuffer's finalizer would call vkDestroyBuffer against a
        // device that has already been destroyed by VulkanBackend.Dispose.
        foreach (VulkanBuffer t in _transientBuffers) t.Dispose();
        _transientBuffers.Clear();
        _weightCache.Clear();
        _activationCache.Clear();
        _weightCastCache.Clear();
        _cachedBuffers.Clear();
        _cachedBytes = 0;
        _hits = 0;
        _misses = 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    // ComputeByteCount handles both plain dtypes (ElementCount * SizeInBytes) and GGUF-quantized ones
    // (whose SizeInBytes is 0 — the real per-element size is packed as BlockByteSize/BlockElementCount).
    // A prior version of this method used ElementCount * SizeInBytes directly, which silently computed a
    // ZERO byte size for any quantized tensor (Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K) — every upload of one would
    // either throw ("Allocation size must be > 0") or, worse, allocate nothing and read/write garbage,
    // undetected until the Phase 5 GGUF dequant shader tests exercised a quantized tensor for the first
    // time this backend had ever actually been asked to upload one.
    public static long ByteSize(Tensor t) => t.DType.ComputeByteCount(t.ElementCount);

    public (long cachedBytes, long hits, long misses) GetStats() => (_cachedBytes, _hits, _misses);

    public void Dispose() => FreeAllCached();
}

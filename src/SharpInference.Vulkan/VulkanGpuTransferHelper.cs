using System.Runtime.CompilerServices;
using SharpInference.Core.Tensors;

namespace SharpInference.Vulkan;

/// <summary>
/// Vulkan analogue of <c>SharpInference.Cuda.GpuTransferHelper</c>. Holds two caches keyed by Tensor reference:
/// <list type="bullet">
///   <item><description>Weight cache: permanent until <see cref="FreeWeights"/> / <see cref="FreeAllCached"/>.</description></item>
///   <item><description>Activation cache: set by <see cref="CacheActivation"/> after each op, consumed by next op's <see cref="CopyToDevice"/>.</description></item>
/// </list>
/// Lazy-sync callbacks on the Tensor mirror CUDA's behavior so model code may freely access <c>DataPointer</c>.
/// </summary>
public sealed class VulkanGpuTransferHelper : IDisposable
{
    private readonly Dictionary<Tensor, VulkanBuffer> _weightCache = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<Tensor, VulkanBuffer> _activationCache = new(ReferenceEqualityComparer.Instance);
    private readonly HashSet<VulkanBuffer> _cachedBuffers = new();

    private readonly nint _device;
    private readonly VulkanMemoryAllocator _allocator;
    private readonly VkPhysicalDeviceMemoryProperties _memProps;
    private readonly VulkanCapabilities _caps;
    private readonly VulkanCommandStream _stream;

    private long _cachedBytes;
    private long _hits;
    private long _misses;

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
        return dst;
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
        // Capture the CPU buffer pointer up front before installing callbacks.
        // The Tensor must have a CPU-owned buffer (created via the standard Tensor ctor); verified by Tensor.OwnsMemory.
        void* cpuPtr = tensor.DataPointer;
        ulong byteSize = (ulong)ByteSize(tensor);

        // Old callbacks may close over an earlier GPU buffer (in-place ops on the same Tensor). Clear before re-cache.
        tensor._gpuSyncCallback = null;
        tensor._gpuDisposeCallback = null;

        _activationCache[tensor] = buffer;
        _cachedBuffers.Add(buffer);

        VulkanGpuTransferHelper self = this;
        Tensor capturedTensor = tensor;
        VulkanBuffer capturedBuffer = buffer;
        nint capturedCpuPtr = (nint)cpuPtr;
        ulong capturedSize = byteSize;

        // Lazy sync: when CPU code reads the tensor, wait for the stream, copy back, free GPU buffer.
        tensor._gpuSyncCallback = () =>
        {
            if (self._activationCache.Remove(capturedTensor, out VulkanBuffer? cached))
            {
                self._stream.WaitIdleHost();
                self.DownloadToHost(capturedCpuPtr, cached, capturedSize);
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

    private void FlushIfNonCoherent(VulkanBuffer buffer)
    {
        VkMemoryType mt = _memProps.GetMemoryType((int)buffer.Allocation.MemoryTypeIndex);
        if ((mt.propertyFlags & VkMemoryPropertyFlags.HostCoherent) != 0) return;

        VkMappedMemoryRange range = new()
        {
            sType = VkStructureType.MappedMemoryRange,
            memory = buffer.Allocation.DeviceMemory,
            offset = AlignDown(buffer.Allocation.Offset, _caps.NonCoherentAtomSize),
            size = AlignUp(buffer.Size, _caps.NonCoherentAtomSize),
        };
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

        VkMappedMemoryRange range = new()
        {
            sType = VkStructureType.MappedMemoryRange,
            memory = buffer.Allocation.DeviceMemory,
            offset = AlignDown(buffer.Allocation.Offset, _caps.NonCoherentAtomSize),
            size = AlignUp(buffer.Size, _caps.NonCoherentAtomSize),
        };
        unsafe
        {
            VulkanApi.vkInvalidateMappedMemoryRanges(_device, 1, (nint)(&range))
                .ThrowOnError("vkInvalidateMappedMemoryRanges");
        }
    }

    private static ulong AlignUp(ulong v, ulong a) => (v + a - 1) & ~(a - 1);
    private static ulong AlignDown(ulong v, ulong a) => v & ~(a - 1);

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
        }
    }

    /// <summary>Drops every cached buffer (weights + activations).</summary>
    public void FreeAllCached()
    {
        _stream.WaitIdleHost();
        foreach (VulkanBuffer b in _cachedBuffers) b.Dispose();
        _weightCache.Clear();
        _activationCache.Clear();
        _cachedBuffers.Clear();
        _cachedBytes = 0;
        _hits = 0;
        _misses = 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long ByteSize(Tensor t) => t.ElementCount * t.DType.SizeInBytes;

    public (long cachedBytes, long hits, long misses) GetStats() => (_cachedBytes, _hits, _misses);

    public void Dispose() => FreeAllCached();
}

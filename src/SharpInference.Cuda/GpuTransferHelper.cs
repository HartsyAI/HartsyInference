using System.Runtime.CompilerServices;
using SharpInference.Core.Tensors;

namespace SharpInference.Cuda;

/// <summary>
/// GPU memory transfer helper with weight and activation caching.
/// Weight cache: preloaded via PreloadWeight(), permanent until FreeAllCached().
/// Activation cache: set by CacheActivation() after each op, consumed by next op's CopyToDevice().
/// Lazy sync: if CPU code accesses DataPointer, activation data syncs GPU→CPU on demand.
/// </summary>
internal static unsafe class GpuTransferHelper
{
    /// <summary>Cache mapping Tensor object references to GPU device pointers (weights — permanent).</summary>
    private static readonly Dictionary<Tensor, ulong> _weightCache = new(ReferenceEqualityComparer.Instance);

    /// <summary>Cache mapping Tensor object references to GPU activation data from previous ops.</summary>
    private static readonly Dictionary<Tensor, (ulong gpuPtr, nuint bytes)> _activationCache = new(ReferenceEqualityComparer.Instance);

    /// <summary>Set of GPU pointers that belong to either cache (skip in FreeDevice).</summary>
    private static readonly HashSet<ulong> _cachedPointers = new();

    /// <summary>Stream handle for deferred GPU memory frees and sync-before-D2H.</summary>
    private static nint _streamHandle;

    private static long _cachedBytes;
    private static long _hits;
    private static long _misses;

    /// <summary>Sets the CUDA stream handle used for FreeAsync and sync-before-D2H in lazy callbacks.</summary>
    public static void SetStream(nint stream) => _streamHandle = stream;

    /// <summary>Synchronizes the CUDA stream to flush pending FreeAsync operations. Called by CudaMemory.Allocate on OOM retry.</summary>
    public static void SyncStream()
    {
        if (_streamHandle != 0)
            CudaDriverApi.cuStreamSynchronize(_streamHandle).ThrowOnError();
    }

    /// <summary>
    /// Returns the GPU device pointer for a tensor, using caches to avoid transfers.
    /// Priority: weight cache → activation cache → fresh H2D transfer.
    /// </summary>
    public static ulong CopyToDevice(Tensor cpuTensor)
    {
        // 1. Weight cache (permanent, highest priority)
        if (_weightCache.TryGetValue(cpuTensor, out ulong cached))
        {
            _hits++;
            return cached;
        }

        // 2. Activation cache (GPU data from previous op — zero-copy reuse)
        if (_activationCache.TryGetValue(cpuTensor, out (ulong gpuPtr, nuint bytes) activation))
        {
            _hits++;
            return activation.gpuPtr;
        }

        // 3. Cache miss — fresh H2D transfer
        _misses++;
        nuint byteSize = ByteSize(cpuTensor);
        ulong dptr = CudaMemory.Allocate(byteSize);
        CudaMemory.CopyHostToDevice(dptr, cpuTensor.DataPointer, byteSize);
        return dptr;
    }

    /// <summary>Copies data from a GPU buffer back into a CPU tensor.</summary>
    public static void CopyToHost(Tensor cpuTensor, ulong gpuPtr, nuint byteSize)
    {
        CudaMemory.CopyDeviceToHost(cpuTensor.DataPointer, gpuPtr, byteSize);
    }

    /// <summary>Allocates a GPU buffer.</summary>
    public static ulong AllocateDevice(nuint byteSize)
    {
        return CudaMemory.Allocate(byteSize);
    }

    /// <summary>Frees a GPU buffer asynchronously on the compute stream. Skips cached pointers (weight + activation).</summary>
    public static void FreeDevice(ulong gpuPtr)
    {
        if (gpuPtr != 0 && !_cachedPointers.Contains(gpuPtr))
        {
            CudaMemory.FreeAsync(gpuPtr, _streamHandle);
        }
    }

    /// <summary>
    /// Caches an op's output GPU pointer with the tensor, avoiding D2H transfer.
    /// Sets lazy sync callbacks: DataPointer access triggers D2H, Dispose frees GPU memory.
    /// </summary>
    public static void CacheActivation(Tensor tensor, ulong gpuPtr, nuint byteSize)
    {
        // Capture CPU buffer pointer before setting callbacks.
        // Tensor is freshly created (no existing callback), so DataPointer is safe here.
        void* cpuPtr = tensor.DataPointer;

        _activationCache[tensor] = (gpuPtr, byteSize);
        _cachedPointers.Add(gpuPtr);

        // Lazy sync: when CPU code accesses DataPointer, wait for stream, copy GPU→CPU, then free.
        // Stream sync is needed because per-op Sync() has been removed — the producing kernel may still be in flight.
        tensor._gpuSyncCallback = () =>
        {
            if (_activationCache.Remove(tensor, out (ulong gpuPtr, nuint bytes) cached))
            {
                CudaDriverApi.cuStreamSynchronize(_streamHandle).ThrowOnError();
                CudaMemory.CopyDeviceToHost(cpuPtr, cached.gpuPtr, cached.bytes);
                _cachedPointers.Remove(cached.gpuPtr);
                CudaMemory.FreeAsync(cached.gpuPtr, _streamHandle);
            }
        };

        // On dispose without sync: free GPU memory asynchronously (skip D2H — data not needed)
        tensor._gpuDisposeCallback = () =>
        {
            if (_activationCache.Remove(tensor, out (ulong gpuPtr, nuint bytes) cached))
            {
                _cachedPointers.Remove(cached.gpuPtr);
                CudaMemory.FreeAsync(cached.gpuPtr, _streamHandle);
            }
        };
    }

    /// <summary>Uploads a weight tensor to GPU and caches it for future CopyToDevice calls.</summary>
    public static void PreloadWeight(Tensor weight)
    {
        if (_weightCache.ContainsKey(weight))
            return; // Already cached

        nuint byteSize = ByteSize(weight);
        ulong dptr = CudaMemory.Allocate(byteSize);
        CudaMemory.CopyHostToDevice(dptr, weight.DataPointer, byteSize);

        _weightCache[weight] = dptr;
        _cachedPointers.Add(dptr);
        _cachedBytes += (long)byteSize;
    }

    /// <summary>Frees specific weight tensors from the GPU cache to reclaim VRAM.</summary>
    public static void FreeWeights(IEnumerable<Tensor> weights)
    {
        if (_streamHandle != 0)
            CudaDriverApi.cuStreamSynchronize(_streamHandle).ThrowOnError();

        foreach (Tensor weight in weights)
        {
            if (_weightCache.Remove(weight, out ulong dptr))
            {
                _cachedPointers.Remove(dptr);
                CudaMemory.Free(dptr);
                _cachedBytes -= (long)ByteSize(weight);
            }
        }
    }

    /// <summary>Frees all cached GPU buffers (weights + activations) and clears all caches.</summary>
    public static void FreeAllCached()
    {
        // Sync stream before freeing — pending async work may still reference these buffers
        if (_streamHandle != 0)
        {
            CudaDriverApi.cuStreamSynchronize(_streamHandle).ThrowOnError();
        }

        foreach (ulong dptr in _cachedPointers)
        {
            CudaMemory.Free(dptr);
        }
        _weightCache.Clear();
        _activationCache.Clear();
        _cachedPointers.Clear();
        _cachedBytes = 0;
        _hits = 0;
        _misses = 0;
    }

    /// <summary>Evicts all cached GPU buffers.</summary>
    public static void EvictAll()
    {
        FreeAllCached();
    }

    /// <summary>Computes the byte size of a tensor's data.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static nuint ByteSize(Tensor tensor)
    {
        return (nuint)(tensor.ElementCount * tensor.DType.SizeInBytes);
    }

    /// <summary>Returns GPU cache statistics.</summary>
    public static (long cachedBytes, long hits, long misses) GetStats()
    {
        return (_cachedBytes, _hits, _misses);
    }
}

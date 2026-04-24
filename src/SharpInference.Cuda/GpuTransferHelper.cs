using System.Runtime.CompilerServices;
using SharpInference.Core.Tensors;

namespace SharpInference.Cuda;

/// <summary>
/// GPU memory transfer helper with weight caching support.
/// When weights are preloaded via PreloadWeight(), they remain on GPU and are reused
/// across operations. Non-cached tensors follow the original pattern: fresh alloc + copy per op.
/// </summary>
internal static unsafe class GpuTransferHelper
{
    /// <summary>Cache mapping Tensor object references to GPU device pointers.</summary>
    private static readonly Dictionary<Tensor, ulong> _weightCache = new(ReferenceEqualityComparer.Instance);

    /// <summary>Set of GPU pointers that belong to the cache (skip in FreeDevice).</summary>
    private static readonly HashSet<ulong> _cachedPointers = new();

    private static long _cachedBytes;
    private static long _hits;
    private static long _misses;

    /// <summary>
    /// Allocates a GPU buffer and copies CPU tensor data to it.
    /// If the tensor is in the weight cache, returns the cached GPU pointer (no transfer).
    /// Cache check happens BEFORE accessing DataPointer, so disposed tensors with cached weights work.
    /// </summary>
    public static ulong CopyToDevice(Tensor cpuTensor)
    {
        // Check cache FIRST (before touching DataPointer — tensor may be disposed with cached GPU data)
        if (_weightCache.TryGetValue(cpuTensor, out ulong cached))
        {
            _hits++;
            return cached;
        }

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

    /// <summary>Frees a GPU buffer. Skips cached weight pointers — they are managed by FreeAllCached.</summary>
    public static void FreeDevice(ulong gpuPtr)
    {
        if (gpuPtr != 0 && !_cachedPointers.Contains(gpuPtr))
        {
            CudaMemory.Free(gpuPtr);
        }
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

    /// <summary>Frees all cached GPU weight buffers and clears the cache.</summary>
    public static void FreeAllCached()
    {
        foreach (ulong dptr in _cachedPointers)
        {
            CudaMemory.Free(dptr);
        }
        _weightCache.Clear();
        _cachedPointers.Clear();
        _cachedBytes = 0;
        _hits = 0;
        _misses = 0;
    }

    /// <summary>Evicts all cached GPU weight buffers.</summary>
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

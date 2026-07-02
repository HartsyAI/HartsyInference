using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>GPU memory transfer helper with weight and activation caching. Weights preload via PreloadWeight() and stay until FreeAllCached(); activations set by CacheActivation() after each op and are consumed by the next op's CopyToDevice(). Lazy sync: CPU access to DataPointer triggers a GPU→CPU sync on demand.</summary>
internal static unsafe class GpuTransferHelper
{
    /// <summary>Cache mapping Tensor object references to GPU device pointers (weights — permanent).</summary>
    private static readonly Dictionary<Tensor, ulong> _weightCache = new(ReferenceEqualityComparer.Instance);

    /// <summary>Cache mapping Tensor object references to GPU activation data from previous ops.</summary>
    private static readonly Dictionary<Tensor, (ulong gpuPtr, nuint bytes)> _activationCache = new(ReferenceEqualityComparer.Instance);

    /// <summary>Cache of dtype-upcast copies of preloaded weights (e.g. fp8 → BF16 for tensor-core GEMM).
    /// The cast result is identical every forward, so it is computed once and reused — avoiding a per-Linear
    /// re-cast of the whole 9.3B weight set on every denoise step. Keyed by the source weight tensor; freed
    /// alongside its weight in <see cref="FreeWeights"/> / <see cref="FreeAllCached"/>.</summary>
    private static readonly Dictionary<Tensor, ulong> _weightCastCache = new(ReferenceEqualityComparer.Instance);

    /// <summary>Set of GPU pointers that belong to either cache (skip in FreeDevice).</summary>
    private static readonly HashSet<ulong> _cachedPointers = new();

    /// <summary>Stream handle for deferred GPU memory frees and sync-before-D2H.</summary>
    private static nint _streamHandle;

    /// <summary>Streaming cache reference, used to drain its upload stream + trim the
    /// device's stream-ordered allocator pool when an OOM retry needs to reclaim
    /// memory locked up in pool reservations. Null when the backend's streaming
    /// cache hasn't been wired (test setups, CPU/Vulkan).</summary>
    private static IStreamingWeightCache? _streamingCache;

    /// <summary>The active CUDA context. Held so the lazy sync/dispose callbacks (which fire
    /// from arbitrary threads — finalizers, async continuations, etc.) can bind the context
    /// before issuing any CUDA Driver API call. Without this, a callback that fires on a
    /// thread that's never bound the context would hit CUDA_ERROR_INVALID_CONTEXT.</summary>
    private static CudaContext? _context;

    private static long _cachedBytes;
    private static long _hits;
    private static long _misses;

    /// <summary>Count of lazy D2H sync callbacks fired (each forces a cuStreamSynchronize + device-to-host copy).
    /// A residency-health metric: during a fully GPU-resident denoise loop this must stay at ~0. Reset via
    /// <see cref="ResetSyncCount"/> and read via <see cref="GetSyncCount"/>.</summary>
    private static long _d2hSyncs;

    /// <summary>Sets the CUDA stream handle used for FreeAsync and sync-before-D2H in lazy callbacks.</summary>
    public static void SetStream(nint stream) => _streamHandle = stream;

    /// <summary>Sets the CUDA context that the lazy callbacks bind on demand. Called once
    /// from <see cref="CudaBackend"/>'s constructor.</summary>
    public static void SetContext(CudaContext context) => _context = context;

    /// <summary>Sets the streaming cache so the OOM retry path can drain its upload
    /// stream and trim the device mempool. Called once from <see cref="CudaBackend"/>'s
    /// constructor; safe to leave null on test setups that don't construct a backend.</summary>
    public static void SetStreamingCache(IStreamingWeightCache cache) => _streamingCache = cache;

    /// <summary>Synchronizes the CUDA stream to flush pending FreeAsync operations. Called by CudaMemory.Allocate on OOM retry.</summary>
    public static void SyncStream()
    {
        if (_streamHandle != 0)
        {
            _context?.EnsureCurrent();
            CudaDriverApi.cuStreamSynchronize(_streamHandle).ThrowOnError();
        }
    }

    /// <summary>OOM-retry hook: drains both the compute stream and the streaming cache's
    /// upload stream, then trims the device mempool so memory queued via
    /// <c>cuMemFreeAsync</c> is released back to the driver allocator. Called from
    /// <see cref="CudaMemory.Allocate"/> when the first <c>cuMemAlloc</c> returned OOM.
    /// Without this, an op that should succeed against just-evicted streaming memory
    /// will throw OOM even though several GB are technically free.</summary>
    public static void SyncStreamsAndReleasePool()
    {
        _context?.EnsureCurrent();
        SyncStream();
        // Cache also drains its own upload stream + calls cuMemPoolTrimTo on the
        // default mempool. No-op if no streaming cache is wired (CPU/Vulkan, tests).
        _streamingCache?.DrainAndReleasePool();
    }

    /// <summary>Returns the GPU device pointer for a tensor, using caches to avoid transfers. Priority: weight cache → activation cache → fresh H2D transfer.</summary>
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

        // 3. Cache miss — fresh H2D transfer. The buffer is a transient (the caller frees it via the async
        // FreeDevice), so allocate it from the stream-ordered pool. cuMemAllocAsync is stream-ordered, so sync the
        // stream before the synchronous host→device copy to guarantee the allocation has completed.
        _misses++;
        nuint byteSize = ByteSize(cpuTensor);
        ulong dptr = CudaMemory.Allocate(byteSize);
        if (_streamHandle != 0) SyncStream();
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

    /// <summary>Caches an op's output GPU pointer on the tensor, avoiding D2H transfer. Sets lazy callbacks: DataPointer access triggers D2H, Dispose frees GPU memory.</summary>
    public static void CacheActivation(Tensor tensor, ulong gpuPtr, nuint byteSize)
    {
        // In-place op re-caching its own output (e.g. backend.Gelu(x, x) / AffineBroadcastLastDim(x, x, …)): the
        // tensor already maps to its OLD device buffer. Drop that old pointer from the cached set WITHOUT freeing it
        // here — the calling op's `finally FreeDevice(pInput)` then frees it exactly once (FreeDevice only skips
        // pointers still in _cachedPointers). Leaving it would orphan the old buffer: no tensor maps to it, so
        // neither Dispose nor FreeActivations nor GC ever reclaims it → a permanent per-op device-memory leak (this
        // was the Wan full-res multi-step OOM; latent in every in-place backend op across LLM/Vision/Diffusion).
        if (gpuPtr != 0 && _activationCache.TryGetValue(tensor, out (ulong gpuPtr, nuint bytes) prev) && prev.gpuPtr != gpuPtr)
            _cachedPointers.Remove(prev.gpuPtr);

        // Do NOT touch tensor.DataPointer here: that would force the lazy host buffer to allocate (and zero) for
        // every GPU-resident activation, the exact host malloc+memset cost we are avoiding. The host buffer is
        // allocated only if/when CPU code actually reads the tensor, inside the sync callback below.
        _activationCache[tensor] = (gpuPtr, byteSize);
        _cachedPointers.Add(gpuPtr);

        // Lazy sync: when CPU code accesses DataPointer, wait for stream, copy GPU→CPU, then free.
        // Stream sync is needed because per-op Sync() has been removed — the producing kernel may still be in flight.
        // EnsureCurrent in both callbacks: they fire from whatever thread later reads/disposes
        // the tensor (potentially the GC finalizer thread), which won't have bound the context.
        tensor._gpuSyncCallback = () =>
        {
            if (_activationCache.Remove(tensor, out (ulong gpuPtr, nuint bytes) cached))
            {
                _d2hSyncs++;
                _context?.EnsureCurrent();
                CudaDriverApi.cuStreamSynchronize(_streamHandle).ThrowOnError();
                // Allocate the host destination only now, on the first real CPU read of this activation.
                void* cpuPtr = tensor.EnsureHostBuffer();
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
                _context?.EnsureCurrent();
                _cachedPointers.Remove(cached.gpuPtr);
                CudaMemory.FreeAsync(cached.gpuPtr, _streamHandle);
            }
        };
    }

    /// <summary>Returns a cached dtype-upcast of a weight (e.g. fp8→BF16), if one was already computed.</summary>
    public static bool TryGetWeightCast(Tensor weight, out ulong castPtr) => _weightCastCache.TryGetValue(weight, out castPtr);

    /// <summary>Records a dtype-upcast of a weight so subsequent forwards reuse it instead of re-casting.
    /// The pointer is tracked as cached so <see cref="FreeDevice"/> won't reclaim it as a transient.</summary>
    public static void CacheWeightCast(Tensor weight, ulong castPtr, nuint byteSize)
    {
        _weightCastCache[weight] = castPtr;
        _cachedPointers.Add(castPtr);
        _cachedBytes += (long)byteSize;
    }

    /// <summary>Uploads a weight tensor to GPU and caches it for future CopyToDevice calls.</summary>
    public static void PreloadWeight(Tensor weight)
    {
        if (_weightCache.ContainsKey(weight))
            return; // Already cached

        nuint byteSize = ByteSize(weight);
        // Resident weights are freed with the synchronous Free (FreeWeights/FreeAllCached), so they must be
        // allocated synchronously too — keep them out of the stream-ordered transient pool.
        ulong dptr = CudaMemory.AllocatePersistent(byteSize);
        CudaMemory.CopyHostToDevice(dptr, weight.DataPointer, byteSize);

        RegisterCachedWeight(weight, dptr, byteSize);
    }

    // ── Cache-state hooks for the streaming weight cache ────────────────
    //
    // The streaming cache (CudaStreamingWeightCache) does its own async alloc + memcpy
    // on a side stream rather than going through PreloadWeight's synchronous path, but
    // the bookkeeping that follows (registering in _weightCache so MatMul etc. find the
    // dptr, tracking _cachedPointers so FreeDevice doesn't free, accumulating
    // _cachedBytes) needs to stay in sync. Exposing these as internal helpers keeps a
    // single source of truth for the cache state without forcing the streaming cache
    // to reach into private fields.

    /// <summary>True if the weight is currently cached on the device. Streaming
    /// uploads check this to skip already-resident tensors.</summary>
    internal static bool IsWeightCached(Tensor weight) => _weightCache.ContainsKey(weight);

    /// <summary>Registers an already-uploaded weight in the cache. The caller is
    /// responsible for the alloc + H2D copy (sync or async); this just records the
    /// tensor → dptr mapping and bumps the byte counter.</summary>
    internal static void RegisterCachedWeight(Tensor weight, ulong dptr, nuint byteSize)
    {
        _weightCache[weight] = dptr;
        _cachedPointers.Add(dptr);
        _cachedBytes += (long)byteSize;
    }

    /// <summary>Removes a weight from the cache and returns its dptr, leaving the
    /// caller responsible for the actual <c>cuMemFree*</c> call. Returns <c>false</c>
    /// if the weight wasn't cached.</summary>
    internal static bool TryUnregisterCachedWeight(Tensor weight, out ulong dptr)
    {
        if (_weightCache.Remove(weight, out dptr))
        {
            _cachedPointers.Remove(dptr);
            _cachedBytes -= (long)ByteSize(weight);
            return true;
        }
        dptr = 0;
        return false;
    }

    /// <summary>Frees specific weight tensors from the GPU cache to reclaim VRAM.</summary>
    public static void FreeWeights(IEnumerable<Tensor> weights)
    {
        _context?.EnsureCurrent();
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
            if (_weightCastCache.Remove(weight, out ulong castPtr))
            {
                _cachedPointers.Remove(castPtr);
                CudaMemory.Free(castPtr);
            }
        }
    }

    /// <summary>Frees all cached GPU buffers (weights + activations) and clears all caches.</summary>
    public static void FreeAllCached()
    {
        _context?.EnsureCurrent();
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
        _weightCastCache.Clear();
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

    /// <summary>Frees only cached ACTIVATION device buffers; preloaded weights and weight-casts are kept. Call
    /// between denoise steps to deterministically reclaim device memory held by activations that were neither read
    /// back to host (which frees via the sync callback) nor explicitly disposed — those otherwise linger in the
    /// cache until non-deterministic GC finalization and accumulate to OOM over multi-step diffusion. Safe because
    /// the only cross-step state (the latent) lives on the host; anything still cached here is dead. The per-tensor
    /// sync/dispose callbacks stay valid: they re-check <c>_activationCache</c> and no-op once the entry is gone.</summary>
    public static void FreeActivations()
    {
        _context?.EnsureCurrent();
        foreach (KeyValuePair<Tensor, (ulong gpuPtr, nuint bytes)> kv in _activationCache)
        {
            _cachedPointers.Remove(kv.Value.gpuPtr);
            CudaMemory.FreeAsync(kv.Value.gpuPtr, _streamHandle);
        }
        _activationCache.Clear();

        // Return pooled memory to the driver. cuMemFreeAsync (used by every activation/dispose free) hands memory
        // back to the stream-ordered mempool, which RESERVES it (cuMemGetInfo counts it as used) until trimmed —
        // otherwise the pool's high-water mark grows every op and multi-step diffusion OOMs even though the memory
        // is logically free. SyncStreamsAndReleasePool only trims when the streaming weight cache is wired (it isn't
        // for the video pipelines), so trim the default pool directly. Sync first so the queued async frees complete.
        // Return pooled memory to the driver so it counts as free (cuMemFreeAsync only hands blocks back to the
        // stream-ordered pool, which reserves them). Sync first so the queued async frees complete.
        TrimPool();
    }

    /// <summary>Returns pool-reserved-but-free device memory to the driver WITHOUT clearing the activation cache.
    /// <c>cuMemFreeAsync</c> (every activation/dispose free) hands blocks back to the stream-ordered mempool, which
    /// RESERVES them (counts as used in cuMemGetInfo) until trimmed. Unlike <see cref="FreeActivations"/> this leaves
    /// live cached activations intact — only already-freed blocks are reclaimed — so it is safe to call mid-computation
    /// (e.g. between VAE decode tiles) to cap peak at one unit's working set without corrupting tensors still in use.
    /// Syncs the stream first so queued async frees complete before the trim.</summary>
    public static void TrimPool()
    {
        if (_context is not null && _streamHandle != 0)
        {
            _context.EnsureCurrent();
            CudaDriverApi.cuStreamSynchronize(_streamHandle).ThrowOnError();
            if (CudaDriverApi.cuDeviceGetDefaultMemPool(out nint pool, _context.DeviceOrdinal) == 0)
                CudaDriverApi.cuMemPoolTrimTo(pool, 0);
        }
    }

    /// <summary>Computes the byte size of a tensor's data. Uses <see cref="DType.ComputeByteCount"/> so quantized tensors (Q4_K, Q5_K, Q8_0, etc.) report their true on-disk byte count rather than <c>elementCount * 0</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static nuint ByteSize(Tensor tensor)
    {
        return (nuint)tensor.DType.ComputeByteCount(tensor.ElementCount);
    }

    /// <summary>Returns GPU cache statistics.</summary>
    public static (long cachedBytes, long hits, long misses) GetStats()
    {
        return (_cachedBytes, _hits, _misses);
    }

    /// <summary>Number of lazy D2H sync callbacks fired since the last reset. Each one is a full GPU stall plus a
    /// device-to-host copy; a GPU-resident hot loop should fire none.</summary>
    public static long GetSyncCount() => _d2hSyncs;

    /// <summary>Resets the D2H sync counter (call at the start of a region you want to measure for residency).</summary>
    public static void ResetSyncCount() => _d2hSyncs = 0;
}

using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>GPU memory transfer helper with weight and activation caching. Weights preload via PreloadWeight() and stay until FreeAllCached(); activations set by CacheActivation() after each op and are consumed by the next op's CopyToDevice(). Lazy sync: CPU access to DataPointer triggers a GPU→CPU sync on demand.
/// <para><b>Multi-backend safety:</b> all mutable state lives in a per-CUDA-context <see cref="State"/> object,
/// registered by each <see cref="CudaBackend"/> at construction and resolved at every call via the calling
/// thread's CURRENT context (every CudaBackend op binds its context before calling in here — the long-standing
/// invariant this relies on). Activation callbacks capture their owning State directly, so a tensor produced on
/// backend A always syncs/frees against A's context and stream even when backend B was used more recently.
/// Previously this was a single set of static fields: constructing a second CudaBackend (e.g. a 3060 alongside
/// the 4090) silently retargeted ALL cached state and callbacks at the new context → cross-device frees →
/// CUDA_ERROR_ILLEGAL_ADDRESS on both GPUs and a poisoned process. Two backends on the SAME device share the
/// primary context handle and therefore a State — supported for caches (they're per-Tensor), with
/// last-registered-wins for the stream/streaming-cache bindings.</para></summary>
internal static unsafe class GpuTransferHelper
{
    /// <summary>All per-backend mutable state, keyed by the owning CUDA context.</summary>
    internal sealed class State
    {
        /// <summary>Cache mapping Tensor object references to GPU device pointers (weights — permanent).</summary>
        public readonly Dictionary<Tensor, ulong> WeightCache = new(ReferenceEqualityComparer.Instance);

        /// <summary>Cache mapping Tensor object references to GPU activation data from previous ops.</summary>
        public readonly Dictionary<Tensor, (ulong gpuPtr, nuint bytes)> ActivationCache = new(ReferenceEqualityComparer.Instance);

        /// <summary>Cache of dtype-upcast copies of preloaded weights (e.g. fp8 → BF16 for tensor-core GEMM).
        /// The cast result is identical every forward, so it is computed once and reused — avoiding a per-Linear
        /// re-cast of the whole 9.3B weight set on every denoise step. Keyed by the source weight tensor; freed
        /// alongside its weight in <see cref="FreeWeights"/> / <see cref="FreeAllCached"/>.</summary>
        public readonly Dictionary<Tensor, ulong> WeightCastCache = new(ReferenceEqualityComparer.Instance);

        /// <summary>Set of GPU pointers that belong to either cache (skip in FreeDevice).</summary>
        public readonly HashSet<ulong> CachedPointers = new();

        /// <summary>Stream handle for deferred GPU memory frees and sync-before-D2H.</summary>
        public nint StreamHandle;

        /// <summary>Streaming cache reference, used to drain its upload stream + trim the
        /// device's stream-ordered allocator pool when an OOM retry needs to reclaim
        /// memory locked up in pool reservations. Null when the backend's streaming
        /// cache hasn't been wired (test setups, CPU/Vulkan).</summary>
        public IStreamingWeightCache? StreamingCache;

        /// <summary>The owning CUDA context. Held so the lazy sync/dispose callbacks (which fire
        /// from arbitrary threads — finalizers, async continuations, etc.) can bind the context
        /// before issuing any CUDA Driver API call. Without this, a callback that fires on a
        /// thread that's never bound the context would hit CUDA_ERROR_INVALID_CONTEXT.</summary>
        public CudaContext? Context;

        public long CachedBytes;
        public long Hits;
        public long Misses;

        /// <summary>Count of lazy D2H sync callbacks fired (each forces a cuStreamSynchronize + device-to-host copy).
        /// A residency-health metric: during a fully GPU-resident denoise loop this must stay at ~0.</summary>
        public long D2hSyncs;
    }

    /// <summary>Registered states keyed by CUDA context handle. Concurrent: registration happens on backend
    /// construction threads while Resolve() reads from compute threads.</summary>
    private static readonly ConcurrentDictionary<nint, State> _states = new();

    /// <summary>Fast path: the single registered state when only one backend exists (the overwhelmingly
    /// common case), avoiding a cuCtxGetCurrent per call. Null when zero or 2+ states are registered.</summary>
    private static volatile State? _single;

    /// <summary>Fallback state for calls made before any backend registers (unit tests exercising pure
    /// helpers). Its stream is 0 and context null, so every code path degrades to the safe no-op branch.</summary>
    private static readonly State _unregistered = new();

    /// <summary>Registers (or re-binds) the state for a backend's context. Called once from
    /// <see cref="CudaBackend"/>'s constructor with the context, compute stream, and streaming cache.</summary>
    public static void Register(CudaContext context, nint stream, IStreamingWeightCache? streamingCache)
    {
        State state = _states.GetOrAdd(context.Handle, _ => new State());
        state.Context = context;
        state.StreamHandle = stream;
        state.StreamingCache = streamingCache;
        _single = _states.Count == 1 ? state : null;
    }

    /// <summary>Removes a context's state at backend disposal (after <see cref="FreeAllCached"/>).
    /// Only removes when the registered state still belongs to this context instance — a same-device
    /// backend that re-registered the handle keeps its binding.</summary>
    public static void Unregister(CudaContext context)
    {
        if (_states.TryGetValue(context.Handle, out State? state) && ReferenceEquals(state.Context, context))
        {
            _states.TryRemove(context.Handle, out _);
        }
        _single = _states.Count == 1 ? System.Linq.Enumerable.First(_states.Values) : null;
    }

    /// <summary>Resolves the state for the calling thread's CURRENT CUDA context. Every CudaBackend op
    /// (and the streaming cache) binds its own context via EnsureCurrent before calling in here, so the
    /// current context identifies the owning backend. Falls back to the single registered state, then to
    /// an inert empty state (pre-registration test paths).</summary>
    private static State Resolve()
    {
        State? single = _single;
        if (single is not null) return single;
        if (_states.IsEmpty) return _unregistered;
        if (CudaDriverApi.cuCtxGetCurrent(out nint current) == 0 && _states.TryGetValue(current, out State? state))
            return state;
        return _unregistered;
    }

    /// <summary>Synchronizes the CUDA stream to flush pending FreeAsync operations. Called by CudaMemory.Allocate on OOM retry.</summary>
    public static void SyncStream()
    {
        State s = Resolve();
        if (s.StreamHandle != 0)
        {
            s.Context?.EnsureCurrent();
            CudaDriverApi.cuStreamSynchronize(s.StreamHandle).ThrowOnError();
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
        State s = Resolve();
        s.Context?.EnsureCurrent();
        SyncStream();
        // Cache also drains its own upload stream + calls cuMemPoolTrimTo on the
        // default mempool. No-op if no streaming cache is wired (CPU/Vulkan, tests).
        s.StreamingCache?.DrainAndReleasePool();
        // Always trim the default pool too: with no streaming cache wired (auto-transfer paths, tests), every
        // cuMemFreeAsync'd transient stays RESERVED in the stream-ordered pool. An OOM retry that never trims
        // reports "GPU full" (cuMemGetInfo counts reservations as used) even though most of it is reusable —
        // the fp8 auto-transfer Flux/T5 OOM at 0.9% free.
        TrimPool();
    }

    /// <summary>Returns the GPU device pointer for a tensor, using caches to avoid transfers. Priority: weight cache → activation cache → fresh H2D transfer.</summary>
    public static ulong CopyToDevice(Tensor cpuTensor)
    {
        State s = Resolve();

        // 1. Weight cache (permanent, highest priority)
        if (s.WeightCache.TryGetValue(cpuTensor, out ulong cached))
        {
            s.Hits++;
            return cached;
        }

        // 2. Activation cache (GPU data from previous op — zero-copy reuse)
        if (s.ActivationCache.TryGetValue(cpuTensor, out (ulong gpuPtr, nuint bytes) activation))
        {
            s.Hits++;
            return activation.gpuPtr;
        }

        // 3. Cache miss — fresh H2D transfer. The buffer is a transient (the caller frees it via the async
        // FreeDevice), so allocate it from the stream-ordered pool. cuMemAllocAsync is stream-ordered, so sync the
        // stream before the synchronous host→device copy to guarantee the allocation has completed.
        s.Misses++;
        nuint byteSize = ByteSize(cpuTensor);
        ulong dptr = CudaMemory.Allocate(byteSize);
        if (s.StreamHandle != 0) SyncStream();
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
        State s = Resolve();
        if (gpuPtr != 0 && !s.CachedPointers.Contains(gpuPtr))
        {
            CudaMemory.FreeAsync(gpuPtr, s.StreamHandle);
        }
    }

    /// <summary>Caches an op's output GPU pointer on the tensor, avoiding D2H transfer. Sets lazy callbacks: DataPointer access triggers D2H, Dispose frees GPU memory. The callbacks capture this backend's <see cref="State"/>, so they stay correct even after another backend registers.</summary>
    public static void CacheActivation(Tensor tensor, ulong gpuPtr, nuint byteSize)
    {
        State s = Resolve();

        // In-place op re-caching its own output (e.g. backend.Gelu(x, x) / AffineBroadcastLastDim(x, x, …)): the
        // tensor already maps to its OLD device buffer. Drop that old pointer from the cached set WITHOUT freeing it
        // here — the calling op's `finally FreeDevice(pInput)` then frees it exactly once (FreeDevice only skips
        // pointers still in CachedPointers). Leaving it would orphan the old buffer: no tensor maps to it, so
        // neither Dispose nor FreeActivations nor GC ever reclaims it → a permanent per-op device-memory leak (this
        // was the Wan full-res multi-step OOM; latent in every in-place backend op across LLM/Vision/Diffusion).
        if (gpuPtr != 0 && s.ActivationCache.TryGetValue(tensor, out (ulong gpuPtr, nuint bytes) prev) && prev.gpuPtr != gpuPtr)
            s.CachedPointers.Remove(prev.gpuPtr);

        // Do NOT touch tensor.DataPointer here: that would force the lazy host buffer to allocate (and zero) for
        // every GPU-resident activation, the exact host malloc+memset cost we are avoiding. The host buffer is
        // allocated only if/when CPU code actually reads the tensor, inside the sync callback below.
        s.ActivationCache[tensor] = (gpuPtr, byteSize);
        s.CachedPointers.Add(gpuPtr);

        // Lazy sync: when CPU code accesses DataPointer, wait for stream, copy GPU→CPU, then free.
        // Stream sync is needed because per-op Sync() has been removed — the producing kernel may still be in flight.
        // EnsureCurrent in both callbacks: they fire from whatever thread later reads/disposes
        // the tensor (potentially the GC finalizer thread), which won't have bound the context.
        tensor._gpuSyncCallback = () =>
        {
            if (s.ActivationCache.Remove(tensor, out (ulong gpuPtr, nuint bytes) cached))
            {
                s.D2hSyncs++;
                s.Context?.EnsureCurrent();
                CudaDriverApi.cuStreamSynchronize(s.StreamHandle).ThrowOnError();
                // Allocate the host destination only now, on the first real CPU read of this activation.
                void* cpuPtr = tensor.EnsureHostBuffer();
                CudaMemory.CopyDeviceToHost(cpuPtr, cached.gpuPtr, cached.bytes);
                s.CachedPointers.Remove(cached.gpuPtr);
                CudaMemory.FreeAsync(cached.gpuPtr, s.StreamHandle);
            }
        };

        // On dispose without sync: free GPU memory asynchronously (skip D2H — data not needed)
        tensor._gpuDisposeCallback = () =>
        {
            if (s.ActivationCache.Remove(tensor, out (ulong gpuPtr, nuint bytes) cached))
            {
                s.Context?.EnsureCurrent();
                s.CachedPointers.Remove(cached.gpuPtr);
                CudaMemory.FreeAsync(cached.gpuPtr, s.StreamHandle);
            }
        };
    }

    /// <summary>Returns a cached dtype-upcast of a weight (e.g. fp8→BF16), if one was already computed.</summary>
    public static bool TryGetWeightCast(Tensor weight, out ulong castPtr) => Resolve().WeightCastCache.TryGetValue(weight, out castPtr);

    /// <summary>Records a dtype-upcast of a weight so subsequent forwards reuse it instead of re-casting.
    /// The pointer is tracked as cached so <see cref="FreeDevice"/> won't reclaim it as a transient.</summary>
    public static void CacheWeightCast(Tensor weight, ulong castPtr, nuint byteSize)
    {
        State s = Resolve();
        s.WeightCastCache[weight] = castPtr;
        s.CachedPointers.Add(castPtr);
        s.CachedBytes += (long)byteSize;
    }

    /// <summary>Uploads a weight tensor to GPU and caches it for future CopyToDevice calls.</summary>
    public static void PreloadWeight(Tensor weight)
    {
        State s = Resolve();
        if (s.WeightCache.ContainsKey(weight))
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
    // the bookkeeping that follows (registering in WeightCache so MatMul etc. find the
    // dptr, tracking CachedPointers so FreeDevice doesn't free, accumulating
    // CachedBytes) needs to stay in sync. Exposing these as internal helpers keeps a
    // single source of truth for the cache state without forcing the streaming cache
    // to reach into private fields.

    /// <summary>True if the weight is currently cached on the device. Streaming
    /// uploads check this to skip already-resident tensors.</summary>
    internal static bool IsWeightCached(Tensor weight) => Resolve().WeightCache.ContainsKey(weight);

    /// <summary>Registers an already-uploaded weight in the cache. The caller is
    /// responsible for the alloc + H2D copy (sync or async); this just records the
    /// tensor → dptr mapping and bumps the byte counter.</summary>
    internal static void RegisterCachedWeight(Tensor weight, ulong dptr, nuint byteSize)
    {
        State s = Resolve();
        s.WeightCache[weight] = dptr;
        s.CachedPointers.Add(dptr);
        s.CachedBytes += (long)byteSize;
    }

    /// <summary>Removes a weight from the cache and returns its dptr, leaving the
    /// caller responsible for the actual <c>cuMemFree*</c> call. Returns <c>false</c>
    /// if the weight wasn't cached. Also frees any cached dtype-cast of the weight:
    /// streamed blocks otherwise orphan their F16 casts on eviction (the cast is keyed
    /// by the Tensor and only reclaimed via <see cref="FreeWeights"/>, which streaming
    /// eviction doesn't call) — for a streamed 12B fp8 DiT that accumulated ~19 GB of
    /// dead casts by VAE-decode time and OOM'd the decode.</summary>
    internal static bool TryUnregisterCachedWeight(Tensor weight, out ulong dptr)
    {
        State s = Resolve();
        if (s.WeightCastCache.Remove(weight, out ulong castPtr))
        {
            s.CachedPointers.Remove(castPtr);
            // Stream-ordered free: the cast was allocated via the async pool and may be referenced by
            // GEMMs still in flight on the compute stream; FreeAsync orders the release after them.
            CudaMemory.FreeAsync(castPtr, s.StreamHandle);
        }
        if (s.WeightCache.Remove(weight, out dptr))
        {
            s.CachedPointers.Remove(dptr);
            s.CachedBytes -= (long)ByteSize(weight);
            return true;
        }
        dptr = 0;
        return false;
    }

    /// <summary>Frees specific weight tensors from the GPU cache to reclaim VRAM.</summary>
    public static void FreeWeights(IEnumerable<Tensor> weights)
    {
        State s = Resolve();
        s.Context?.EnsureCurrent();
        if (s.StreamHandle != 0)
            CudaDriverApi.cuStreamSynchronize(s.StreamHandle).ThrowOnError();

        foreach (Tensor weight in weights)
        {
            if (s.WeightCache.Remove(weight, out ulong dptr))
            {
                s.CachedPointers.Remove(dptr);
                CudaMemory.Free(dptr);
                s.CachedBytes -= (long)ByteSize(weight);
            }
            if (s.WeightCastCache.Remove(weight, out ulong castPtr))
            {
                s.CachedPointers.Remove(castPtr);
                CudaMemory.Free(castPtr);
            }
        }
    }

    /// <summary>Frees all cached GPU buffers (weights + activations) and clears all caches for the CURRENT backend.</summary>
    public static void FreeAllCached()
    {
        State s = Resolve();
        s.Context?.EnsureCurrent();
        // Sync stream before freeing — pending async work may still reference these buffers
        if (s.StreamHandle != 0)
        {
            CudaDriverApi.cuStreamSynchronize(s.StreamHandle).ThrowOnError();
        }

        foreach (ulong dptr in s.CachedPointers)
        {
            CudaMemory.Free(dptr);
        }
        s.WeightCache.Clear();
        s.ActivationCache.Clear();
        s.WeightCastCache.Clear();
        s.CachedPointers.Clear();
        s.CachedBytes = 0;
        s.Hits = 0;
        s.Misses = 0;
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
    /// sync/dispose callbacks stay valid: they re-check the activation cache and no-op once the entry is gone.</summary>
    public static void FreeActivations()
    {
        State s = Resolve();
        s.Context?.EnsureCurrent();
        foreach (KeyValuePair<Tensor, (ulong gpuPtr, nuint bytes)> kv in s.ActivationCache)
        {
            s.CachedPointers.Remove(kv.Value.gpuPtr);
            CudaMemory.FreeAsync(kv.Value.gpuPtr, s.StreamHandle);
        }
        s.ActivationCache.Clear();

        // Return pooled memory to the driver. cuMemFreeAsync (used by every activation/dispose free) hands memory
        // back to the stream-ordered mempool, which RESERVES it (cuMemGetInfo counts it as used) until trimmed —
        // otherwise the pool's high-water mark grows every op and multi-step diffusion OOMs even though the memory
        // is logically free. Sync first so the queued async frees complete.
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
        State s = Resolve();
        if (s.Context is not null && s.StreamHandle != 0)
        {
            s.Context.EnsureCurrent();
            CudaDriverApi.cuStreamSynchronize(s.StreamHandle).ThrowOnError();
            if (CudaDriverApi.cuDeviceGetDefaultMemPool(out nint pool, s.Context.DeviceOrdinal) == 0)
                CudaDriverApi.cuMemPoolTrimTo(pool, 0);
        }
    }

    /// <summary>Computes the byte size of a tensor's data. Uses <see cref="DType.ComputeByteCount"/> so quantized tensors (Q4_K, Q5_K, Q8_0, etc.) report their true on-disk byte count rather than <c>elementCount * 0</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static nuint ByteSize(Tensor tensor)
    {
        return (nuint)tensor.DType.ComputeByteCount(tensor.ElementCount);
    }

    /// <summary>Returns GPU cache statistics for the current backend.</summary>
    public static (long cachedBytes, long hits, long misses) GetStats()
    {
        State s = Resolve();
        return (s.CachedBytes, s.Hits, s.Misses);
    }

    /// <summary>Number of lazy D2H sync callbacks fired since the last reset. Each one is a full GPU stall plus a
    /// device-to-host copy; a GPU-resident hot loop should fire none.</summary>
    public static long GetSyncCount() => Resolve().D2hSyncs;

    /// <summary>Resets the D2H sync counter (call at the start of a region you want to measure for residency).</summary>
    public static void ResetSyncCount() => Resolve().D2hSyncs = 0;
}

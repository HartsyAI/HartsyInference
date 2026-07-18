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
        public readonly Dictionary<Tensor, (ulong castPtr, nuint bytes)> WeightCastCache = new(ReferenceEqualityComparer.Instance);

        /// <summary>Upload counts for host tensors that miss both caches. A tensor re-uploaded with unchanged host data
        /// is behaving like a weight, whoever created it — on its second upload it is promoted into the weight cache
        /// (see <see cref="TryAutoPromote"/>), making pipelines that never call <c>PreloadWeights</c> (the audio stack)
        /// GPU-resident instead of PCIe-bound. Weak-keyed so tracked tensors stay collectible; the state dies with its
        /// tensor. Per-State so promotion bookkeeping stays with the backend that owns the device copy.</summary>
        public readonly ConditionalWeakTable<Tensor, UploadState> UploadTracker = new();

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

        /// <summary>Set when this state's backend is torn down (<see cref="Unregister"/>). Stale promoted-weight
        /// callbacks that survived teardown (queued by tensor finalizers before their callbacks could be detached)
        /// check this FIRST and bail out — reading a bool field is safe on a resurrected object graph, whereas
        /// touching <see cref="UploadTracker"/> is not (a ConditionalWeakTable whose Container was finalized while
        /// the state was unreachable throws NRE from its freed dependent handles — the GGUF model-switch crash).</summary>
        public volatile bool Unregistered;

        public long CachedBytes;
        public long Hits;
        public long Misses;

        /// <summary>Count of lazy D2H sync callbacks fired (each forces a cuStreamSynchronize + device-to-host copy).
        /// A residency-health metric: during a fully GPU-resident denoise loop this must stay at ~0.</summary>
        public long D2hSyncs;
    }

    /// <summary>Per-tensor H2D upload bookkeeping for weight auto-promotion.</summary>
    internal sealed class UploadState { public int Count; public bool Promoted; public bool Blocked; }

    /// <summary>Auto-promotion kill switch: set <c>HARTSY_NO_AUTOPROMOTE=1</c> to reproduce the old always-re-upload
    /// behavior (A/B benchmarking, or if a pipeline mutates host weight data through a stashed raw pointer that
    /// bypasses <c>DataPointer</c>/<c>AsSpan</c> and so can't be seen by the demote-on-host-access hook).</summary>
    public static readonly bool AutoPromoteWeights = Environment.GetEnvironmentVariable("HARTSY_NO_AUTOPROMOTE") != "1";

    /// <summary>Free-VRAM floor preserved by auto-promotion (activations, transients, cuBLAS workspaces need room).
    /// A promotion that would dip below this floor is skipped and the tensor streams as before. Override via
    /// <c>HARTSY_AUTOPROMOTE_HEADROOM_MB</c>.</summary>
    private static readonly long _autoPromoteHeadroomBytes =
        long.TryParse(Environment.GetEnvironmentVariable("HARTSY_AUTOPROMOTE_HEADROOM_MB"), out long mb) ? mb << 20 : 1536L << 20;

    /// <summary>Tensors below this size are never auto-promoted: small hot tensors are cheap to re-upload and are
    /// the most likely to be mutated scratch buffers.</summary>
    private const nuint AutoPromoteMinBytes = 1 << 20;

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
        // A same-device re-registration revives a previously torn-down handle's state.
        state.Unregistered = false;
        _single = _states.Count == 1 ? state : null;
    }

    /// <summary>Removes a context's state at backend disposal (after <see cref="FreeAllCached"/>).
    /// Only removes when the registered state still belongs to this context instance — a same-device
    /// backend that re-registered the handle keeps its binding. Returns true when the state was removed,
    /// i.e. the caller was the handle's last owner and may discard the handle's pending-cleanup queue.</summary>
    public static bool Unregister(CudaContext context)
    {
        bool removed = false;
        if (_states.TryGetValue(context.Handle, out State? state) && ReferenceEquals(state.Context, context))
        {
            // Mark dead BEFORE removal: any promoted-weight callback that still fires (finalizer-queued before
            // its tensor could be detached) must see the flag and no-op instead of touching this state's caches.
            state.Unregistered = true;
            removed = _states.TryRemove(context.Handle, out _);
        }
        _single = _states.Count == 1 ? System.Linq.Enumerable.First(_states.Values) : null;
        return removed;
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

        // 3. Second-plus upload of the same unchanged host tensor → promote it to a resident cached weight.
        s.Misses++;
        nuint byteSize = ByteSize(cpuTensor);
        if (TryAutoPromote(s, cpuTensor, byteSize, out ulong promotedPtr))
        {
            return promotedPtr;
        }

        // 4. Cache miss — fresh H2D transfer. The buffer is a transient (the caller frees it via the async
        // FreeDevice). Allocate from the stream-ordered pool (cuMemAllocAsync on the compute stream) and copy with a
        // STREAM-ORDERED async H2D on the SAME stream — the copy is naturally ordered after the alloc, and the
        // consuming kernel (queued next on that stream) sees the data. This replaces a per-miss full
        // `cuStreamSynchronize` that drained the entire async pipeline on EVERY small host-tensor upload — the Wan
        // DiT alone missed ~14 tiny modulation/scratch tensors per block-forward, so that drain was ~94 s of a
        // ~63 s×... gen (dominant cost). Pageable src stages synchronously before returning (host buffer safe to
        // reuse); pinned src stays alive until the stream-ordered FreeDevice. No CPU read happens here, so no
        // correctness dependency on the copy completing before this returns — only stream order, which holds.
        using Profiling.NvtxRange _miss = Profiling.NvtxRange.Push(byteSize > (1u << 20) ? "H2D_MISS_BIG" : "H2D_MISS_SMALL");   // HARTSY_PROFILE visibility into miss H2D volume
        ulong dptr = CudaMemory.Allocate(byteSize);
        if (s.StreamHandle != 0)
            CudaMemory.CopyHostToDeviceAsync(dptr, cpuTensor.DataPointer, byteSize, s.StreamHandle);
        else
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

        // In-place op mutated an auto-promoted weight's device buffer (CopyToDevice returned the cached ptr, the
        // kernel wrote through it). The buffer's contents no longer match host data, so it can't stay a cached
        // weight: hand ownership to the activation cache (registered below) and block re-promotion. Any cached
        // dtype-cast of the old contents is stale — free it.
        if (gpuPtr != 0 && s.WeightCache.TryGetValue(tensor, out ulong promotedPtr) && promotedPtr == gpuPtr
            && s.UploadTracker.TryGetValue(tensor, out UploadState? promoState) && promoState.Promoted)
        {
            promoState.Promoted = false;
            promoState.Blocked = true;
            s.WeightCache.Remove(tensor);
            s.CachedBytes -= (long)byteSize;
            if (s.WeightCastCache.Remove(tensor, out (ulong castPtr, nuint bytes) staleCast))
            {
                s.CachedPointers.Remove(staleCast.castPtr);
                // Stream-ordered free: in-flight GEMMs may still read the stale cast.
                CudaMemory.FreeAsync(staleCast.castPtr, s.StreamHandle);
                s.CachedBytes -= (long)staleCast.bytes;
            }
        }

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
        // Route this tensor's finalizer cleanup into THIS backend's context bucket so a concurrent backend's
        // drain thread never runs it against the wrong (and unsynchronized) State.
        tensor._gpuCleanupContext = s.Context?.Handle ?? 0;
    }

    /// <summary>Returns a cached dtype-upcast of a weight (e.g. fp8→BF16), if one was already computed.</summary>
    public static bool TryGetWeightCast(Tensor weight, out ulong castPtr)
    {
        bool found = Resolve().WeightCastCache.TryGetValue(weight, out (ulong castPtr, nuint bytes) cast);
        castPtr = cast.castPtr;
        return found;
    }

    /// <summary>Records a dtype-upcast of a weight so subsequent forwards reuse it instead of re-casting.
    /// The pointer is tracked as cached so <see cref="FreeDevice"/> won't reclaim it as a transient.</summary>
    public static void CacheWeightCast(Tensor weight, ulong castPtr, nuint byteSize)
    {
        State s = Resolve();
        s.WeightCastCache[weight] = (castPtr, byteSize);
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

    /// <summary>Promotes a repeatedly-uploaded host tensor into the resident weight cache. Fires on the second
    /// cache-missing upload of the same tensor object: weights are the only tensors that live long enough to be
    /// uploaded twice (activations are fresh objects per op), so this catches every weight of pipelines that never
    /// call <see cref="PreloadWeight"/> at the cost of one duplicate upload. Correctness hinges on the demote hook:
    /// promotion plants <c>_gpuSyncCallback</c>/<c>_gpuDisposeCallback</c>, so ANY later CPU access (which always
    /// funnels through <c>EnsureCpuData</c>) or Dispose evicts the device copy before host data can diverge — and
    /// blocks re-promotion, so host-mutated scratch tensors settle back to plain streaming instead of thrashing.
    /// Skipped when the promotion would drop free VRAM below <see cref="_autoPromoteHeadroomBytes"/>.</summary>
    private static bool TryAutoPromote(State s, Tensor cpuTensor, nuint byteSize, out ulong dptr)
    {
        dptr = 0;
        if (!AutoPromoteWeights || byteSize < AutoPromoteMinBytes)
        {
            return false;
        }
        UploadState state = s.UploadTracker.GetOrCreateValue(cpuTensor);
        state.Count++;
        if (state.Blocked || state.Count < 2)
        {
            return false;
        }
        if (CudaDriverApi.cuMemGetInfo(out nuint free, out _) != 0 || (long)free - (long)byteSize < _autoPromoteHeadroomBytes)
        {
            return false;
        }
        // Read DataPointer BEFORE planting the demote callbacks — it triggers EnsureCpuData.
        dptr = CudaMemory.AllocatePersistent(byteSize);
        CudaMemory.CopyHostToDevice(dptr, cpuTensor.DataPointer, byteSize);
        RegisterCachedWeight(cpuTensor, dptr, byteSize);
        state.Promoted = true;
        // Capture the owning State: demotion must free against this backend's context/stream even if
        // another backend registers later (same rule as the activation callbacks).
        cpuTensor._gpuSyncCallback = () => OnPromotedHostAccess(s, cpuTensor);
        cpuTensor._gpuDisposeCallback = () => OnPromotedHostAccess(s, cpuTensor);
        cpuTensor._gpuCleanupContext = s.Context?.Handle ?? 0;
        return true;
    }

    /// <summary>Demote hook for auto-promoted weights: fires from <c>EnsureCpuData</c> (host about to read/write) or
    /// Dispose/finalizer (via the pending-cleanup queue). Frees the device copy (and any cached dtype-cast) after a
    /// stream sync so in-flight kernels finish first. Host data is authoritative for promoted tensors, so no D2H copy
    /// is needed. Blocks re-promotion only when an entry was actually evicted — after <see cref="FreeAllCached"/>
    /// (backend teardown) the stale callback finds nothing and the tensor stays promotable for the next session.</summary>
    private static void OnPromotedHostAccess(State s, Tensor tensor)
    {
        // Torn-down backend: the caches were already freed wholesale and the state's ConditionalWeakTable may
        // have been finalized while the state was unreachable (resurrected via the finalizer-cleanup queue) —
        // touching it would NRE. The bool read is always safe.
        if (s.Unregistered)
        {
            return;
        }
        if (!s.UploadTracker.TryGetValue(tensor, out UploadState? state) || !state.Promoted)
        {
            return;
        }
        state.Promoted = false;
        s.Context?.EnsureCurrent();
        if (s.WeightCache.Remove(tensor, out ulong dptr))
        {
            state.Blocked = true;
            s.CachedPointers.Remove(dptr);
            s.CachedBytes -= (long)ByteSize(tensor);
            if (s.StreamHandle != 0)
            {
                CudaDriverApi.cuStreamSynchronize(s.StreamHandle).ThrowOnError();
            }
            CudaMemory.Free(dptr);
            if (s.WeightCastCache.Remove(tensor, out (ulong castPtr, nuint bytes) cast))
            {
                s.CachedPointers.Remove(cast.castPtr);
                CudaMemory.Free(cast.castPtr);
                s.CachedBytes -= (long)cast.bytes;
            }
        }
    }

    /// <summary>Detaches the auto-promotion lifecycle from a tensor whose cached device copy is being freed by a
    /// bulk eviction path (<see cref="FreeWeights"/> / <see cref="FreeAllCached"/> /
    /// <see cref="TryUnregisterCachedWeight"/>): resets the promoted flag and removes the planted sync/dispose
    /// callbacks. Without this, a later Dispose — or worse, a finalizer — of the tensor enqueues a stale
    /// <see cref="OnPromotedHostAccess"/> against a state that may since have been torn down; the CUDA driver
    /// reuses primary-context handles, so the NEXT backend on the device drains and runs those stale callbacks
    /// (the GGUF model-switch NRE). Re-promotion stays possible: the tensor's upload count is intact, so the
    /// next session's second upload re-promotes it (matching the documented FreeAllCached semantics).</summary>
    private static void DetachPromotedTensor(State s, Tensor tensor)
    {
        if (s.UploadTracker.TryGetValue(tensor, out UploadState? promo) && promo.Promoted)
        {
            promo.Promoted = false;
            tensor._gpuSyncCallback = null;
            tensor._gpuDisposeCallback = null;
            tensor._gpuCleanupContext = 0;
        }
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

    internal static bool IsActivationCached(Tensor tensor) => Resolve().ActivationCache.ContainsKey(tensor);

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
        if (s.WeightCastCache.Remove(weight, out (ulong castPtr, nuint bytes) cast))
        {
            s.CachedPointers.Remove(cast.castPtr);
            // Stream-ordered free: the cast was allocated via the async pool and may be referenced by
            // GEMMs still in flight on the compute stream; FreeAsync orders the release after them.
            CudaMemory.FreeAsync(cast.castPtr, s.StreamHandle);
            s.CachedBytes -= (long)cast.bytes;
        }
        if (s.WeightCache.Remove(weight, out dptr))
        {
            s.CachedPointers.Remove(dptr);
            s.CachedBytes -= (long)ByteSize(weight);
            DetachPromotedTensor(s, weight);
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
                DetachPromotedTensor(s, weight);
            }
            if (s.WeightCastCache.Remove(weight, out (ulong castPtr, nuint bytes) cast))
            {
                s.CachedPointers.Remove(cast.castPtr);
                CudaMemory.Free(cast.castPtr);
                s.CachedBytes -= (long)cast.bytes;
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

        // Detach promotion callbacks BEFORE clearing: after this wholesale free, a promoted tensor's
        // Dispose/finalizer must not queue a cleanup against this state (see DetachPromotedTensor).
        foreach (Tensor weight in s.WeightCache.Keys)
        {
            DetachPromotedTensor(s, weight);
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
    public static void FreeActivations(bool trimPool = true)
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
        // is logically free. Sync first so the queued async frees complete. Hot per-step/per-tile callers pass
        // trimPool=false: the next iteration re-uses the reservation directly, and a trim there costs a multi-GB
        // driver release + re-map every iteration (persistent cuMemAlloc callers reclaim the pool via their
        // OOM-retry if they ever need it).
        if (trimPool) TrimPool();
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

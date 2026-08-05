using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>GPU memory transfer helper with weight and activation caching. Weights preload via PreloadWeight() and stay
/// until FreeAllCached(); activations set by CacheActivation() after each op and are consumed by the next op's
/// CopyToDevice(). Lazy sync: CPU access to DataPointer triggers a GPU→CPU sync on demand.
/// <para><b>Multi-backend safety:</b> all mutable state lives in a per-BACKEND <see cref="State"/> object (one per
/// <see cref="CudaBackend"/> registration, identified by a process-unique <see cref="State.Key"/> that is never
/// reused), resolved at every call via the ambient thread-static the owning backend binds at each op entry
/// (<c>CudaBackend.EnterOp</c>). Activation callbacks capture their owning State directly, so a tensor produced on
/// backend A always syncs/frees against A's context and stream even when backend B was used more recently.
/// History: this was once a single set of static fields (constructing a second CudaBackend silently retargeted ALL
/// cached state → cross-device frees → CUDA_ERROR_ILLEGAL_ADDRESS on both GPUs), then keyed by CUDA context handle
/// — which collapsed two backends on the SAME device into one State (primary contexts are one-per-device), giving
/// last-writer-wins stream bindings, cross-backend H2D copies on the wrong stream, and Dispose wiping the sibling's
/// live weights. Per-backend keys + the ambient make same-device backends fully independent; the cost is that a
/// host tensor shared by two same-device backends holds two device copies (accepted: preserves the "each backend
/// owns its VRAM" invariant and PreloadWeight's ownership contract, no refcounting).</para></summary>
internal static unsafe class GpuTransferHelper
{
    /// <summary>All per-backend mutable state, identified by a process-unique <see cref="Key"/>.</summary>
    internal sealed class State
    {
        /// <summary>Process-unique identity of this backend registration, never reused. Keys the registry, each
        /// tensor's GPU bindings, and the finalizer-cleanup buckets.</summary>
        public nint Key;

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

        /// <summary>Buffers displaced by a <see cref="CacheActivation"/> rebind, awaiting a caller's
        /// <see cref="FreeDevice"/>; whatever is still here when the next op starts had no owner and is freed.</summary>
        public readonly HashSet<ulong> PendingOrphans = new();

        /// <summary>Graph-capture arenas: while a decode-step graph is being captured
        /// (<see cref="BeginGraphArena"/>), <see cref="AllocateDevice"/> bump-allocates from a per-capture
        /// pre-reserved buffer instead of the stream-ordered pool — so the captured graph contains ZERO
        /// memAlloc/memFree nodes for step intermediates (measured on gemma3: 875 alloc + 797 free nodes of
        /// 2264 total, each replaying every token). One arena per LIVE graph (the batch scheduler can hold
        /// several captured graphs at once — sharing one bump buffer would alias them); pointers inside any
        /// live arena are never freed individually (every free path checks <see cref="IsArenaPtr"/>); an
        /// arena is released as a whole when its graph is disposed (<see cref="FreeGraphArena"/>). Overflow
        /// falls back to normal pool allocation (correct, just adds nodes) and logs once.</summary>
        public readonly List<(ulong basePtr, nuint capacity)> LiveArenas = new();
        public ulong ArenaBase;      // the arena of the capture in progress (also in LiveArenas)
        public nuint ArenaCapacity;
        public nuint ArenaOffset;
        public bool ArenaActive;
        public bool ArenaOverflowLogged;

        /// <summary>Q8_1 activation sidecars emitted by quantize-at-producer kernels (xq int8 + per-32-block
        /// scale xd + int-sum xs device buffers, K = the producing row width). Keyed by the F32 output tensor;
        /// consumed by the dp4a Linear path in place of its own quantize launch. Invalidated (buffers freed)
        /// whenever the tensor is re-bound, synced to host, or disposed — see <see cref="CacheActivation"/>.</summary>
        public readonly Dictionary<Tensor, (ulong xq, ulong xd, ulong xs, int k)> SidecarCache = new(ReferenceEqualityComparer.Instance);

        /// <summary>Activations pinned to SURVIVE <see cref="FreeActivations"/> — cross-step state whose only
        /// authoritative copy lives on-device (e.g. the across-step feature cache's previous indicator and
        /// residual, which per-step FreeActivations in the video pipelines would otherwise silently destroy:
        /// the host buffer was never materialized, so the next CopyToDevice would re-upload garbage). Pinned
        /// tensors are still freed by their own Dispose/sync callbacks and by <see cref="FreeAllCached"/>.</summary>
        public readonly HashSet<Tensor> PinnedActivations = new(ReferenceEqualityComparer.Instance);

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

        /// <summary>Step-graph capture-window alloc/free tracker (diagnostic — see <see cref="CudaBackend.StepGraphBegin"/>).
        /// Per-State: this used to be process-wide statics on <see cref="CudaMemory"/>, so a second backend
        /// beginning its own capture cleared the first backend's in-flight window and folded its own
        /// non-capturing allocations into the wrong backend's report. Diagnostic-only (guarded by
        /// <see cref="CaptureAllocs"/>'s lock, never touches memory), but the leak-detection numbers were
        /// silently wrong under two live backends.</summary>
        public bool TrackCaptureWindow;
        public readonly Dictionary<ulong, nuint> CaptureAllocs = new();
        public long CaptureAllocBytes, CaptureFreeBytes;
        public int CaptureAllocCount, CaptureFreeCount;
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

    /// <summary>Registered states keyed by <see cref="State.Key"/>. Concurrent: registration happens on backend
    /// construction threads while Resolve() reads from compute threads.</summary>
    private static readonly ConcurrentDictionary<nint, State> _states = new();

    /// <summary>Last-registered live state per CUDA context handle — the ambient-less resolution fallback for
    /// helper calls arriving outside a backend op (tests, tooling). Ambiguous by construction when two backends
    /// share a device; <see cref="Resolve"/> warns once per handle in that case.</summary>
    private static readonly ConcurrentDictionary<nint, State> _byContext = new();

    /// <summary>The calling thread's current backend State, bound by <c>CudaBackend.EnterOp</c> at every op entry.
    /// This is THE resolution path — cuCtxGetCurrent cannot distinguish two same-device backends (they share the
    /// primary context), so context identity stopped being usable as the owner identity.</summary>
    [ThreadStatic]
    private static State? _ambient;

    /// <summary>Fast path for ambient-less callers: the sole registered state when exactly one backend exists.
    /// Null when zero or 2+ states are registered.</summary>
    private static volatile State? _sole;

    /// <summary>Monotonic source for <see cref="State.Key"/>. Starts at 1; key 0 is the context-less bucket.</summary>
    private static long _nextKey;

    /// <summary>Serializes Register/Unregister so <see cref="_sole"/> and the indexes stay consistent.</summary>
    private static readonly object _registryLock = new();

    /// <summary>Context handles already warned about ambiguous ambient-less resolution.</summary>
    private static readonly ConcurrentDictionary<nint, bool> _ambiguityWarned = new();

    /// <summary>Debug tripwire (HARTSY_ASSERT_AMBIENT=1): throw instead of falling back when an ambient-less call
    /// happens while multiple backends are live — catches op entry points the EnterOp transform missed.</summary>
    private static readonly bool _assertAmbient = Environment.GetEnvironmentVariable("HARTSY_ASSERT_AMBIENT") == "1";

    /// <summary>Fallback state for calls made before any backend registers (unit tests exercising pure
    /// helpers). Its stream is 0 and context null, so every code path degrades to the safe no-op branch.</summary>
    private static readonly State _unregistered = new();

    /// <summary>Creates and registers a NEW state for one backend. Called once from <see cref="CudaBackend"/>'s
    /// constructor with the context, compute stream, and streaming cache; the returned state is the backend's
    /// identity for ambient binding, tensor GPU bindings, and finalizer-cleanup routing.</summary>
    public static State Register(CudaContext context, nint stream, IStreamingWeightCache? streamingCache)
    {
        State state = new State
        {
            Key = (nint)Interlocked.Increment(ref _nextKey),
            Context = context,
            StreamHandle = stream,
            StreamingCache = streamingCache,
        };
        lock (_registryLock)
        {
            _states[state.Key] = state;
            _byContext[context.Handle] = state;
            _sole = _states.Count == 1 ? state : null;
        }
        return state;
    }

    /// <summary>Removes a backend's state at disposal (after <see cref="FreeAllCached"/>). Keys are never reused,
    /// so the caller may always discard its key's pending-cleanup queue afterwards.</summary>
    public static void Unregister(State state)
    {
        // Mark dead BEFORE removal: any promoted-weight callback that still fires (finalizer-queued before
        // its tensor could be detached) must see the flag and no-op instead of touching this state's caches.
        state.Unregistered = true;
        lock (_registryLock)
        {
            _states.TryRemove(state.Key, out _);
            // The context-handle index only drops when it still points at us — a same-device sibling that
            // registered later keeps its binding.
            if (_byContext.TryGetValue(state.Context?.Handle ?? 0, out State? current) && ReferenceEquals(current, state))
            {
                _byContext.TryRemove(state.Context!.Handle, out _);
            }
            _sole = _states.Count == 1 ? Enumerable.First(_states.Values) : null;
        }
        if (ReferenceEquals(_ambient, state))
        {
            _ambient = null;
        }
    }

    /// <summary>Binds <paramref name="state"/> as the calling thread's current backend. Called by
    /// <c>CudaBackend.EnterOp</c> on every op entry; cheap (one thread-static write).</summary>
    internal static void SetAmbient(State state) => _ambient = state;

    /// <summary>Resolves the calling thread's owning backend State: the ambient bound by the current op, else the
    /// sole registered state, else the last state registered for the thread's current CUDA context (ambiguous for
    /// same-device siblings — warned once), else an inert empty state (pre-registration test paths).</summary>
    private static State Resolve()
    {
        State? ambient = _ambient;
        if (ambient is not null && !ambient.Unregistered) return ambient;
        State? sole = _sole;
        if (sole is not null) return sole;
        if (_states.IsEmpty) return _unregistered;
        if (_assertAmbient)
        {
            throw new InvalidOperationException(
                "GpuTransferHelper.Resolve: no ambient State with multiple backends registered — an op entry point "
                + "missed the EnterOp transform (HARTSY_ASSERT_AMBIENT=1).");
        }
        if (CudaDriverApi.cuCtxGetCurrent(out nint current) == 0 && _byContext.TryGetValue(current, out State? state))
        {
            if (_ambiguityWarned.TryAdd(current, true) && SameContextStateCount(current) > 1)
            {
                Logs.Warning("[Cuda] Ambient-less state resolution with two backends on one device — using the "
                    + "most recently registered. Op entry points should bind the ambient (EnterOp).");
            }
            return state;
        }
        return _unregistered;
    }

    /// <summary>Live states bound to <paramref name="contextHandle"/> (2+ = same-device siblings).</summary>
    private static int SameContextStateCount(nint contextHandle)
    {
        int count = 0;
        foreach (State s in _states.Values)
        {
            if ((s.Context?.Handle ?? 0) == contextHandle) count++;
        }
        return count;
    }

    /// <summary>Every live registered state sharing <paramref name="deviceOrdinal"/> — the same-device escalation
    /// sweep for OOM retries (a sibling backend's pool reservations can hold the memory this one needs).</summary>
    internal static List<State> StatesOnDevice(int deviceOrdinal)
    {
        List<State> result = [];
        foreach (State s in _states.Values)
        {
            if (!s.Unregistered && s.Context?.DeviceOrdinal == deviceOrdinal) result.Add(s);
        }
        return result;
    }

    /// <summary>The ambient (or resolved) state's compute stream, for <see cref="CudaMemory"/>'s allocator routing;
    /// 0 when nothing is resolvable (sync-allocation fallback).</summary>
    internal static nint ResolvedStreamHandle => Resolve().StreamHandle;

    /// <summary>The calling thread's owning backend State, for callers (<see cref="CudaMemory"/>'s capture-window
    /// tracker) that need more than just the stream handle. Same ambient-first resolution as every other lookup.</summary>
    internal static State CurrentState => Resolve();

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
        // Same-device escalation: a SIBLING backend's stream-ordered pool reservations can hold the very bytes
        // this OOM needs (each backend frees onto its own stream). Draining a sibling is only stream syncs +
        // pool trims — no mutation of its caches — and the per-device generation gate means it is quiescent
        // while we are mid-allocation, so this cannot stall its in-flight work.
        if (s.Context is not null)
        {
            foreach (State sibling in StatesOnDevice(s.Context.DeviceOrdinal))
            {
                if (ReferenceEquals(sibling, s))
                {
                    continue;
                }
                if (sibling.StreamHandle != 0)
                {
                    CudaDriverApi.cuStreamSynchronize(sibling.StreamHandle);
                }
                sibling.StreamingCache?.DrainAndReleasePool();
            }
            // The sibling cache's entry guard re-bound the thread ambient to the sibling — restore OURS, or the
            // very allocation retry this drain serves would land on the sibling's stream.
            SetAmbient(s);
        }
        // Always trim the default pool too: with no streaming cache wired (auto-transfer paths, tests), every
        // cuMemFreeAsync'd transient stays RESERVED in the stream-ordered pool. An OOM retry that never trims
        // reports "GPU full" (cuMemGetInfo counts reservations as used) even though most of it is reusable —
        // the fp8 auto-transfer Flux/T5 OOM at 0.9% free.
        TrimPool();
    }

    // Diagnostic: HARTSY_H2D_TRACE=1 logs the first big cache misses with shape/dtype so a re-uploaded weight set
    // is distinguishable from ordinary activation traffic.
    private static readonly bool _traceBigMisses = Environment.GetEnvironmentVariable("HARTSY_H2D_TRACE") == "1";
    private static int _bigMissTraceCount;

    /// <summary>How many big misses to log; raise it to see past the text-encode phase into denoise.</summary>
    private static readonly int _traceBigMissLimit =
        int.TryParse(Environment.GetEnvironmentVariable("HARTSY_H2D_TRACE_LIMIT"), out int lim) ? lim : 24;

    /// <summary>Returns the GPU device pointer for a tensor, using caches to avoid transfers. Priority: weight cache
    /// → activation cache → fresh H2D transfer.</summary>
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
        // HARTSY_PROFILE visibility into miss H2D volume.
        using Profiling.NvtxRange _miss = Profiling.NvtxRange.Push(byteSize > (1u << 20) ? "H2D_MISS_BIG" : "H2D_MISS_SMALL");
        if (byteSize > (1u << 20) && _traceBigMisses && Interlocked.Increment(ref _bigMissTraceCount) <= _traceBigMissLimit)
        {
            string shape = string.Join("x", Enumerable.Range(0, cpuTensor.Shape.Rank).Select(i => cpuTensor.Shape[i]));
            Logs.Info($"[h2d-trace] MISS [{shape}] {cpuTensor.DType} {byteSize >> 20} MB "
                + $"cachedWeights={s.WeightCache.Count} hits={s.Hits} misses={s.Misses}");
        }
        // A miss during graph capture bakes a per-replay H2D memcpy node into the graph — always worth
        // knowing about (HARTSY_GRAPH_DUMP=1 logs the offender so it can be made resident pre-capture).
        if (s.ArenaActive && Environment.GetEnvironmentVariable("HARTSY_GRAPH_DUMP") == "1")
        {
            string shape = string.Join(",", Enumerable.Range(0, cpuTensor.Shape.Rank).Select(i => cpuTensor.Shape[i]));
            Logs.Info($"[Cuda] H2D MISS inside graph capture: shape=[{shape}] dtype={cpuTensor.DType} bytes={byteSize}");
        }
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
        State s = Resolve();
        if (s.ArenaActive)
        {
            nuint aligned = (byteSize + 255) & ~(nuint)255;
            if (s.ArenaOffset + aligned <= s.ArenaCapacity)
            {
                ulong p = s.ArenaBase + s.ArenaOffset;
                s.ArenaOffset += aligned;
                return p;
            }
            if (!s.ArenaOverflowLogged)
            {
                s.ArenaOverflowLogged = true;
                Logs.Warning($"[Cuda] graph-capture arena exhausted ({(long)s.ArenaCapacity >> 20} MB) — " +
                    "remaining capture allocations fall back to pool nodes (set HARTSY_GRAPH_ARENA_MB higher).");
            }
        }
        return CudaMemory.Allocate(byteSize);
    }

    /// <summary>True when the pointer lies inside ANY live graph-capture arena (never individually freed).</summary>
    internal static bool IsArenaPtr(State s, ulong p)
    {
        for (int i = 0; i < s.LiveArenas.Count; i++)
            if (p >= s.LiveArenas[i].basePtr && p < s.LiveArenas[i].basePtr + s.LiveArenas[i].capacity)
                return true;
        return false;
    }

    /// <summary>Allocates a fresh per-capture arena and activates it. Call immediately before a decode-step
    /// graph capture; pair with <see cref="EndGraphArena"/>, and release via <see cref="FreeGraphArena"/>
    /// when the captured graph is disposed. Returns the arena base (0 = allocation failed, arena disabled
    /// for this capture).</summary>
    public static ulong BeginGraphArena()
    {
        State s = Resolve();
        long mb = long.TryParse(Environment.GetEnvironmentVariable("HARTSY_GRAPH_ARENA_MB"), out long v) ? Math.Clamp(v, 8, 2048) : 32;
        nuint cap = (nuint)(mb << 20);
        ulong basePtr;
        try { basePtr = CudaMemory.Allocate(cap); }
        // VRAM-tight (e.g. gemma2 non-low-vram edge): run without arena. OutOfVramException is the typed
        // exhaustion CudaMemory now raises; CudaException still covers the non-capacity driver failures.
        catch (OutOfVramException) { return 0; }
        catch (CudaException) { return 0; }
        s.LiveArenas.Add((basePtr, cap));
        s.ArenaBase = basePtr;
        s.ArenaCapacity = cap;
        s.ArenaOffset = 0;
        s.ArenaOverflowLogged = false;
        s.ArenaActive = true;
        return basePtr;
    }

    /// <summary>Deactivates the in-progress capture arena (buffers handed out stay valid for the graph's
    /// lifetime) and logs the actual bytes used, for capacity tuning.</summary>
    public static void EndGraphArena()
    {
        State s = Resolve();
        if (!s.ArenaActive) return;
        s.ArenaActive = false;
        Logs.Debug($"[Cuda] graph-capture arena used {(long)s.ArenaOffset >> 10} KB of {(long)s.ArenaCapacity >> 20} MB.");
    }

    /// <summary>Releases a per-capture arena when its graph is disposed (stream-ordered free).</summary>
    public static void FreeGraphArena(ulong basePtr)
    {
        if (basePtr == 0) return;
        State s = Resolve();
        for (int i = 0; i < s.LiveArenas.Count; i++)
        {
            if (s.LiveArenas[i].basePtr == basePtr)
            {
                s.LiveArenas.RemoveAt(i);
                // SYNCHRONOUS free on a drained stream: arena release is a cold path (end of a
                // generation), and an async free here races the pool — the freed block can be handed to
                // another backend's allocations while replay work is still in flight on the original
                // stream (intermittent CudaGraph_RepeatedReplay flake when suites interleave backends).
                if (s.StreamHandle != 0) CudaDriverApi.cuStreamSynchronize(s.StreamHandle);
                CudaMemory.Free(basePtr);
                return;
            }
        }
    }

    /// <summary>Frees a GPU buffer asynchronously on the compute stream. Skips cached pointers (weight + activation) and arena pointers.</summary>
    public static void FreeDevice(ulong gpuPtr)
    {
        State s = Resolve();
        if (gpuPtr != 0 && !s.CachedPointers.Contains(gpuPtr) && !IsArenaPtr(s, gpuPtr))
        {
            // The caller owns this one after all, so SweepOrphans must not free it a second time.
            if (s.PendingOrphans.Count != 0) s.PendingOrphans.Remove(gpuPtr);
            CudaMemory.FreeAsync(gpuPtr, s.StreamHandle);
        }
    }

    /// <summary>Frees buffers displaced by a <see cref="CacheActivation"/> rebind that no caller claimed.</summary>
    /// <remarks>Called at the start of each op, so every previous op's <c>finally</c> has already run and anything
    /// still parked provably has no owner. Sweeping here rather than inside CacheActivation is what keeps the
    /// in-place case (where the displaced buffer is the op's own input) from being double-freed.</remarks>
    /// <summary><c>HARTSY_ORPHAN_SWEEP=0</c> restores the pre-fix behaviour (displaced buffers leak) — a
    /// bisect handle for a change that sits on every op's allocation path.</summary>
    private static readonly bool OrphanSweepEnabled =
        Environment.GetEnvironmentVariable("HARTSY_ORPHAN_SWEEP") != "0";

    internal static void SweepOrphans()
    {
        if (!OrphanSweepEnabled) return;
        State s = Resolve();
        if (s.PendingOrphans.Count == 0) return;
        // cuMemFreeAsync on a buffer allocated BEFORE the capture began is rejected outright
        // (CUDA_ERROR_INVALID_VALUE, which aborted all three CudaGraphTests). Leave them parked; the first
        // op after capture ends sweeps them. The probe sits behind the empty-set check on purpose — a driver
        // call on every EnterOp would cost more than the leak did.
        if (s.StreamHandle != 0)
        {
            CudaDriverApi.cuStreamIsCapturing(s.StreamHandle, out int captureStatus).ThrowOnError();
            if (captureStatus != 0) return;
        }
        foreach (ulong ptr in s.PendingOrphans)
        {
            if (!s.CachedPointers.Contains(ptr) && !IsArenaPtr(s, ptr))
            {
                CudaMemory.FreeAsync(ptr, s.StreamHandle);
            }
        }
        s.PendingOrphans.Clear();
    }

    /// <summary>Registers a Q8_1 sidecar (from a quantize-at-producer kernel) for an activation tensor.
    /// Call AFTER <see cref="CacheActivation"/> for the same tensor — CacheActivation invalidates any
    /// previous sidecar as part of rebinding.</summary>
    internal static void RegisterSidecar(Tensor tensor, ulong xq, ulong xd, ulong xs, int k)
    {
        State s = Resolve();
        RemoveSidecar(s, tensor);
        s.SidecarCache[tensor] = (xq, xd, xs, k);
    }

    /// <summary>Looks up a Q8_1 sidecar for a dp4a GEMV input (M=1 decode rows only — producers emit
    /// per-row sidecars and the decode path is single-row).</summary>
    internal static bool TryGetSidecar(Tensor tensor, int k, out ulong xq, out ulong xd, out ulong xs)
    {
        if (Resolve().SidecarCache.TryGetValue(tensor, out (ulong xq, ulong xd, ulong xs, int k) sc) && sc.k == k)
        {
            xq = sc.xq; xd = sc.xd; xs = sc.xs;
            return true;
        }
        xq = xd = xs = 0;
        return false;
    }

    private static void RemoveSidecar(State s, Tensor tensor)
    {
        if (s.SidecarCache.Remove(tensor, out (ulong xq, ulong xd, ulong xs, int k) sc))
        {
            if (!IsArenaPtr(s, sc.xq)) CudaMemory.FreeAsync(sc.xq, s.StreamHandle);
            if (!IsArenaPtr(s, sc.xd)) CudaMemory.FreeAsync(sc.xd, s.StreamHandle);
            if (!IsArenaPtr(s, sc.xs)) CudaMemory.FreeAsync(sc.xs, s.StreamHandle);
        }
    }

    /// <summary>Caches an op's output GPU pointer on the tensor, avoiding D2H transfer. Sets lazy callbacks: DataPointer
    /// access triggers D2H, Dispose frees GPU memory. The callbacks capture this backend's <see cref="State"/>, so
    /// they stay correct even after another backend registers.</summary>
    public static void CacheActivation(Tensor tensor, ulong gpuPtr, nuint byteSize)
    {
        State s = Resolve();

        // Any rebind of this tensor's device buffer stales a producer-emitted Q8_1 sidecar — drop it.
        RemoveSidecar(s, tensor);

        // This tensor already maps to a DIFFERENT device buffer, so that buffer is being displaced. Drop it from
        // the cached set — but who frees it depends on the op:
        //   in-place (backend.Gelu(x, x) into a fresh output): the displaced buffer IS the op's pInput, and its
        //     `finally FreeDevice(pInput)` frees it exactly once (FreeDevice only skips CachedPointers members);
        //   not in-place (Linear(output, input, weight) reusing `output`): the displaced buffer is nobody's input,
        //     so nothing ever frees it. Removing it from the cached set and stopping there orphaned it — no tensor
        //     maps to it, so neither Dispose nor FreeActivations nor GC reclaimed it. Measured: 12 Linear calls at
        //     a 563 MB output stranded 5942 MB, and it broke unrelated tests sharing the GPU.
        // So park it instead: FreeDevice claims it if the caller does own it, and SweepOrphans frees whatever is
        // still unclaimed when the next op begins (by which point the owning op's finally blocks have all run).
        if (gpuPtr != 0 && s.ActivationCache.TryGetValue(tensor, out (ulong gpuPtr, nuint bytes) prev) && prev.gpuPtr != gpuPtr)
        {
            s.CachedPointers.Remove(prev.gpuPtr);
            if (!IsArenaPtr(s, prev.gpuPtr)) s.PendingOrphans.Add(prev.gpuPtr);
        }

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

        // Arena ownership is decided AT BINDING TIME and captured: the callbacks below may fire after the
        // owning graph arena has been destroyed (e.g. GraphStream teardown disposes the graph, THEN its
        // fixed buffers) — at that point IsArenaPtr's live-arena check returns false and the callback
        // would double-free memory the arena already returned (the CUDA_ERROR_INVALID_VALUE "dispose
        // failed during cleanup" warnings in HeartMuLa's session teardown, 2026-07-25). An arena-born
        // pointer is NEVER individually freed, live arena or not.
        bool arenaBacked = IsArenaPtr(s, gpuPtr);

        // Lazy sync: when CPU code accesses DataPointer, wait for stream, copy GPU→CPU, then free.
        // Stream sync is needed because per-op Sync() has been removed — the producing kernel may still be in flight.
        // EnsureCurrent in both callbacks: they fire from whatever thread later reads/disposes
        // the tensor (potentially the GC finalizer thread), which won't have bound the context.
        Action syncCallback = () =>
        {
            if (s.ActivationCache.Remove(tensor, out (ulong gpuPtr, nuint bytes) cached))
            {
                s.PinnedActivations.Remove(tensor);
                s.D2hSyncs++;
                s.Context?.EnsureCurrent();
                RemoveSidecar(s, tensor);
                if (arenaBacked && !IsArenaPtr(s, cached.gpuPtr))
                {
                    // The owning arena is gone — the device data went with it; reading it would touch freed
                    // memory. Surface loudly instead of copying garbage.
                    Logs.Warning("[Cuda] Activation read after its graph arena was freed — returning zeros (read the tensor before DisposeGraph).");
                    tensor.EnsureHostBuffer();
                    s.CachedPointers.Remove(cached.gpuPtr);
                    return;
                }
                CudaDriverApi.cuStreamSynchronize(s.StreamHandle).ThrowOnError();
                // Allocate the host destination only now, on the first real CPU read of this activation.
                void* cpuPtr = tensor.EnsureHostBuffer();
                CudaMemory.CopyDeviceToHost(cpuPtr, cached.gpuPtr, cached.bytes);
                s.CachedPointers.Remove(cached.gpuPtr);
                if (!arenaBacked && !IsArenaPtr(s, cached.gpuPtr)) CudaMemory.FreeAsync(cached.gpuPtr, s.StreamHandle);
            }
        };

        // On dispose without sync: free GPU memory asynchronously (skip D2H — data not needed)
        Action disposeCallback = () =>
        {
            if (s.ActivationCache.Remove(tensor, out (ulong gpuPtr, nuint bytes) cached))
            {
                s.PinnedActivations.Remove(tensor);
                s.Context?.EnsureCurrent();
                RemoveSidecar(s, tensor);
                s.CachedPointers.Remove(cached.gpuPtr);
                if (!arenaBacked && !IsArenaPtr(s, cached.gpuPtr)) CudaMemory.FreeAsync(cached.gpuPtr, s.StreamHandle);
            }
        };
        // Keyed binding routes this tensor's finalizer cleanup into THIS backend's context bucket so a concurrent
        // backend's drain thread never runs it against the wrong (and unsynchronized) State.
        tensor.SetGpuBinding(s.Key, syncCallback, disposeCallback);
    }

    /// <summary>Frees EVERY cached weight cast (they are pure caches — always rebuildable from the source
    /// weight) and returns the bytes released. Called from CudaMemory's OOM retry so opportunistic cast
    /// caching can never make an allocation fail that would have succeeded without it.</summary>
    internal static long EvictAllWeightCasts()
    {
        State s = Resolve();
        long released = 0;
        foreach (KeyValuePair<Tensor, (ulong castPtr, nuint bytes)> kv in s.WeightCastCache)
        {
            s.CachedPointers.Remove(kv.Value.castPtr);
            CudaMemory.FreeAsync(kv.Value.castPtr, s.StreamHandle);
            s.CachedBytes -= (long)kv.Value.bytes;
            released += (long)kv.Value.bytes;
        }
        s.WeightCastCache.Clear();
        return released;
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

    /// <summary>Uploads a weight tensor to GPU and caches it for future CopyToDevice calls. Returns false when the
    /// weight was already resident and no upload happened.</summary>
    /// <remarks>The return value is what makes a failed bulk preload rollback-able: only weights this call actually
    /// uploaded may be freed, since one already resident from an earlier phase (or from HARTSY_KEEP_MODELS) belongs to
    /// someone else. Reporting it from here rather than having the caller pre-check with <see cref="IsWeightCached"/>
    /// keeps the answer tied to the same <see cref="Resolve"/> result that performs the registration — a separate
    /// lookup could in principle resolve a different backend's <see cref="State"/> and mis-attribute ownership.</remarks>
    public static bool PreloadWeight(Tensor weight)
    {
        State s = Resolve();
        if (s.WeightCache.ContainsKey(weight))
            return false;

        nuint byteSize = ByteSize(weight);
        // Resident weights are freed with the synchronous Free (FreeWeights/FreeAllCached), so they must be
        // allocated synchronously too — keep them out of the stream-ordered transient pool.
        ulong dptr = CudaMemory.AllocatePersistent(byteSize);
        CudaMemory.CopyHostToDevice(dptr, weight.DataPointer, byteSize);

        RegisterCachedWeight(weight, dptr, byteSize);
        return true;
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
        // another backend registers later (same rule as the activation callbacks). The keyed binding is what
        // lets a SECOND backend promote this same host tensor without overwriting our demote hook.
        Action demote = () => OnPromotedHostAccess(s, cpuTensor);
        cpuTensor.SetGpuBinding(s.Key, demote, demote);
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
            // Keyed clear: only THIS backend's binding goes — a sibling backend's promotion of the same host
            // tensor keeps its own demote hook.
            tensor.ClearGpuBinding(s.Key);
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
            if (!IsArenaPtr(s, dptr)) CudaMemory.Free(dptr);
        }
        foreach ((ulong basePtr, nuint _) in s.LiveArenas) CudaMemory.Free(basePtr);
        s.LiveArenas.Clear();
        s.ArenaBase = 0; s.ArenaCapacity = 0; s.ArenaOffset = 0; s.ArenaActive = false;
        foreach (Tensor t in s.SidecarCache.Keys.ToList())
            RemoveSidecar(s, t);
        s.WeightCache.Clear();
        s.ActivationCache.Clear();
        s.WeightCastCache.Clear();
        s.CachedPointers.Clear();
        s.PinnedActivations.Clear();
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
        // Called between pipeline stages, never mid-op, so any parked orphan is already ownerless.
        SweepOrphans();
        // Q8_1 sidecars are per-step transients riding on activations — sweep them all here (pinned
        // survivors included: a consumer that misses the sidecar simply re-quantizes).
        foreach (Tensor t in s.SidecarCache.Keys.ToList())
            RemoveSidecar(s, t);
        List<KeyValuePair<Tensor, (ulong gpuPtr, nuint bytes)>>? survivors = null;
        foreach (KeyValuePair<Tensor, (ulong gpuPtr, nuint bytes)> kv in s.ActivationCache)
        {
            if (s.PinnedActivations.Contains(kv.Key))
            {
                (survivors ??= new List<KeyValuePair<Tensor, (ulong gpuPtr, nuint bytes)>>()).Add(kv);
                continue;
            }
            s.CachedPointers.Remove(kv.Value.gpuPtr);
            if (!IsArenaPtr(s, kv.Value.gpuPtr)) CudaMemory.FreeAsync(kv.Value.gpuPtr, s.StreamHandle);
        }
        s.ActivationCache.Clear();
        if (survivors is not null)
            foreach (KeyValuePair<Tensor, (ulong gpuPtr, nuint bytes)> kv in survivors)
                s.ActivationCache[kv.Key] = kv.Value;

        // Return pooled memory to the driver. cuMemFreeAsync (used by every activation/dispose free) hands memory
        // back to the stream-ordered mempool, which RESERVES it (cuMemGetInfo counts it as used) until trimmed —
        // otherwise the pool's high-water mark grows every op and multi-step diffusion OOMs even though the memory
        // is logically free. Sync first so the queued async frees complete. Hot per-step/per-tile callers pass
        // trimPool=false: the next iteration re-uses the reservation directly, and a trim there costs a multi-GB
        // driver release + re-map every iteration (persistent cuMemAlloc callers reclaim the pool via their
        // OOM-retry if they ever need it).
        if (trimPool) TrimPool();
    }

    /// <summary>Marks a tensor's device activation as surviving <see cref="FreeActivations"/> (cross-step state
    /// whose only copy is on-device). Keyed by tensor object identity — safe to call before or after the entry
    /// exists. The tensor's own Dispose/sync callbacks and <see cref="FreeAllCached"/> still reclaim it.</summary>
    public static void PinActivation(Tensor tensor) => Resolve().PinnedActivations.Add(tensor);

    /// <summary>Removes a <see cref="PinActivation"/> mark; the next <see cref="FreeActivations"/> reclaims the
    /// tensor's device buffer like any other activation.</summary>
    public static void UnpinActivation(Tensor tensor) => Resolve().PinnedActivations.Remove(tensor);

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

    /// <summary>Computes the byte size of a tensor's data. Uses <see cref="DType.ComputeByteCount"/> so quantized
    /// tensors (Q4_K, Q5_K, Q8_0, etc.) report their true on-disk byte count rather than <c>elementCount * 0</c>.</summary>
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

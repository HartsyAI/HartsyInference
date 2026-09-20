using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Gpu;

namespace HartsyInference.Cuda;

/// <summary>GPU memory transfer helper with weight and activation caching. Weights preload via PreloadWeight() and stay until FreeAllCached(); activations set by CacheActivation() after each op and are consumed by the next op's CopyToDevice(). Lazy sync: CPU access to DataPointer triggers a GPU→CPU sync on demand. <para><b>Multi-backend safety:</b> all mutable state lives in a per-BACKEND <see cref="State"/> object (one per <see cref="CudaBackend"/> registration, identified by a process-unique <see cref="State.Key"/> that is never reused), resolved at every call via the ambient thread-static the owning backend binds at each op entry (<c>CudaBackend.EnterOp</c>). Activation callbacks capture their owning State directly, so a tensor produced on backend A always syncs/frees against A's context and stream even when backend B was used more recently. History: this was once a single set of static fields (constructing a second CudaBackend silently retargeted ALL cached state → cross-device frees → CUDA_ERROR_ILLEGAL_ADDRESS on both GPUs), then keyed by CUDA context handle — which collapsed two backends on the SAME device into one State (primary contexts are one-per-device), giving last-writer-wins stream bindings, cross-backend H2D copies on the wrong stream, and Dispose wiping the sibling's live weights. Per-backend keys + the ambient make same-device backends fully independent; the cost is that a host tensor shared by two same-device backends holds two device copies (accepted: preserves the "each backend owns its VRAM" invariant and PreloadWeight's ownership contract, no refcounting).</para></summary>
internal static unsafe class GpuTransferHelper
{
    /// <summary>All per-backend mutable state, identified by a process-unique <see cref="Key"/>.</summary>
    internal sealed class State : GpuResidencyCache<ulong>
    {

        /// <summary>Original registry-routing handle. Kept stable until final retirement even if the owning CudaContext has already zeroed its native handle during a faulted cleanup path.</summary>
        /// <summary>Process-unique identity of this backend registration, never reused. Keys the registry, each
        /// tensor's GPU bindings, and the finalizer-cleanup buckets. The base draws it from
        /// <see cref="GpuBindingKeys"/> — the sequence every backend shares — so a tensor resident on a CUDA and a
        /// Vulkan device at once holds one binding per cache and neither teardown clears the other's.</summary>
        public nint Key => BindingKey;

        public nint RegisteredContextHandle;

        /// <summary>Upload counts for host tensors that miss both caches. A tensor re-uploaded with unchanged host data is behaving like a weight, whoever created it — on its second upload it is promoted into the weight cache (see <see cref="TryAutoPromote"/>), making pipelines that never call <c>PreloadWeights</c> (the audio stack) GPU-resident instead of PCIe-bound. Weak-keyed so tracked tensors stay collectible; the state dies with its tensor. Per-State so promotion bookkeeping stays with the backend that owns the device copy.</summary>
        public readonly ConditionalWeakTable<Tensor, UploadState> UploadTracker = new();

        /// <summary>Releases every unpinned activation, and every Q8_1 sidecar riding on one.</summary>
        /// <remarks>An override rather than the base's version because two things here are CUDA's: the sidecars,
        /// which are swept wholesale — pinned survivors included, since a consumer that misses one simply
        /// re-quantizes — and the graph arenas, whose pointers are owned as a block and must never be handed back
        /// individually. Everything else matches the base: sweep first, skip the pinned, clear the binding as the
        /// entry goes so a much-later finalizer cannot resurrect a retired cleanup bucket.</remarks>
        public override void FreeActivations()
        {
            Context?.EnsureCurrent();
            // Called between pipeline stages, never mid-op, so any parked orphan is already ownerless.
            SweepOrphans();
            foreach (Tensor tensor in SidecarCache.Keys.ToList())
            {
                RemoveSidecar(this, tensor);
            }
            List<KeyValuePair<Tensor, (ulong Buffer, long Bytes)>>? survivors = null;
            foreach (KeyValuePair<Tensor, (ulong Buffer, long Bytes)> entry in ActivationCache)
            {
                if (PinnedActivations.Contains(entry.Key))
                {
                    (survivors ??= new List<KeyValuePair<Tensor, (ulong Buffer, long Bytes)>>()).Add(entry);
                    continue;
                }
                entry.Key.ClearGpuBinding(Key);
                CachedPointers.Remove(entry.Value.Buffer);
                if (!IsArenaPtr(this, entry.Value.Buffer))
                {
                    CudaMemory.FreeAsync(entry.Value.Buffer, StreamHandle);
                }
            }
            ActivationCache.Clear();
            if (survivors is not null)
            {
                foreach (KeyValuePair<Tensor, (ulong Buffer, long Bytes)> entry in survivors)
                {
                    ActivationCache[entry.Key] = entry.Value;
                }
            }
        }

        /// <summary>The shared cache's sweep, with the three guards CUDA's free paths need.</summary>
        /// <remarks>Reached from the shared op scope, which is the only reason this is an override rather than a
        /// wrapper: the base calls <c>Residency.SweepOrphans()</c>, so a guard that lives anywhere else is a guard
        /// the op scope does not have.
        ///
        /// <para>Never during a stream capture. <c>cuMemFreeAsync</c> on a buffer allocated BEFORE the capture
        /// began is rejected outright with <c>CUDA_ERROR_INVALID_VALUE</c> — measured, and it is what aborted all
        /// three graph tests the first time this migration was attempted. They stay parked, and the first op after
        /// the capture ends sweeps them. The capture probe sits behind the empty-set check on purpose: a driver
        /// call on every op entry would cost more than the deferral does.</para>
        ///
        /// <para>Demoted auto-promoted weights go back through <c>cuMemFree</c>, not the async pool they were never
        /// allocated from, so they are swept separately.</para></remarks>
        public override void SweepOrphans()
        {
            if (!OrphanSweepEnabled)
            {
                return;
            }
            if (PendingPersistentFrees.Count != 0)
            {
                SweepPersistentFrees(this);
            }
            if (PendingOrphanCount == 0)
            {
                return;
            }
            if (StreamHandle != 0)
            {
                CudaDriverApi.cuStreamIsCapturing(StreamHandle, out int captureStatus).ThrowOnError();
                if (captureStatus != 0)
                {
                    return;
                }
            }
            base.SweepOrphans();
        }

        /// <summary>Auto-promoted weight buffers demoted mid-op because a device write rebound their tensor to a different buffer. They come from <c>cuMemAlloc</c> (<see cref="CudaMemory.AllocatePersistent"/>), so they must be released with <c>cuMemFree</c> and NOT parked in <see cref="PendingOrphans"/>, which frees against the async pool. Freed by the next op's sweep, after the current op's finally blocks have run — freeing inline would double-free, since the demoted buffer is usually that same op's input.</summary>
        public readonly HashSet<ulong> PendingPersistentFrees = new();

        /// <summary>Graph-capture arenas: while a decode-step graph is being captured (<see cref="BeginGraphArena"/>), <see cref="AllocateDevice"/> bump-allocates from a per-capture pre-reserved buffer instead of the stream-ordered pool — so the captured graph contains ZERO memAlloc/memFree nodes for step intermediates (measured on gemma3: 875 alloc + 797 free nodes of 2264 total, each replaying every token). One arena per LIVE graph (the batch scheduler can hold several captured graphs at once — sharing one bump buffer would alias them); pointers inside any live arena are never freed individually (every free path checks <see cref="IsArenaPtr"/>); an arena is released as a whole when its graph is disposed (<see cref="FreeGraphArena"/>). Overflow falls back to normal pool allocation (correct, just adds nodes) and logs once.</summary>
        public readonly List<(ulong basePtr, nuint capacity)> LiveArenas = new();
        public ulong ArenaBase;      // the arena of the capture in progress (also in LiveArenas)
        public nuint ArenaCapacity;
        public nuint ArenaOffset;
        public bool ArenaActive;
        public bool ArenaOverflowLogged;

        /// <summary>Q8_1 activation sidecars emitted by quantize-at-producer kernels (xq int8 + per-32-block scale xd + int-sum xs device buffers, K = the producing row width). Keyed by the F32 output tensor; consumed by the dp4a Linear path in place of its own quantize launch. Invalidated (buffers freed) whenever the tensor is re-bound, synced to host, or disposed — see <see cref="CacheActivation"/>.</summary>
        public readonly Dictionary<Tensor, (ulong xq, ulong xd, ulong xs, int k)> SidecarCache = new(ReferenceEqualityComparer.Instance);

        /// <summary>Stream handle for deferred GPU memory frees and sync-before-D2H.</summary>
        public nint StreamHandle;

        /// <summary>Streaming cache reference, used to drain its upload stream + trim the device's stream-ordered allocator pool when an OOM retry needs to reclaim memory locked up in pool reservations. Null when the backend's streaming cache hasn't been wired (test setups, CPU/Vulkan).</summary>
        public IStreamingWeightCache? StreamingCache;

        /// <summary>The owning CUDA context. Held so the lazy sync/dispose callbacks (which fire from arbitrary threads — finalizers, async continuations, etc.) can bind the context before issuing any CUDA Driver API call. Without this, a callback that fires on a thread that's never bound the context would hit CUDA_ERROR_INVALID_CONTEXT.</summary>
        public CudaContext? Context;

        /// <summary>Set at the start of backend teardown. A retiring state remains in the registry only so its owner can clean it through explicit-state APIs; it is immediately excluded from ambient, sole-state, context-fallback, and same-device routing.</summary>
        public volatile bool Retiring;

        /// <summary>Set when this state's backend is fully torn down (<see cref="CompleteRetire"/>). Stale promoted-weight callbacks that survived teardown (queued by tensor finalizers before their callbacks could be detached) check this FIRST and bail out — reading a bool field is safe on a resurrected object graph, whereas touching <see cref="UploadTracker"/> is not (a ConditionalWeakTable whose Container was finalized while the state was unreachable throws NRE from its freed dependent handles — the GGUF model-switch crash).</summary>
        public volatile bool Unregistered;

        private int _activeCallbacks;
        private readonly ManualResetEventSlim _callbacksDrained = new(initialState: true);

        /// <summary>Claims a tensor lifecycle callback only while this State is routable. The second retirement check closes the race where teardown publishes Retiring immediately after the first check.</summary>
        protected override bool TryEnterCallback()
        {
            if (Retiring || Unregistered) return false;
            if (Interlocked.Increment(ref _activeCallbacks) == 1) _callbacksDrained.Reset();
            if (!Retiring && !Unregistered) return true;
            ExitCallback();
            return false;
        }

        protected override void ExitCallback()
        {
            if (Interlocked.Decrement(ref _activeCallbacks) == 0) _callbacksDrained.Set();
        }

        public void WaitForCallbacksToDrain() => _callbacksDrained.Wait();

        // ── Residency queries ───────────────────────────────────────────────────────────────────────────────
        // What callers and tests actually want to know about this cache, stated as questions about a tensor rather
        // than as reads of whichever dictionary happens to hold it. The collections above are an implementation of
        // residency, not the definition of it, and asserting against them couples every test to that choice.

        /// <summary>Which cache holds this tensor's device copy, if either.</summary>
        /// <exception cref="InvalidOperationException">Both do. <see cref="CopyToDevice"/> checks weights first, so
        /// the activation — the newer bytes, written by an op — would be shadowed by the stale weight on every later
        /// read. That is the "auto-promote discards device writes" bug, and the demotion in
        /// <see cref="CacheActivation"/> exists to prevent it; this asks rather than assumes.</exception>
        public GpuResidencyTier TierOf(Tensor tensor)
        {
            bool weight = Weights.ContainsKey(tensor);
            bool activation = Activations.ContainsKey(tensor);
            if (weight && activation)
            {
                throw new InvalidOperationException(
                    $"Tensor {tensor.Shape} {tensor.DType} is cached as BOTH a weight and an activation on state "
                    + $"{Key}. A weight lookup wins, so every later read would serve the pre-op bytes and silently "
                    + "discard what the op wrote.");
            }
            return weight ? GpuResidencyTier.Weight
                : activation ? GpuResidencyTier.Activation
                : GpuResidencyTier.None;
        }

        /// <summary>Whether this tensor's activation is marked to survive <see cref="FreeActivations"/>.</summary>
        public bool IsPinnedActivation(Tensor tensor) => Pinned.Contains(tensor);

        /// <summary>Whether this cache owns the allocation behind a device pointer, and so will free it itself.</summary>
        public bool OwnsBuffer(ulong devicePtr) => CachedBuffers.Contains(devicePtr);

        /// <summary>Resident weights.</summary>
        public int WeightCount => Weights.Count;

        /// <summary>Resident activations.</summary>
        public int ActivationCount => Activations.Count;

        /// <summary>Cached dtype conversions of resident weights.</summary>
        public int WeightCastCount => WeightCasts.Count;

        /// <summary>Activations marked to survive a bulk free.</summary>
        public int PinnedActivationCount => Pinned.Count;

        /// <summary>Device allocations this cache owns, across every tier. Zero after a full teardown, whatever
        /// route the buffers took to get there.</summary>
        public int CachedBufferCount => CachedBuffers.Count;

        /// <summary>Device bytes held by resident activations.</summary>
        /// <remarks>A total rather than a per-tensor lookup because the callers that need it — teardown probes that
        /// deliberately drop every tensor reference to prove the cache still holds the memory — have no tensor left
        /// to ask about. <see cref="CachedBytes"/> deliberately measures permanent weight/cast residency instead.</remarks>
        public long ActivationBytes
        {
            get
            {
                long total = 0;
                foreach ((ulong _, long bytes) in Activations.Values)
                {
                    total += bytes;
                }
                return total;
            }
        }


        // ── What the shared cache needs from this backend ────────────────────────────────────────────────

        /// <summary>Transient allocation: stream-ordered, out of the async pool, or bump-allocated from the live
        /// graph-capture arena so a captured graph contains no alloc/free nodes for step intermediates.</summary>
        protected override ulong AllocateDevice(long bytes) => GpuTransferHelper.AllocateDeviceFor(this, (nuint)bytes);

        /// <summary>Resident weights are freed with the synchronous free, so they are allocated synchronously and
        /// kept out of the stream-ordered pool. Recorded, because the free path cannot tell the two apart from the
        /// pointer alone.</summary>
        protected override ulong AllocateWeight(long bytes)
        {
            Context?.EnsureCurrent();
            ulong dptr = CudaMemory.AllocatePersistent((nuint)bytes);
            PersistentBuffers.Add(dptr);
            return dptr;
        }

        /// <summary>Pointers that came from <see cref="AllocateWeight"/> and must go back through the synchronous
        /// free rather than the stream-ordered pool.</summary>
        internal readonly HashSet<ulong> PersistentBuffers = new();

        /// <summary>Set for the duration of a teardown sweep, where every free is synchronous on a drained
        /// stream.</summary>
        /// <remarks>Mid-op, a transient goes back to the stream-ordered pool: the free is ordered after the work
        /// that used it, which is the whole reason the pool exists. At teardown that ordering is meaningless and
        /// the async path is actively wrong — the stream is about to go away, and the driver rejects a
        /// stream-ordered free against it. The original teardown drained the stream and freed synchronously; this
        /// keeps that, now that one method serves both.</remarks>
        internal bool SynchronousFrees;

        protected override void FreeDevice(ulong buffer, long bytes)
        {
            if (SynchronousFrees || PersistentBuffers.Remove(buffer))
            {
                CudaMemory.Free(buffer);
                return;
            }
            CudaMemory.FreeAsync(buffer, StreamHandle, this);
        }

        protected override void Upload(ulong destination, Tensor source, long bytes) =>
            GpuTransferHelper.UploadTo(this, destination, source, (nuint)bytes);

        protected override void DownloadSynced(nint hostDestination, ulong source, long bytes)
        {
            Context?.EnsureCurrent();
            if (StreamHandle != 0)
            {
                CudaDriverApi.cuStreamSynchronize(StreamHandle).ThrowOnError();
            }
            CudaMemory.CopyDeviceToHost((void*)hostDestination, source, (nuint)bytes);
        }

        protected override void MakeCurrent() => Context?.EnsureCurrent();

        /// <summary>Buffers bump-allocated from a graph-capture arena, recorded at ALLOCATION time.</summary>
        /// <remarks>Asking "is this address inside a live arena?" at free time is not the same question. A captured
        /// graph's arena leaves <see cref="LiveArenas"/> when the graph is disposed, and after that every buffer it
        /// handed out looks like an ordinary allocation — so the next teardown frees, individually, memory that was
        /// already released wholesale. The original code avoided this by capturing an arena-backed flag when the
        /// activation was bound; this records the same fact at the same moment, and
        /// <see cref="GpuTransferHelper.FreeGraphArena"/> drops the range when the arena goes.</remarks>
        internal readonly HashSet<ulong> ArenaBuffers = new();

        /// <summary>A graph arena owns its allocations wholesale; nothing inside one is freed individually.</summary>
        protected override bool IsExternallyOwned(ulong buffer) =>
            ArenaBuffers.Contains(buffer) || GpuTransferHelper.IsArenaPtr(this, buffer);

        /// <summary>A Q8_1 sidecar describes an activation's CONTENTS, so a rebind stales it as surely as a swap
        /// does — including an in-place op that writes through the same buffer.</summary>
        protected override void OnActivationEvicted(Tensor tensor, ulong buffer) =>
            GpuTransferHelper.RemoveSidecar(this, tensor);

        /// <summary>A pin here means "survive FreeActivations", which destroys the device copy. Paging out keeps
        /// the contents, so it is allowed — and it is the whole point of the low-VRAM lever, whose target IS the
        /// pinned cross-step state.</summary>
        protected override bool MayOffload(Tensor tensor) => true;

        /// <summary>Bulk offload is memory pressure, so a tensor reclaimed by one must not be auto-promoted
        /// straight back: the device copy is precisely what is being given up.</summary>
        protected override void OnActivationOffloaded(Tensor tensor)
        {
            UploadTracker.GetOrCreateValue(tensor).Blocked = true;
        }

        /// <summary>An op wrote through a tensor this cache had promoted: the promoted copy is stale. Block
        /// re-promotion so a host-mutated scratch tensor settles back to plain streaming instead of thrashing, and
        /// route the buffer to the synchronous free it was allocated from.</summary>
        protected override void OnWeightDemoted(Tensor tensor, ulong buffer)
        {
            if (UploadTracker.TryGetValue(tensor, out UploadState? promo) && promo.Promoted)
            {
                promo.Promoted = false;
                promo.Blocked = true;
            }
            // Deliberately NOT queued for a separate free here. The shared cache parks the displaced buffer as an
            // orphan and frees it at the start of the next op — which is the same deferral PendingPersistentFrees
            // existed to provide, and for the same reason: this buffer is usually the current op's own input, whose
            // finally block has not run yet. Queueing it as well meant two frees of one pointer. Leaving it in
            // PersistentBuffers is what makes the orphan sweep release it through cuMemFree rather than the pool.
        }

        /// <summary>A tensor uploaded twice with unchanged host data is behaving like a weight, whoever made it.
        /// Promotion happens HERE — on the miss, before a transient exists — because a promoted weight needs the
        /// persistent allocator, and re-uploading into a pool block would be the wrong lifetime.</summary>
        protected override bool TryMakeResidentOnMiss(Tensor tensor, long bytes, out ulong buffer)
        {
            buffer = 0;
            if (!GpuTransferHelper.ShouldAutoPromote(this, tensor, (nuint)bytes))
            {
                return false;
            }
            // Read the host pointer BEFORE allocating. It runs EnsureCpuData, and when ANOTHER backend already
            // holds this tensor that fires its demote hook, which makes the OTHER backend's context current and
            // does not put ours back — allocating then lands in the wrong context and the pointer cached here is
            // unusable (CUDA_ERROR_INVALID_VALUE on first use). Materialize first, re-assert our context, then
            // allocate against it.
            nint hostData = (nint)tensor.DataPointer;
            Context?.EnsureCurrent();
            buffer = AllocateWeight(bytes);
            CudaMemory.CopyHostToDevice(buffer, (void*)hostData, (nuint)bytes);
            // PromoteToWeight plants the keyed demotion binding: host data stays authoritative for a promoted
            // weight, so any later host access must DROP the device copy rather than sync it back.
            PromoteToWeight(tensor, buffer);
            return true;
        }


        // ── Views onto the shared cache's storage ────────────────────────────────────────────────────────
        // The base owns the collections now. These exist so the CUDA-only paths in this file — graph arenas,
        // Q8_1 sidecars, the capture window, persistent frees — can still reach them. They are deliberately
        // internal and deliberately not exposed further: the tests were moved onto TierOf/the counters in
        // alpha.103 precisely so nothing outside this file depends on the storage shape.

        internal Dictionary<Tensor, ulong> WeightCache => Weights;

        internal Dictionary<Tensor, (ulong Buffer, long Bytes)> ActivationCache => Activations;

        internal Dictionary<Tensor, Dictionary<DType, (ulong Buffer, long Bytes)>> WeightCastCache => WeightCasts;

        internal HashSet<ulong> CachedPointers => CachedBuffers;

        internal Dictionary<ulong, long> Orphans => PendingOrphans;

        /// <summary>Buffers displaced by a rebind that no caller has claimed yet.</summary>
        internal int PendingOrphanCount => PendingOrphans.Count;

        internal HashSet<Tensor> PinnedActivations => Pinned;


        /// <summary>Device bytes held by PERMANENT residency — weights and their cached dtype conversions.</summary>
        /// <remarks>Computed rather than accumulated. The running total was maintained by hand at half a dozen
        /// insert and evict sites, and delegating those to the shared cache silently stopped updating it — a
        /// counter that drifts to zero while the memory is still resident is worse than no counter. Activations are
        /// deliberately excluded and reported separately by <see cref="ActivationBytes"/>.</remarks>
        public long CachedBytes
        {
            get
            {
                long total = 0;
                foreach (Tensor weight in Weights.Keys)
                {
                    total += (long)GpuTransferHelper.ByteSize(weight);
                }
                foreach (Dictionary<DType, (ulong Buffer, long Bytes)> casts in WeightCasts.Values)
                {
                    foreach ((ulong _, long bytes) in casts.Values)
                    {
                        total += bytes;
                    }
                }
                return total;
            }
        }

        /// <summary>Step-graph capture-window alloc/free tracker. Per-State: this used to be process-wide statics on <see cref="CudaMemory"/>, so a second backend beginning its own capture cleared the first backend's in-flight window and folded its own non-capturing allocations into the wrong backend's report. <para>Not diagnostic-only any more: the recorded stream is what lets <see cref="GpuTransferHelper"/> tell a GRAPH-PRIVATE allocation (made on the capturing stream — a virtual address the driver releases the instant an aborted capture is discarded) apart from an ordinary allocation that merely happened while capture was active (the streaming weight cache's uploads run on a separate, non-capturing upload stream and are real regardless of the compute stream's capture outcome). See <see cref="PurgeAbortedCaptureAllocs"/>.</para></summary>
        public bool TrackCaptureWindow;
        public readonly Dictionary<ulong, (nuint bytes, nint stream)> CaptureAllocs = new();
        public long CaptureAllocBytes, CaptureFreeBytes;
        public int CaptureAllocCount, CaptureFreeCount;
    }

    /// <summary>Per-tensor H2D upload bookkeeping for weight auto-promotion.</summary>
    internal sealed class UploadState { public int Count; public bool Promoted; public bool Blocked; }

    /// <summary>Auto-promotion kill switch: set <c>HARTSY_NO_AUTOPROMOTE=1</c> to reproduce the old always-re-upload behavior (A/B benchmarking, or if a pipeline mutates host weight data through a stashed raw pointer that bypasses <c>DataPointer</c>/<c>AsSpan</c> and so can't be seen by the demote-on-host-access hook).</summary>
    public static bool AutoPromoteWeights => !EngineKnobs.NoAutopromote.Value;

    /// <summary>Free-VRAM floor preserved by auto-promotion (activations, transients, cuBLAS workspaces need room). A promotion that would dip below this floor is skipped and the tensor streams as before. Override via <c>HARTSY_AUTOPROMOTE_HEADROOM_MB</c>.</summary>
    private static long _autoPromoteHeadroomBytes => EngineKnobs.AutopromoteHeadroomMb.Value << 20;

    /// <summary>Tensors below this size are never auto-promoted: small hot tensors are cheap to re-upload and are the most likely to be mutated scratch buffers.</summary>
    private const nuint AutoPromoteMinBytes = 1 << 20;

    /// <summary>Registered states keyed by <see cref="State.Key"/>. Concurrent: registration happens on backend construction threads while Resolve() reads from compute threads.</summary>
    private static readonly ConcurrentDictionary<nint, State> _states = new();

    /// <summary>Last-registered live state per CUDA context handle — the ambient-less resolution fallback for helper calls arriving outside a backend op (tests, tooling). Ambiguous by construction when two backends share a device; <see cref="Resolve"/> warns once per handle in that case.</summary>
    private static readonly ConcurrentDictionary<nint, State> _byContext = new();

    /// <summary>The calling thread's current backend State, bound by <c>CudaBackend.EnterOp</c> at every op entry. This is THE resolution path — cuCtxGetCurrent cannot distinguish two same-device backends (they share the primary context), so context identity stopped being usable as the owner identity.</summary>
    [ThreadStatic]
    private static State? _ambient;

    /// <summary>Fast path for ambient-less callers: the sole registered state when exactly one backend exists. Null when zero or 2+ states are registered.</summary>
    private static volatile State? _sole;

    /// <summary>Serializes registration/retirement so <see cref="_sole"/> and the indexes stay consistent.</summary>
    private static readonly object _registryLock = new();

    /// <summary>Context handles already warned about ambiguous ambient-less resolution.</summary>
    private static readonly ConcurrentDictionary<nint, bool> _ambiguityWarned = new();

    /// <summary>Debug tripwire (HARTSY_ASSERT_AMBIENT=1): throw instead of falling back when an ambient-less call happens while multiple backends are live — catches op entry points the EnterOp transform missed.</summary>
    private static bool _assertAmbient => EngineKnobs.AssertAmbient.Value;

    /// <summary>Fallback state for calls made before any backend registers (unit tests exercising pure helpers). Its stream is 0 and context null, so every code path degrades to the safe no-op branch.</summary>
    private static readonly State _unregistered = new();

    /// <summary>Creates and registers a NEW state for one backend. Called once from <see cref="CudaBackend"/>'s constructor with the context, compute stream, and streaming cache; the returned state is the backend's identity for ambient binding, tensor GPU bindings, and finalizer-cleanup routing.</summary>
    public static State Register(CudaContext context, nint stream, IStreamingWeightCache? streamingCache)
    {
        State state = new State
        {
            RegisteredContextHandle = context.Handle,
            Context = context,
            StreamHandle = stream,
            StreamingCache = streamingCache,
        };
        lock (_registryLock)
        {
            _states[state.Key] = state;
            _byContext[context.Handle] = state;
            RecomputeSoleLocked(excluded: null);
        }
        return state;
    }

    private static bool IsResolvable(State state) => !state.Retiring && !state.Unregistered;

    /// <summary>Atomically removes a state from every implicit routing path while retaining its explicit handle for owner-driven cleanup. Must run before native teardown starts.</summary>
    internal static void BeginRetire(State state)
    {
        try
        {
            lock (_registryLock)
            {
                if (!state.Retiring && !state.Unregistered)
                {
                    state.Retiring = true;
                    RebuildRoutes(state.RegisteredContextHandle, state);
                }
            }
        }
        finally
        {
            // Even an unexpected route-rebuild exception cannot leave a claimed callback racing native teardown.
            if (ReferenceEquals(_ambient, state)) _ambient = null;
            state.WaitForCallbacksToDrain();
        }
    }

    /// <summary>Removes and inerts a state after all explicit-state cleanup attempts have completed.</summary>
    internal static void CompleteRetire(State state)
    {
        lock (_registryLock)
        {
            nint contextHandle = state.RegisteredContextHandle;
            state.Retiring = true;
            // Publish the terminal flag before dropping the strong registry root. Stale tensor callbacks can
            // then bail out without touching finalized ConditionalWeakTable/cache state.
            state.Unregistered = true;
            _states.TryRemove(state.Key, out _);
            RebuildRoutes(contextHandle, state);
            state.StreamHandle = 0;
            state.StreamingCache = null;
            state.Context = null;
            state.RegisteredContextHandle = 0;
        }
        if (ReferenceEquals(_ambient, state)) _ambient = null;
    }

    /// <summary>Compatibility entry point for callers that have already released the state's resources.</summary>
    public static void Unregister(State state)
    {
        BeginRetire(state);
        CompleteRetire(state);
    }

    /// <summary>Recomputes the context fallback and sole-state fast path from resolvable states.</summary>
    private static void RebuildRoutes(nint contextHandle, State excluded)
    {
        if (contextHandle != 0 && _byContext.TryGetValue(contextHandle, out State? current)
            && (ReferenceEquals(current, excluded) || !IsResolvable(current)))
        {
            State? replacement = null;
            foreach (State candidate in _states.Values)
            {
                if (!IsResolvable(candidate) || ReferenceEquals(candidate, excluded)
                    || candidate.RegisteredContextHandle != contextHandle) continue;
                if (replacement is null || candidate.Key.ToInt64() > replacement.Key.ToInt64()) replacement = candidate;
            }
            if (replacement is null) _byContext.TryRemove(contextHandle, out _);
            else _byContext[contextHandle] = replacement;
        }

        RecomputeSoleLocked(excluded);
    }

    private static void RecomputeSoleLocked(State? excluded)
    {
        State? sole = null;
        foreach (State candidate in _states.Values)
        {
            if (!IsResolvable(candidate) || ReferenceEquals(candidate, excluded)) continue;
            if (sole is not null) { sole = null; break; }
            sole = candidate;
        }
        _sole = sole;
    }

    internal static int RegisteredStateCount => _states.Count;
    internal static nint[] RegisteredStateKeysForTests => [.. _states.Keys];
    internal static bool IsStateRegistered(nint key) => _states.ContainsKey(key);
    internal static nint ContextFallbackKey(nint contextHandle) =>
        _byContext.TryGetValue(contextHandle, out State? state) && IsResolvable(state) ? state.Key : 0;

    /// <summary>Binds <paramref name="state"/> as the calling thread's current backend. Called by <c>CudaBackend.EnterOp</c> on every op entry; cheap (one thread-static write).</summary>
    internal static void SetAmbient(State state) => _ambient = state;

    /// <summary>Resolves the calling thread's owning backend State: the ambient bound by the current op, else the sole registered state, else the last state registered for the thread's current CUDA context (ambiguous for same-device siblings — warned once), else an inert empty state (pre-registration test paths).</summary>
    private static State Resolve()
    {
        State? ambient = _ambient;
        if (ambient is not null)
        {
            if (IsResolvable(ambient)) return ambient;
            // A backend began teardown after this thread entered an operation. Falling through to _sole or a
            // same-context sibling would silently route the remainder of B's op through A's caches/stream. Backend
            // disposal requires externally quiesced requests; make a violation loud and ownership-safe.
            throw new ObjectDisposedException(nameof(CudaBackend),
                "The ambient CUDA backend is retiring. Quiesce active operations before disposing the backend.");
        }
        State? sole = _sole;
        if (sole is not null && IsResolvable(sole)) return sole;
        if (_states.IsEmpty) return _unregistered;
        if (_assertAmbient)
        {
            throw new InvalidOperationException(
                "GpuTransferHelper.Resolve: no ambient State with multiple backends registered — an op entry point "
                + "missed the EnterOp transform (HARTSY_ASSERT_AMBIENT=1).");
        }
        if (CudaDriverApi.cuCtxGetCurrent(out nint current) == 0
            && _byContext.TryGetValue(current, out State? state) && IsResolvable(state))
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
            if (IsResolvable(s) && s.RegisteredContextHandle == contextHandle) count++;
        }
        return count;
    }

    /// <summary>Every live registered state sharing <paramref name="deviceOrdinal"/> — the same-device escalation sweep for OOM retries (a sibling backend's pool reservations can hold the memory this one needs).</summary>
    internal static List<State> StatesOnDevice(int deviceOrdinal)
    {
        List<State> result = [];
        foreach (State s in _states.Values)
        {
            if (IsResolvable(s) && s.Context?.DeviceOrdinal == deviceOrdinal) result.Add(s);
        }
        return result;
    }

    /// <summary>The ambient (or resolved) state's compute stream, for <see cref="CudaMemory"/>'s allocator routing; 0 when nothing is resolvable (sync-allocation fallback).</summary>
    internal static nint ResolvedStreamHandle => Resolve().StreamHandle;

    /// <summary>The calling thread's owning backend State, for callers (<see cref="CudaMemory"/>'s capture-window tracker) that need more than just the stream handle. Same ambient-first resolution as every other lookup.</summary>
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

    /// <summary>OOM-retry hook: drains both the compute stream and the streaming cache's upload stream, then trims the device mempool so memory queued via <c>cuMemFreeAsync</c> is released back to the driver allocator. Called from <see cref="CudaMemory.Allocate"/> when the first <c>cuMemAlloc</c> returned OOM. Without this, an op that should succeed against just-evicted streaming memory will throw OOM even though several GB are technically free.</summary>
    public static void SyncStreamsAndReleasePool()
    {
        State s = Resolve();
        s.Context?.EnsureCurrent();
        SyncStream();
        // Cache also drains its own upload stream + calls cuMemPoolTrimTo on the
        // default mempool. No-op if no streaming cache is wired (CPU/Vulkan, tests).
        s.StreamingCache?.DrainAndReleasePool();
        // Same-device escalation: a SIBLING backend's stream-ordered pool reservations can hold the very bytes
        // this OOM needs (each backend frees onto its own stream). Draining a sibling touches only its streams
        // and the shared device pool, never its caches — but CudaStreamingWeightCache.Enter() rebinds the
        // THREAD AMBIENT to the sibling's State as a side effect (same-device backends share a primary context,
        // so context identity can't route these calls). Under HARTSY_SAME_GPU_CONCURRENT=1 the sibling is NOT
        // guaranteed quiescent — that "it cannot stall its in-flight work" assumption only ever held behind
        // DeviceGate — so a sibling stream/pool call here CAN throw (e.g. the sibling tears down concurrently,
        // or a transient driver error). A throw here used to skip the ambient restore below, leaving this
        // thread's ambient stuck on the sibling's State: every GpuTransferHelper call this thread makes
        // afterwards — including cuMemFreeAsync — would silently target the sibling's cache and stream instead
        // of ours. That is the FreeActivations CUDA_ERROR_INVALID_VALUE / Dispose double-free this fixes.
        // try/finally + per-sibling catch make the escalation genuinely best-effort: one sibling failing to
        // drain must not corrupt our ambient (finally) nor abort the sweep for other siblings (catch+log).
        if (s.Context is not null)
        {
            try
            {
                foreach (State sibling in StatesOnDevice(s.Context.DeviceOrdinal))
                {
                    if (ReferenceEquals(sibling, s) || sibling.Unregistered)
                    {
                        continue;
                    }
                    try
                    {
                        if (sibling.StreamHandle != 0)
                        {
                            CudaDriverApi.cuStreamSynchronize(sibling.StreamHandle);
                        }
                        sibling.StreamingCache?.DrainAndReleasePool();
                    }
                    catch (Exception ex)
                    {
                        Logs.Error("[Cuda] same-device sibling drain failed during OOM escalation — " +
                            "continuing with this backend's own pool trim only.", ex);
                    }
                }
            }
            finally
            {
                // Unconditional: a sibling drain rebinds our thread's ambient (CudaStreamingWeightCache.Enter),
                // success or failure. Restore ours here, or the very allocation retry this call serves — and
                // anything this thread touches before its next EnterOp — lands on the sibling's stream/state.
                SetAmbient(s);
            }
        }
        // Always trim the default pool too: with no streaming cache wired (auto-transfer paths, tests), every
        // cuMemFreeAsync'd transient stays RESERVED in the stream-ordered pool. An OOM retry that never trims
        // reports "GPU full" (cuMemGetInfo counts reservations as used) even though most of it is reusable —
        // the fp8 auto-transfer Flux/T5 OOM at 0.9% free.
        TrimPool();
    }

    // Diagnostic: HARTSY_H2D_TRACE=1 logs the first cache misses with shape/dtype so a re-uploaded weight set is
    // distinguishable from ordinary activation traffic. Small misses are logged too — a DiT that re-uploads a
    // handful of tiny per-channel vectors every block hides thousands of them per step behind a trace that only
    // showed the megabyte-scale ones.
    private static bool _traceBigMisses => EngineKnobs.H2dTrace.Value;
    private static int _bigMissTraceCount;

    /// <summary>How many big misses to log; raise it to see past the text-encode phase into denoise.</summary>
    private static int _traceBigMissLimit => EngineKnobs.H2dTraceLimit.Value;

    /// <summary>The device pointer for a tensor: cached weight, then cached activation, then a fresh upload.</summary>
    /// <remarks>Priority, miss counting, auto-promotion and the transient upload are the shared cache's. What is
    /// CUDA's rides the overrides: <c>TryMakeResidentOnMiss</c> promotes a twice-uploaded tensor before any
    /// transient exists, <c>AllocateDevice</c> serves the arena during a graph capture, and <c>Upload</c> carries
    /// the stream-ordered copy and the H2D profiling.</remarks>
    public static ulong CopyToDevice(Tensor cpuTensor) => Resolve().CopyToDevice(cpuTensor);


    /// <summary>Copies data from a GPU buffer back into a CPU tensor.</summary>
    public static void CopyToHost(Tensor cpuTensor, ulong gpuPtr, nuint byteSize)
    {
        CudaMemory.CopyDeviceToHost(cpuTensor.DataPointer, gpuPtr, byteSize);
    }

    /// <summary>Allocates a GPU buffer.</summary>
    public static ulong AllocateDevice(nuint byteSize) => AllocateDeviceFor(Resolve(), byteSize);

    /// <summary>The allocation body, against an explicit state so the shared cache's <c>AllocateDevice</c> override
    /// can reach it without another ambient resolution.</summary>
    internal static ulong AllocateDeviceFor(State s, nuint byteSize)
    {
        if (s.ArenaActive)
        {
            nuint aligned = (byteSize + 255) & ~(nuint)255;
            if (s.ArenaOffset + aligned <= s.ArenaCapacity)
            {
                ulong p = s.ArenaBase + s.ArenaOffset;
                s.ArenaOffset += aligned;
                s.ArenaBuffers.Add(p);
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

    /// <summary>Copies a tensor's host bytes to the device on the compute stream.</summary>
    /// <remarks>STREAM-ORDERED on the same stream as the allocation, so the copy is naturally ordered after it and
    /// the consuming kernel sees the data. This replaced a per-miss full <c>cuStreamSynchronize</c> that drained the
    /// whole async pipeline on EVERY small host-tensor upload — the Wan DiT alone missed ~14 tiny
    /// modulation/scratch tensors per block-forward, and that drain dominated the generation. Pageable source
    /// stages synchronously before returning, so the host buffer is safe to reuse; pinned source stays alive until
    /// the stream-ordered free. No CPU read happens here, so only stream order matters, and it holds.</remarks>
    internal static unsafe void UploadTo(State s, ulong destination, Tensor source, nuint byteSize)
    {
        using Profiling.NvtxRange _upload = Profiling.NvtxRange.Push(byteSize > (1u << 20)
            ? (Profiling.NvtxRange.ProfileShapes
                ? $"H2D_BIG {string.Join("x", Enumerable.Range(0, source.Shape.Rank).Select(i => source.Shape[i]))} {source.DType}"
                : "H2D_BIG")
            : "H2D_SMALL");
        if (_traceBigMisses && byteSize > (1u << 20)
            && Interlocked.Increment(ref _bigMissTraceCount) <= _traceBigMissLimit)
        {
            Logs.Debug($"[Cuda][H2D] {(long)byteSize >> 20} MB {source.DType} "
                + $"[{string.Join("x", Enumerable.Range(0, source.Shape.Rank).Select(i => source.Shape[i]))}]");
        }
        CudaMemory.CopyHostToDeviceAsync(destination, source.DataPointer, byteSize, s.StreamHandle);
    }

    /// <summary>True when the pointer lies inside ANY live graph-capture arena (never individually freed).</summary>
    internal static bool IsArenaPtr(State s, ulong p)
    {
        for (int i = 0; i < s.LiveArenas.Count; i++)
            if (p >= s.LiveArenas[i].basePtr && p < s.LiveArenas[i].basePtr + s.LiveArenas[i].capacity)
                return true;
        return false;
    }

    /// <summary>Allocates a fresh per-capture arena and activates it. Call immediately before a decode-step graph capture; pair with <see cref="EndGraphArena"/>, and release via <see cref="FreeGraphArena"/> when the captured graph is disposed. Returns the arena base (0 = allocation failed, arena disabled for this capture).</summary>
    public static ulong BeginGraphArena()
    {
        State s = Resolve();
        long mb = EngineKnobs.GraphArenaMb.Value;
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

    /// <summary>Deactivates the in-progress capture arena (buffers handed out stay valid for the graph's lifetime) and logs the actual bytes used, for capacity tuning.</summary>
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
                // The recorded pointers are deliberately NOT forgotten here. Once an arena is released its
                // addresses are gone, so a buffer that came from it must stay un-freeable for as long as anything
                // still references it — dropping the record is what lets a later sweep free an address the driver
                // already reclaimed. The original carried the same fact as a per-activation flag that was never
                // cleared either.
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
        FreeDevice(Resolve(), gpuPtr);
    }

    /// <summary>Explicit-owner variant used after a state has been removed from implicit routing for teardown.</summary>
    internal static void FreeDevice(State s, ulong gpuPtr)
    {
        if (gpuPtr != 0 && !s.CachedPointers.Contains(gpuPtr) && !IsArenaPtr(s, gpuPtr))
        {
            // The caller owns this one after all, so SweepOrphans must not free it a second time.
            if (s.Orphans.Count != 0) s.Orphans.Remove(gpuPtr);
            CudaMemory.FreeAsync(gpuPtr, s.StreamHandle, s);
        }
    }

    /// <summary>Frees buffers displaced by a <see cref="CacheActivation"/> rebind that no caller claimed.</summary>
    /// <remarks>Called at the start of each op, so every previous op's <c>finally</c> has already run and anything
    /// still parked provably has no owner. Sweeping here rather than inside CacheActivation is what keeps the
    /// in-place case (where the displaced buffer is the op's own input) from being double-freed.</remarks>
    /// <summary><c>HARTSY_ORPHAN_SWEEP=0</c> restores the pre-fix behaviour (displaced buffers leak) — a bisect handle for a change that sits on every op's allocation path.</summary>
    private static bool OrphanSweepEnabled => EngineKnobs.OrphanSweep.Value;




    /// <summary>Releases demoted auto-promoted weight buffers with <c>cuMemFree</c>, the allocator they came from. Runs a stream sync first: the op that demoted them may still have had them queued as an input.</summary>
    private static void SweepPersistentFrees(State s)
    {
        if (s.StreamHandle != 0)
        {
            CudaDriverApi.cuStreamIsCapturing(s.StreamHandle, out int captureStatus).ThrowOnError();
            if (captureStatus != 0) return;
            CudaDriverApi.cuStreamSynchronize(s.StreamHandle).ThrowOnError();
        }
        foreach (ulong ptr in s.PendingPersistentFrees)
        {
            s.CachedPointers.Remove(ptr);
            CudaMemory.Free(ptr);
        }
        s.PendingPersistentFrees.Clear();
    }

    /// <summary>Registers a Q8_1 sidecar (from a quantize-at-producer kernel) for an activation tensor. Call AFTER <see cref="CacheActivation"/> for the same tensor — CacheActivation invalidates any previous sidecar as part of rebinding.</summary>
    internal static void RegisterSidecar(Tensor tensor, ulong xq, ulong xd, ulong xs, int k)
    {
        State s = Resolve();
        RemoveSidecar(s, tensor);
        s.SidecarCache[tensor] = (xq, xd, xs, k);
    }

    /// <summary>Looks up a Q8_1 sidecar for a dp4a GEMV input (M=1 decode rows only — producers emit per-row sidecars and the decode path is single-row).</summary>
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
            if (!IsArenaPtr(s, sc.xq)) CudaMemory.FreeAsync(sc.xq, s.StreamHandle, s);
            if (!IsArenaPtr(s, sc.xd)) CudaMemory.FreeAsync(sc.xd, s.StreamHandle, s);
            if (!IsArenaPtr(s, sc.xs)) CudaMemory.FreeAsync(sc.xs, s.StreamHandle, s);
        }
    }

    /// <summary>Binds an op's output buffer to its tensor so the next op reads it on the device.</summary>
    /// <remarks>The four-step rebind, the stale-buffer eviction, the weight demotion and the keyed binding all live
    /// in the shared cache now. What is CUDA's rides the hooks: <c>OnActivationEvicted</c> drops the tensor's Q8_1
    /// sidecar, <c>OnWeightDemoted</c> blocks re-promotion and routes the displaced persistent buffer to the
    /// synchronous free, and <c>IsExternallyOwned</c> keeps graph-arena pointers out of every free path.</remarks>
    public static void CacheActivation(Tensor tensor, ulong gpuPtr, nuint byteSize) =>
        Resolve().CacheActivation(tensor, gpuPtr, (long)byteSize);




    /// <summary>True when this backend's activation cache currently holds a device copy of <paramref name="tensor"/>.</summary>
    internal static bool HasCachedActivation(Tensor tensor) => Resolve().ActivationCache.ContainsKey(tensor);

    /// <summary>The device pointer already backing <paramref name="tensor"/>, without uploading anything. For an
    /// op that OVERWRITES its destination in full, where <see cref="CopyToDevice"/> would stage host bytes the very
    /// next line discards.</summary>
    internal static bool TryGetCachedDevice(Tensor tensor, out ulong gpuPtr) =>
        Resolve().TryGetCached(tensor, out gpuPtr);


    /// <summary>Number of activations this backend currently holds on device.</summary>
    internal static int CachedActivationCount => Resolve().ActivationCache.Count;

    /// <summary>Materializes one cached activation to host and releases its device buffer — the named spelling of the bare <c>_ = t.DataPointer</c> host-materialize idiom, running the very same sync callback. No-op for a tensor with no device copy. Auto-promotion is deliberately left alone: the cross-step caches that use this are re-uploaded unchanged every step and are MEANT to be promoted back into the weight cache. The bulk <see cref="OffloadActivations"/> blocks promotion instead, because there the resident copy is the thing being reclaimed.</summary>
    public static void OffloadActivation(Tensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        _ = tensor.DataPointer;
    }

    /// <summary>Materializes cached activations to host, largest first, until <paramref name="targetBytes"/> has
    /// been released; returns the bytes actually freed. Skips pinned tensors.</summary>
    public static long OffloadActivations(long targetBytes)
    {
        long freed = Resolve().OffloadActivations(targetBytes);
        // The frees above are cuMemFreeAsync, which returns blocks to the stream-ordered pool and leaves them
        // RESERVED until trimmed — without this the reclaimed VRAM never shows up as free.
        if (freed > 0) TrimPool();
        return freed;
    }


    /// <summary> Removes a just-published activation binding when a multi-output operation fails while publishing a later output. The caller retains ownership of <paramref name="expectedGpuPtr"/> and must free it afterward. Existing contents displaced by <see cref="CacheActivation"/> follow the normal orphan-sweep lifecycle; this helper only prevents a failed operation from exposing a partial new result set. </summary>
    internal static bool TryUncacheActivation(Tensor tensor, ulong expectedGpuPtr)
    {
        State s = Resolve();
        if (!s.ActivationCache.TryGetValue(tensor, out (ulong Buffer, long Bytes) cached)
            || cached.Buffer != expectedGpuPtr)
        {
            return false;
        }

        s.ActivationCache.Remove(tensor);
        s.PinnedActivations.Remove(tensor);
        s.CachedPointers.Remove(expectedGpuPtr);
        tensor.ClearGpuBinding(s.Key);
        return true;
    }

    /// <summary>Releases every cached dtype conversion, keeping the weights themselves. For the streaming cache,
    /// whose per-block eviction would otherwise orphan the casts.</summary>
    internal static long EvictAllWeightCasts()
    {
        State s = Resolve();
        long freed = s.ReleaseAllWeightCasts();
        return freed;
    }


    /// <summary>A previously stored conversion of <paramref name="weight"/> to <paramref name="want"/>.</summary>
    /// <remarks>Takes the target dtype now. CUDA kept one cast per weight while the shared cache keys per
    /// (weight, dtype) — a superset — and both call sites already know the GEMM dtype they are asking for.</remarks>
    public static bool TryGetWeightCast(Tensor weight, DType want, out ulong castPtr)
    {
        bool found = Resolve().TryGetWeightCast(weight, want, out ulong cast);
        castPtr = found ? cast : 0;
        return found;
    }


    /// <summary>Stores a conversion of <paramref name="weight"/> to <paramref name="want"/>; the buffer becomes
    /// cache-owned, so a later free by the caller leaves it alone.</summary>
    public static void CacheWeightCast(Tensor weight, DType want, ulong castPtr, nuint byteSize) =>
        Resolve().StoreWeightCast(weight, want, castPtr, (long)byteSize);


    /// <summary>Uploads a weight ahead of first use so no op pays a cache-miss transfer mid-generation. Returns
    /// false when this backend already holds it.</summary>
    /// <remarks>The persistent allocation is the subclass's <c>AllocateWeight</c> override: a resident weight is
    /// freed synchronously, so it must be allocated synchronously and kept out of the stream-ordered pool.</remarks>
    public static bool PreloadWeight(Tensor weight)
    {
        State s = Resolve();
        if (s.TierOf(weight) == GpuResidencyTier.Weight)
        {
            return false;
        }
        s.PreloadWeight(weight);
        return true;
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

    /// <summary>True if the weight is currently cached on the device. Streaming uploads check this to skip already-resident tensors.</summary>
    internal static bool IsWeightCached(Tensor weight) => Resolve().WeightCache.ContainsKey(weight);

    internal static bool IsActivationCached(Tensor tensor) => Resolve().ActivationCache.ContainsKey(tensor);

    /// <summary>Whether a tensor that has now missed twice should be made resident, and there is room for it.</summary>
    /// <remarks>Weights are the only tensors that live long enough to be uploaded twice — an activation is a fresh
    /// object every op — so a second miss on the same object identifies a weight belonging to a pipeline that never
    /// called PreloadWeights, at the cost of one duplicate upload.</remarks>
    internal static bool ShouldAutoPromote(State s, Tensor cpuTensor, nuint byteSize)
    {
        if (!AutoPromoteWeights || byteSize < AutoPromoteMinBytes)
        {
            return false;
        }
        UploadState tracker = s.UploadTracker.GetOrCreateValue(cpuTensor);
        tracker.Count++;
        if (tracker.Blocked || tracker.Count < 2)
        {
            return false;
        }
        if (CudaDriverApi.cuMemGetInfo(out nuint free, out _) != 0
            || (long)free - (long)byteSize < _autoPromoteHeadroomBytes)
        {
            return false;
        }
        tracker.Promoted = true;
        return true;
    }

    /// <summary>Registers an already-uploaded weight in the cache. The caller is responsible for the alloc + H2D copy (sync or async); this just records the tensor → dptr mapping and bumps the byte counter.</summary>
    internal static void RegisterCachedWeight(Tensor weight, ulong dptr, nuint byteSize)
    {
        State s = Resolve();
        // A tensor may not become a weight while it is still a live activation: CopyToDevice checks the weight
        // cache first, so the activation's bytes — what an op just wrote — would be shadowed by this upload on
        // every later read, which is the auto-promote-discards-device-writes bug arriving from the other side.
        //
        // Every caller already satisfies this, but incidentally rather than by intent: each reads DataPointer to
        // find the host bytes to upload, and that fires the activation's sync callback, which evicts the entry.
        // An invariant held by a side effect of an unrelated read is one line away from being lost — a caller that
        // uploads from a pinned or mapped buffer would never touch DataPointer, and CudaStreamingWeightCache is
        // already most of the way there. So it is checked where it is established rather than where it would be
        // observed: this is the write that would corrupt the read, and weight registration is a load-time path.
        if (s.ActivationCache.ContainsKey(weight))
        {
            throw new InvalidOperationException(
                $"Tensor {weight.Shape} {weight.DType} is being registered as a weight on state {s.Key} while it "
                + "is still cached as an activation. Materialize or evict the activation first (reading "
                + "DataPointer does both); registering now would shadow the activation on every later read.");
        }
        s.WeightCache[weight] = dptr;
        s.CachedPointers.Add(dptr);
    }

    /// <summary>Removes a weight from the cache and hands back its pointer, leaving the caller to free it. Also
    /// drops any cached dtype conversion of it.</summary>
    internal static bool TryUnregisterCachedWeight(Tensor weight, out ulong dptr)
    {
        State s = Resolve();
        if (!s.TryEvictWeight(weight, out ulong evicted))
        {
            dptr = 0;
            return false;
        }
        dptr = evicted;
        s.PersistentBuffers.Remove(dptr);
        return true;
    }


    /// <summary>Releases the device copies of these weights, and any cached dtype conversions of them.</summary>
    public static void FreeWeights(IEnumerable<Tensor> weights) => Resolve().FreeWeights(weights);


    /// <summary>Frees every cached GPU buffer (weights, activations and casts) for the CURRENT backend.</summary>
    public static void FreeAllCached() => FreeAllCached(Resolve());


    /// <summary>Releases every cached allocation this state owns. The shared cache clears the bindings, frees the
    /// buffers (weights, activations and casts) and empties the collections; the arena and sidecar bookkeeping
    /// around it is CUDA's.</summary>
    internal static void FreeAllCached(State s)
    {
        s.Context?.EnsureCurrent();
        if (s.StreamHandle != 0) CudaDriverApi.cuStreamSynchronize(s.StreamHandle).ThrowOnError();
        foreach (Tensor tensor in s.ActivationCache.Keys.ToArray()) RemoveSidecar(s, tensor);
        s.SynchronousFrees = true;
        try { s.FreeAllCached(); }
        finally { s.SynchronousFrees = false; }
        s.SidecarCache.Clear();
        s.PendingPersistentFrees.Clear();
    }




    /// <summary>Evicts all cached GPU buffers.</summary>
    public static void EvictAll()
    {
        FreeAllCached();
    }

    internal static void EvictAll(State state) => FreeAllCached(state);

    /// <summary>Purges cache entries left dangling by an ABORTED step-graph capture. Every alloc issued on the CAPTURING stream while capture is active becomes a graph memory node; if the capture is aborted before the graph is ever instantiated/launched (an exception partway through the captured step), the driver releases those virtual addresses as part of discarding the never-launched graph. Anything still mapping to one of them in this State's caches (an activation cached mid-capture, a weight dtype-cast computed mid-capture, …) is now dangling: the next <see cref="FreeActivations"/>/<c>Dispose</c> that tries to actually free one gets <c>CUDA_ERROR_INVALID_VALUE</c> — the pointer is no longer a live stream-ordered allocation. Called from <c>CudaBackend.StepGraphReset</c> right after <c>CudaGraph.AbortCapture</c>. <para>Scoped to <paramref name="capturingStream"/>: <see cref="State.CaptureAllocs"/> also records allocations the streaming weight cache made on its separate upload stream while capture happened to be active — those are ordinary allocations untouched by the abort and must be left alone, or a live streamed weight would be silently stranded (never freed, no longer tracked by anything).</para></summary>
    internal static void PurgeAbortedCaptureAllocs(State s, nint capturingStream)
    {
        HashSet<ulong>? stale = null;
        lock (s.CaptureAllocs)
        {
            foreach (KeyValuePair<ulong, (nuint bytes, nint stream)> kv in s.CaptureAllocs)
            {
                if (kv.Value.stream == capturingStream)
                {
                    (stale ??= new HashSet<ulong>()).Add(kv.Key);
                }
            }
            if (stale is not null)
            {
                foreach (ulong p in stale) s.CaptureAllocs.Remove(p);
            }
        }
        if (stale is null || stale.Count == 0)
        {
            return;
        }

        List<Tensor>? staleActivations = null;
        foreach (KeyValuePair<Tensor, (ulong Buffer, long Bytes)> kv in s.ActivationCache)
            if (stale.Contains(kv.Value.Buffer))
                (staleActivations ??= new List<Tensor>()).Add(kv.Key);
        if (staleActivations is not null)
            foreach (Tensor t in staleActivations)
            {
                s.ActivationCache.Remove(t);
                s.PinnedActivations.Remove(t);
                // The GPU-private buffer is already gone with the discarded graph — clear the binding so a
                // later Dispose/finalizer of this (now-unreachable) tensor doesn't try to free it again.
                t.ClearGpuBinding(s.Key);
            }

        List<Tensor>? staleWeights = null;
        foreach (KeyValuePair<Tensor, ulong> kv in s.WeightCache)
            if (stale.Contains(kv.Value))
                (staleWeights ??= new List<Tensor>()).Add(kv.Key);
        if (staleWeights is not null)
            foreach (Tensor t in staleWeights)
            {
                // Takes the buffer back WITHOUT freeing it: the address belonged to the discarded capture and the
                // driver released it the instant the capture was thrown away, so freeing it again is a double free.
                if (s.TryEvictWeight(t, out ulong staleWeight))
                {
                    s.PersistentBuffers.Remove(staleWeight);
                }
            }

        List<Tensor>? staleCasts = null;
        foreach (KeyValuePair<Tensor, Dictionary<DType, (ulong Buffer, long Bytes)>> kv in s.WeightCastCache)
            if (kv.Value.Values.Any(c => stale.Contains(c.Buffer)))
                (staleCasts ??= new List<Tensor>()).Add(kv.Key);
        if (staleCasts is not null)
            foreach (Tensor t in staleCasts)
                s.WeightCastCache.Remove(t);

        List<Tensor>? staleSidecars = null;
        foreach (KeyValuePair<Tensor, (ulong xq, ulong xd, ulong xs, int k)> kv in s.SidecarCache)
            if (stale.Contains(kv.Value.xq) || stale.Contains(kv.Value.xd) || stale.Contains(kv.Value.xs))
                (staleSidecars ??= new List<Tensor>()).Add(kv.Key);
        if (staleSidecars is not null)
            foreach (Tensor t in staleSidecars)
                s.SidecarCache.Remove(t);

        foreach (ulong p in stale)
        {
            s.CachedPointers.Remove(p);
            s.Orphans.Remove(p);
        }

        Logs.Warning($"[Cuda] step-graph capture aborted mid-window — purged {stale.Count} graph-private cache " +
            "entries that would otherwise dangle (CUDA_ERROR_INVALID_VALUE on the next free).");
    }



    /// <summary>Returns pool-reserved-but-free device memory to the driver WITHOUT clearing the activation cache. <c>cuMemFreeAsync</c> (every activation/dispose free) hands blocks back to the stream-ordered mempool, which RESERVES them (counts as used in cuMemGetInfo) until trimmed. Unlike <see cref="FreeActivations"/> this leaves live cached activations intact — only already-freed blocks are reclaimed — so it is safe to call mid-computation (e.g. between VAE decode tiles) to cap peak at one unit's working set without corrupting tensors still in use. Syncs the stream first so queued async frees complete before the trim.</summary>
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

}

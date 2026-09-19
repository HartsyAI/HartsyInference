using HartsyInference.Core.Tensors;

namespace HartsyInference.Gpu;

/// <summary>Keeps tensors resident on a device and hands out their buffers, so an op chain runs without host
/// round-trips. Generic over the buffer handle, because none of this logic depends on what a buffer is.
///
/// <para>Every GPU backend needs exactly this and had written its own: the same three caches, the same
/// weight → activation → fresh-upload lookup order, and the same four-step activation bind. The copies had drifted
/// in ways that mattered — one backend drained the finalizer queue and the other never did, so tensors finalized
/// rather than disposed leaked their device memory there; and one keyed every device's binding as <c>0</c>, which
/// works only until two of its devices are used at once.</para>
///
/// <para><b>What a subclass supplies</b> is the five operations that genuinely need the API: allocate, free, upload,
/// download, and make-current. Everything else — the caches, the counters, the binding lifecycle, offload — is here
/// and is the same everywhere. The hooks below exist for the parts that are genuinely per-backend: CUDA's graph
/// arenas and auto-promotion, Vulkan's capture retain-list.</para></summary>
/// <typeparam name="TBuffer">The backend's buffer handle. Reference equality (or value equality for a handle struct)
/// must identify a buffer, since the cache keys its own bookkeeping on it.</typeparam>
public abstract class GpuResidencyCache<TBuffer> : IGpuResidency
    where TBuffer : notnull
{
    /// <inheritdoc/>
    /// <remarks>Starts at 1. Zero is left alone as the bucket the older context-less callers used, so a cache that
    /// has not been migrated yet cannot collide with one that has.
    ///
    /// <para>The sequence lives in Core, shared with CUDA's own state registry, rather than on this class. Two
    /// reasons, both learned by getting it wrong: a static field inside a generic class exists once per CLOSED type,
    /// so a counter here would give <c>GpuResidencyCache&lt;ulong&gt;</c> and <c>GpuResidencyCache&lt;VulkanBuffer&gt;</c>
    /// one each and both would hand out key 1; and a counter anywhere in this package would still not cover the CUDA
    /// registry, which allocates its own keys and would collide with this one the first time a tensor was resident on
    /// a CUDA and a Vulkan device at once.</para></remarks>
    public nint BindingKey { get; } = GpuBindingKeys.Next();

    /// <summary>Weights: uploaded once, kept until explicitly freed. Reference equality, because two distinct tensors
    /// with equal contents are still two tensors, and a tensor's contents can change under it.</summary>
    protected readonly Dictionary<Tensor, TBuffer> Weights = new(ReferenceEqualityComparer.Instance);

    /// <summary>Activations: an op's output, kept so the next op reads it in place. Carries its size because
    /// offloading has to know what it is about to reclaim without asking the backend.</summary>
    protected readonly Dictionary<Tensor, (TBuffer Buffer, long Bytes)> Activations = new(ReferenceEqualityComparer.Instance);

    /// <summary>Activations exempt from <see cref="OffloadActivations"/> — cross-step state that must not move.</summary>
    protected readonly HashSet<Tensor> Pinned = new(ReferenceEqualityComparer.Instance);

    /// <summary>A resident weight's dtype conversions, keyed by weight then by target dtype. A quantized or fp8
    /// weight is converted once and reused, instead of being converted again by every GEMM that reads it.
    ///
    /// <para>Only WEIGHTS are cached this way. An activation changes every step, so a keyed entry would never be hit
    /// and would grow without bound.</para></summary>
    /// <remarks>Each entry carries its own size: a conversion's byte count is not the weight's — that is the whole
    /// point of converting — so a backend that needs the size to free an allocation cannot recompute it from the
    /// tensor.</remarks>
    protected readonly Dictionary<Tensor, Dictionary<DType, (TBuffer Buffer, long Bytes)>> WeightCasts = new(ReferenceEqualityComparer.Instance);

    /// <summary>Every buffer the cache owns. A backend frees its own op temporaries through
    /// <see cref="ReleaseIfNotCached"/>, which consults this so a temporary that has since been cached — an output
    /// bound to its tensor — is not freed out from under the tensor that now points at it.</summary>
    protected readonly HashSet<TBuffer> CachedBuffers = [];

    /// <summary>Buffers displaced by a rebind, waiting to find out whether anyone owned them.
    ///
    /// <para>When an op binds a tensor to a new buffer, the buffer it displaces is either the op's own input — whose
    /// <c>finally</c> will release it — or nobody's, in which case no tensor maps to it and nothing ever will. The two
    /// are indistinguishable at the moment of displacement, so the buffer parks here instead: a caller's
    /// <see cref="ReleaseIfNotCached"/> claims it, and <see cref="SweepOrphans"/> frees whatever is still unclaimed
    /// when the NEXT op starts, by which point every previous op's cleanup has provably run.</para></summary>
    /// <remarks>Freeing at teardown instead is not a fix, it is a deferral: the buffers accumulate for the whole
    /// generation. Measured on CUDA before this existed — twelve <c>Linear</c> calls at a 563 MB output stranded
    /// 5942 MB and broke unrelated work sharing the card.</remarks>
    protected readonly Dictionary<TBuffer, long> PendingOrphans = [];

    /// <summary>Whether a weight's dtype conversion is kept resident. Off trades recompute for roughly a third of the
    /// weight footprint, which is what lets a large fp8 model fit a card it otherwise would not.</summary>
    public bool CacheWeightCasts { get; set; } = true;

    private long _hits;
    private long _misses;
    private long _d2hSyncs;
    private int _disposed;

    /// <inheritdoc/>
    public long D2hSyncCount => Interlocked.Read(ref _d2hSyncs);

    /// <summary>Cache lookups served without a transfer.</summary>
    public long Hits => Interlocked.Read(ref _hits);

    /// <summary>Cache lookups that had to upload.</summary>
    public long Misses => Interlocked.Read(ref _misses);

    /// <inheritdoc/>
    public void ResetD2hSyncCount() => Interlocked.Exchange(ref _d2hSyncs, 0);

    // ── What a backend must provide ──────────────────────────────────────────────────────────────────────

    /// <summary>Allocates <paramref name="bytes"/> of device memory for an op output or temporary.</summary>
    protected abstract TBuffer AllocateDevice(long bytes);

    /// <summary>Allocates device memory for a RESIDENT WEIGHT, which may need a different allocator.</summary>
    /// <remarks>Weights and transients can have different lifetimes and therefore different free paths. CUDA is the
    /// case that forces this: a preloaded weight comes from <c>cuMemAlloc</c> and is released with <c>cuMemFree</c>,
    /// deliberately kept out of the stream-ordered pool that every transient uses, because it is freed
    /// synchronously. Routing weights through <see cref="AllocateDevice"/> would hand a pool block to a synchronous
    /// free — an error on that API, not a style question. Backends with one allocator ignore this.</remarks>
    protected virtual TBuffer AllocateWeight(long bytes) => AllocateDevice(bytes);

    /// <summary>Releases a device allocation. May be deferred: the cache never assumes the memory is reclaimed by
    /// the time this returns, only that it has been handed back.</summary>
    protected abstract void FreeDevice(TBuffer buffer, long bytes);

    /// <summary>Copies a tensor's host bytes into a device buffer.</summary>
    protected abstract void Upload(TBuffer destination, Tensor source, long bytes);

    /// <summary>Waits for the work that produced <paramref name="source"/>, then copies it to host memory. The wait
    /// is the callee's: only the backend knows what "produced" means for its queue model.</summary>
    protected abstract void DownloadSynced(nint hostDestination, TBuffer source, long bytes);

    /// <summary>Makes this cache's device current, where the API has such a notion. A no-op elsewhere.</summary>
    protected abstract void MakeCurrent();

    // ── Hooks for the parts that really are per-backend ──────────────────────────────────────────────────

    /// <summary>Called after a fresh upload that was not cached, for a backend that tracks transient buffers so it
    /// can free them at its next flush.</summary>
    protected virtual void OnTransientUploaded(TBuffer buffer, Tensor source) { }

    /// <summary>Offers a cache miss to the backend before the cache allocates a transient for it. Returning true
    /// means the backend has made the tensor resident itself and <paramref name="buffer"/> is what the caller
    /// should use.</summary>
    /// <remarks>A separate hook from <see cref="OnTransientUploaded"/> because it is a different moment and, for
    /// the backend that needs it, a different buffer. CUDA promotes a twice-uploaded tensor to a resident weight
    /// here, BEFORE any transient exists, into a freshly allocated persistent buffer — so a post-upload hook could
    /// only promote the pool block it was handed (wrong allocator for a weight) or upload the same bytes twice.
    /// Fires after the miss is counted, so promotion still reads as a miss exactly as it did before.</remarks>
    protected virtual bool TryMakeResidentOnMiss(Tensor tensor, long bytes, out TBuffer? buffer)
    {
        buffer = default;
        return false;
    }

    /// <summary>Whether this buffer belongs to something else and must never be freed individually — a graph arena
    /// owns its allocations wholesale.</summary>
    protected virtual bool IsExternallyOwned(TBuffer buffer) => false;

    /// <summary>Offers a buffer being released to a backend that must retain it instead — a live graph capture holds
    /// every buffer it recorded. Returning true means the backend has taken ownership.</summary>
    protected virtual bool TryRetainOnRelease(TBuffer buffer) => false;

    /// <summary>Called as an activation leaves the cache, for per-buffer bookkeeping hung off it.</summary>
    protected virtual void OnActivationEvicted(Tensor tensor, TBuffer buffer) { }

    /// <summary>Called on a cache miss while the backend is recording a replayable graph. Uploading then would freeze
    /// this moment's bytes into the graph, so a backend that captures should refuse here.</summary>
    protected virtual void OnMissDuringCapture(Tensor tensor) { }

    /// <summary>Whether a host read of a cached activation is currently illegal — true while recording a graph, where
    /// the read would both violate the capture contract and drain a queue that is mid-record.</summary>
    protected virtual bool HostReadForbidden => false;

    /// <summary>Gate for the callbacks a tensor fires on sync or dispose. Returning false skips the callback
    /// entirely, for a backend that is tearing down and whose device state can no longer be touched.</summary>
    /// <remarks>A tensor's binding outlives the moment it was planted: the tensor may be disposed, or finalized and
    /// resurrected, long after the cache that bound it has retired. On CUDA a stale callback reaching a
    /// <c>ConditionalWeakTable</c> on such an object graph threw outright, which is what made a GGUF model swap
    /// crash. Paired with <see cref="ExitCallback"/> so a subclass can hold a lock across the body.</remarks>
    protected virtual bool TryEnterCallback() => true;

    /// <summary>Releases whatever <see cref="TryEnterCallback"/> took. Runs in a <c>finally</c>.</summary>
    protected virtual void ExitCallback() { }

    /// <summary>Whether a bulk <see cref="OffloadActivations"/> may page this activation out.</summary>
    /// <remarks>Pinned by default, and the default is the conservative reading rather than the obvious one. Paging
    /// out is NOT destructive — the contents go to host and come back on the next read — so a pin, which exists to
    /// survive the destructive bulk free, does not have to block it. A backend whose low-VRAM lever depends on
    /// reclaiming exactly this cross-step state says so by overriding.</remarks>
    protected virtual bool MayOffload(Tensor tensor) => !Pinned.Contains(tensor);

    /// <summary>Called for each activation a BULK <see cref="OffloadActivations"/> reclaims.</summary>
    /// <remarks>Bulk offload is memory pressure: the device copy is the thing being given up, so a backend that
    /// promotes tensors on its own initiative must not immediately promote this one back. A single-tensor offload
    /// deliberately does not fire this — the cross-step caches that use it are re-uploaded unchanged every step and
    /// are meant to become resident again.</remarks>
    protected virtual void OnActivationOffloaded(Tensor tensor) { }

    /// <summary>Called as a resident weight is demoted because an op bound its tensor to an activation buffer. The
    /// weight's own buffer may need releasing through a different allocator than an activation's, and a backend that
    /// promotes weights automatically has to stop re-promoting this one.</summary>
    protected virtual void OnWeightDemoted(Tensor tensor, TBuffer buffer) { }

    // ── The shared algorithm ─────────────────────────────────────────────────────────────────────────────

    /// <summary>A tensor's size in bytes.</summary>
    public static long ByteSize(Tensor tensor) => tensor.DType.ComputeByteCount(tensor.ElementCount);

    /// <summary>Looks the tensor up WITHOUT uploading on a miss, so a caller can take a device-to-device path only
    /// when the source is already resident rather than forcing it to become resident.</summary>
    public bool TryGetCached(Tensor tensor, out TBuffer? buffer)
    {
        if (Weights.TryGetValue(tensor, out TBuffer? weight))
        {
            buffer = weight;
            return true;
        }
        if (Activations.TryGetValue(tensor, out (TBuffer Buffer, long Bytes) activation))
        {
            buffer = activation.Buffer;
            return true;
        }
        buffer = default;
        return false;
    }

    /// <summary>The device buffer holding this tensor: cached weight, then cached activation, then a fresh upload.</summary>
    public TBuffer CopyToDevice(Tensor tensor)
    {
        if (TryGetCached(tensor, out TBuffer? cached))
        {
            Interlocked.Increment(ref _hits);
            return cached!;
        }

        // A miss here while recording means the upload below would bake this instant's bytes into the graph, and
        // every replay would reuse them. The backend decides what to do about it; the default does nothing, which
        // is right for a backend that does not capture.
        OnMissDuringCapture(tensor);

        Interlocked.Increment(ref _misses);
        long bytes = ByteSize(tensor);
        if (TryMakeResidentOnMiss(tensor, bytes, out TBuffer? resident))
        {
            return resident!;
        }
        TBuffer fresh = AllocateDevice(bytes);
        Upload(fresh, tensor, bytes);
        OnTransientUploaded(fresh, tensor);
        return fresh;
    }

    /// <summary>Binds an op's output buffer to its tensor, so the next op reads it on the device and the host only
    /// pays a transfer if it actually looks at the value.</summary>
    public void CacheActivation(Tensor tensor, TBuffer buffer, long bytes)
    {
        // Deliberately does not touch DataPointer: that would allocate and zero a host buffer for every resident
        // activation. The host buffer appears only if host code reads the tensor, inside the sync below.

        // An in-place op re-caches the same tensor. The previous callbacks close over the OLD buffer and would free
        // it out from under the new one, so clear first — keyed, so another device's binding on this tensor stands.
        tensor.ClearGpuBinding(BindingKey);

        if (Activations.TryGetValue(tensor, out (TBuffer Buffer, long Bytes) displaced))
        {
            // Fires on ANY rebind, including an in-place op that writes through the same buffer. What hangs off an
            // activation describes its CONTENTS — a producer-emitted quantized sidecar, say — and a write through
            // the buffer stales that just as surely as swapping the buffer does.
            OnActivationEvicted(tensor, displaced.Buffer);
            if (!EqualityComparer<TBuffer>.Default.Equals(displaced.Buffer, buffer))
            {
                Park(displaced.Buffer, displaced.Bytes);
            }
        }

        // The tensor was a resident weight and an op has just written a device buffer for it. It cannot stay one:
        // CopyToDevice checks Weights BEFORE Activations, so leaving the entry there makes every later read return
        // the pre-op bytes and the device write is silently discarded — a whole-engine correctness bug, not a leak.
        // Demotion here is unconditional, which is wider than strictly necessary and deliberately so: a tensor being
        // bound as an op's output is not a weight any more, whatever route made it one.
        if (Weights.Remove(tensor, out TBuffer? demoted))
        {
            OnWeightDemoted(tensor, demoted);
            if (!EqualityComparer<TBuffer>.Default.Equals(demoted, buffer))
            {
                Park(demoted, ByteSize(tensor));
            }
            // Every cached conversion describes the pre-op contents, so all of them are stale.
            ReleaseWeightCasts(tensor);
        }

        Activations[tensor] = (buffer, bytes);
        CachedBuffers.Add(buffer);
        // Re-cached between being parked and being swept: it has an owner again.
        PendingOrphans.Remove(buffer);

        tensor.SetGpuBinding(
            BindingKey,
            sync: () => SyncActivationToHost(tensor),
            dispose: () => ReleaseActivation(tensor));
    }

    /// <summary>Host code read the tensor: bring the value back and give up the device copy.</summary>
    private void SyncActivationToHost(Tensor tensor)
    {
        // Disposal first, before any collection is touched: this callback was planted on a tensor that can outlive
        // the cache, and by here the device it would talk to may be gone.
        if (Volatile.Read(ref _disposed) != 0 || !TryEnterCallback())
        {
            return;
        }
        try
        {
            SyncActivationToHostCore(tensor);
        }
        finally
        {
            ExitCallback();
        }
    }

    private void SyncActivationToHostCore(Tensor tensor)
    {
        if (HostReadForbidden)
        {
            throw new InvalidOperationException(
                $"A tensor was read from host while {GetType().Name} was recording a graph — the read is illegal "
                + "under the capture contract, and servicing it would drain a queue that is mid-record.");
        }
        if (!Activations.Remove(tensor, out (TBuffer Buffer, long Bytes) entry))
        {
            return;
        }
        Interlocked.Increment(ref _d2hSyncs);
        MakeCurrent();
        unsafe
        {
            DownloadSynced((nint)tensor.EnsureHostBuffer(), entry.Buffer, entry.Bytes);
        }
        OnActivationEvicted(tensor, entry.Buffer);
        Pinned.Remove(tensor);
        CachedBuffers.Remove(entry.Buffer);
        ReleaseBuffer(entry.Buffer, entry.Bytes);
    }

    /// <summary>The tensor is gone: drop the device copy without reading it back.</summary>
    private void ReleaseActivation(Tensor tensor)
    {
        if (Volatile.Read(ref _disposed) != 0 || !TryEnterCallback())
        {
            return;
        }
        try
        {
            ReleaseActivationCore(tensor);
        }
        finally
        {
            ExitCallback();
        }
    }

    private void ReleaseActivationCore(Tensor tensor)
    {
        if (!Activations.Remove(tensor, out (TBuffer Buffer, long Bytes) entry))
        {
            return;
        }
        OnActivationEvicted(tensor, entry.Buffer);
        Pinned.Remove(tensor);
        CachedBuffers.Remove(entry.Buffer);
        ReleaseBuffer(entry.Buffer, entry.Bytes);
    }

    /// <summary>Hands a buffer back, unless something else owns it or wants to keep it.</summary>
    private void ReleaseBuffer(TBuffer buffer, long bytes)
    {
        if (IsExternallyOwned(buffer) || TryRetainOnRelease(buffer))
        {
            return;
        }
        FreeDevice(buffer, bytes);
    }

    /// <summary>Parks a displaced buffer until its ownership is known. See <see cref="PendingOrphans"/>.</summary>
    private void Park(TBuffer buffer, long bytes)
    {
        CachedBuffers.Remove(buffer);
        // Something else frees this one wholesale, so it must never be handed back individually.
        if (!IsExternallyOwned(buffer))
        {
            PendingOrphans[buffer] = bytes;
        }
    }

    /// <summary>Frees the displaced buffers nobody claimed.</summary>
    /// <remarks>Call at the START of an op, never inside <see cref="CacheActivation"/>. That timing is the whole
    /// mechanism: by the start of the next op every previous op's <c>finally</c> has run, so anything still parked
    /// provably has no owner. Sweeping at the point of displacement would double-free the in-place case, where the
    /// displaced buffer is the op's own input and that op's cleanup is still to come.</remarks>
    public virtual void SweepOrphans()
    {
        if (PendingOrphans.Count == 0)
        {
            return;
        }
        foreach ((TBuffer buffer, long bytes) in PendingOrphans)
        {
            // Bound to a tensor again since it was parked, so it is owned and no longer an orphan.
            if (!CachedBuffers.Contains(buffer))
            {
                ReleaseBuffer(buffer, bytes);
            }
        }
        PendingOrphans.Clear();
    }

    /// <summary>Makes an already-uploaded buffer this tensor's resident weight, without re-uploading it.
    ///
    /// <para>The seam a backend needs to promote a tensor it has seen uploaded more than once: the buffer must enter
    /// the weight cache and the owned set together, or the caller's own cleanup frees what the cache now points
    /// at.</para>
    ///
    /// <para>Promotion happens behind the caller's back — nobody asked for this tensor to become resident — so it
    /// plants a demotion binding, and the correctness of the whole mechanism rests on it. Host data stays
    /// authoritative for a promoted weight, so any later host access must DROP the device copy rather than sync it
    /// back; without the binding, a host write would leave the stale device bytes cached and every later
    /// <see cref="CopyToDevice"/> would serve them. That is the same silent wrong-answer bug as a weight an op
    /// writes through, arriving from the other direction.</para></summary>
    /// <remarks>An explicit <see cref="PreloadWeight"/> deliberately plants nothing: the caller asked for residency
    /// and owns the lifetime, so a host read should not silently undo it.</remarks>
    protected void PromoteToWeight(Tensor tensor, TBuffer buffer)
    {
        // A tensor may not be resident in both tiers: a lookup checks weights first, so an activation left behind
        // here would be shadowed by this weight on every later read. Today's only caller fires on a miss, where the
        // tensor is in neither, but this is a general seam and the invariant is cheap to keep rather than assume.
        if (Activations.Remove(tensor, out (TBuffer Buffer, long Bytes) displaced) && !Equals(displaced.Buffer, buffer))
        {
            Pinned.Remove(tensor);
            OnActivationEvicted(tensor, displaced.Buffer);
            Park(displaced.Buffer, displaced.Bytes);
        }
        Weights[tensor] = buffer;
        CachedBuffers.Add(buffer);
        PendingOrphans.Remove(buffer);

        // Keyed, so a second device promoting the same host tensor does not overwrite this one's hook. Sync and
        // dispose are the same action: there is nothing on the device worth reading back.
        tensor.ClearGpuBinding(BindingKey);
        tensor.SetGpuBinding(
            BindingKey,
            sync: () => DemotePromotedWeight(tensor),
            dispose: () => DemotePromotedWeight(tensor));
    }

    /// <summary>Host code touched a tensor this cache promoted on its own initiative: give up the device copy.</summary>
    private void DemotePromotedWeight(Tensor tensor)
    {
        if (Volatile.Read(ref _disposed) != 0 || !TryEnterCallback())
        {
            return;
        }
        try
        {
            if (!Weights.Remove(tensor, out TBuffer? buffer))
            {
                return;
            }
            // No D2H. The host buffer is the authority here — that is what makes this a demotion and not an
            // eviction — and copying the device bytes back would overwrite the write that triggered this.
            OnWeightDemoted(tensor, buffer);
            CachedBuffers.Remove(buffer);
            ReleaseBuffer(buffer, ByteSize(tensor));
            ReleaseWeightCasts(tensor);
        }
        finally
        {
            ExitCallback();
        }
    }

    /// <summary>Releases every cached conversion of one weight.</summary>
    private void ReleaseWeightCasts(Tensor weight)
    {
        if (!WeightCasts.Remove(weight, out Dictionary<DType, (TBuffer Buffer, long Bytes)>? casts))
        {
            return;
        }
        foreach ((TBuffer cast, long castBytes) in casts.Values)
        {
            CachedBuffers.Remove(cast);
            ReleaseBuffer(cast, castBytes);
        }
    }

    /// <summary>Releases every cached conversion of every weight, and reports the bytes reclaimed.</summary>
    /// <remarks>For a backend that evicts weights outside <see cref="FreeWeights"/> — streamed blocks are dropped
    /// one at a time and their casts would otherwise be orphaned, since a cast is keyed by its weight and only
    /// reclaimed alongside it. Measured on a streamed 12B fp8 DiT: ~19 GB of dead casts by VAE-decode time.</remarks>
    public long ReleaseAllWeightCasts()
    {
        long freed = 0;
        foreach (Dictionary<DType, (TBuffer Buffer, long Bytes)> casts in WeightCasts.Values)
        {
            foreach ((TBuffer cast, long bytes) in casts.Values)
            {
                CachedBuffers.Remove(cast);
                ReleaseBuffer(cast, bytes);
                freed += bytes;
            }
        }
        WeightCasts.Clear();
        return freed;
    }

    /// <summary>Removes one weight from the cache and hands its buffer back to the caller, which becomes
    /// responsible for freeing it. Also drops every cached conversion of that weight.</summary>
    /// <remarks>The caller taking the buffer is the point: a streaming cache frees on its own schedule and against
    /// its own stream.</remarks>
    public bool TryEvictWeight(Tensor weight, out TBuffer? buffer)
    {
        ReleaseWeightCasts(weight);
        if (!Weights.Remove(weight, out TBuffer? evicted))
        {
            buffer = default;
            return false;
        }
        weight.ClearGpuBinding(BindingKey);
        CachedBuffers.Remove(evicted!);
        buffer = evicted;
        return true;
    }

    /// <summary>Whether a conversion of this tensor is worth keeping. True only for a resident weight: an activation
    /// is different next step, so its entry would never be hit again.</summary>
    public bool ShouldCacheCast(Tensor weight) => CacheWeightCasts && Weights.ContainsKey(weight);

    /// <summary>A previously stored conversion of <paramref name="weight"/> to <paramref name="want"/>, if any.</summary>
    public bool TryGetWeightCast(Tensor weight, DType want, out TBuffer? buffer)
    {
        buffer = default;
        if (!CacheWeightCasts)
        {
            return false;
        }
        if (WeightCasts.TryGetValue(weight, out Dictionary<DType, (TBuffer Buffer, long Bytes)>? casts)
            && casts.TryGetValue(want, out (TBuffer Buffer, long Bytes) hit))
        {
            Interlocked.Increment(ref _hits);
            buffer = hit.Buffer;
            return true;
        }
        return false;
    }

    /// <summary>Stores a conversion of <paramref name="weight"/> to <paramref name="want"/>. The buffer becomes
    /// cache-owned, so a later <see cref="ReleaseIfNotCached"/> leaves it alone.</summary>
    public void StoreWeightCast(Tensor weight, DType want, TBuffer buffer, long bytes)
    {
        if (!WeightCasts.TryGetValue(weight, out Dictionary<DType, (TBuffer Buffer, long Bytes)>? casts))
        {
            casts = new Dictionary<DType, (TBuffer, long)>();
            WeightCasts[weight] = casts;
        }
        casts[want] = (buffer, bytes);
        CachedBuffers.Add(buffer);
    }

    /// <summary>Releases a buffer the backend allocated for an op, unless the cache has since taken ownership of it.
    /// The guard is the point: an op's output buffer is routinely handed to <see cref="CacheActivation"/> and then
    /// released by the same op's cleanup, and freeing it there would leave the tensor pointing at dead memory.</summary>
    public void ReleaseIfNotCached(TBuffer? buffer, long bytes)
    {
        if (buffer is null || CachedBuffers.Contains(buffer))
        {
            return;
        }
        // The caller did own this one after all, so the sweep must not free it a second time.
        PendingOrphans.Remove(buffer);
        ReleaseBuffer(buffer, bytes);
    }

    /// <summary>Uploads one weight and keeps it resident. Already-resident weights cost nothing.</summary>
    public void PreloadWeight(Tensor weight)
    {
        if (Weights.ContainsKey(weight))
        {
            return;
        }
        // An activation that is about to become a weight keeps its buffer: the bytes are already there, and the
        // binding it holds would otherwise free the buffer the weight cache is now pointing at.
        if (Activations.Remove(weight, out (TBuffer Buffer, long Bytes) promoted))
        {
            weight.ClearGpuBinding(BindingKey);
            Weights[weight] = promoted.Buffer;
            CachedBuffers.Add(promoted.Buffer);
            return;
        }
        MakeCurrent();
        long bytes = ByteSize(weight);
        TBuffer buffer = AllocateWeight(bytes);
        Upload(buffer, weight, bytes);
        Weights[weight] = buffer;
        CachedBuffers.Add(buffer);
    }

    /// <inheritdoc/>
    public virtual void PreloadWeights(IEnumerable<Tensor> weights)
    {
        foreach (Tensor weight in weights)
        {
            PreloadWeight(weight);
        }
    }

    /// <inheritdoc/>
    public virtual void FreeWeights(IEnumerable<Tensor> weights)
    {
        MakeCurrent();
        foreach (Tensor weight in weights)
        {
            if (Weights.Remove(weight, out TBuffer? buffer))
            {
                // A promoted weight carries a demotion binding. Leaving it planted outlives what it refers to: if
                // this tensor is preloaded again the stale callback finds it back in Weights and evicts the new,
                // explicitly-requested residency; and if the tensor is finalized after this cache is disposed, the
                // work queues under a key nobody will ever drain, rooting both forever.
                weight.ClearGpuBinding(BindingKey);
                CachedBuffers.Remove(buffer);
                ReleaseBuffer(buffer, ByteSize(weight));
            }
            // A conversion outlives nothing: its only purpose is to serve the weight that is going away.
            ReleaseWeightCasts(weight);
        }
    }

    /// <inheritdoc/>
    public virtual void FreeAllCached()
    {
        MakeCurrent();
        HashSet<TBuffer> released = [];

        foreach ((Tensor tensor, (TBuffer buffer, long bytes)) in Activations.ToArray())
        {
            tensor.ClearGpuBinding(BindingKey);
            if (released.Add(buffer))
            {
                ReleaseBuffer(buffer, bytes);
            }
        }
        Activations.Clear();
        Pinned.Clear();

        foreach ((Tensor tensor, TBuffer buffer) in Weights.ToArray())
        {
            // Same reason as FreeWeights: a promoted weight's binding must not outlive the cache that planted it.
            tensor.ClearGpuBinding(BindingKey);
            if (released.Add(buffer))
            {
                ReleaseBuffer(buffer, ByteSize(tensor));
            }
        }
        Weights.Clear();

        foreach (Dictionary<DType, (TBuffer Buffer, long Bytes)> casts in WeightCasts.Values)
        {
            foreach ((TBuffer cast, long castBytes) in casts.Values)
            {
                if (released.Add(cast))
                {
                    ReleaseBuffer(cast, castBytes);
                }
            }
        }
        WeightCasts.Clear();

        // Anything still owned but no longer reachable from a cache. An in-place op re-caches its tensor with a NEW
        // buffer, which overwrites the dictionary entry and leaves the previous buffer owned by nothing — clearing
        // the tensor's binding does not release it, it only stops the callbacks firing. Those orphans have to be
        // released HERE or they outlive the device and their finalizer destroys a buffer against a dead one.
        foreach (TBuffer orphan in CachedBuffers)
        {
            if (released.Add(orphan))
            {
                ReleaseBuffer(orphan, 0);
            }
        }
        CachedBuffers.Clear();

        // Parked buffers left CachedBuffers when they were displaced, so the sweep above does not reach them.
        foreach ((TBuffer orphan, long bytes) in PendingOrphans)
        {
            if (released.Add(orphan))
            {
                ReleaseBuffer(orphan, bytes);
            }
        }
        PendingOrphans.Clear();
    }

    /// <inheritdoc/>
    public void PinActivation(Tensor tensor) => Pinned.Add(tensor);

    /// <inheritdoc/>
    public void UnpinActivation(Tensor tensor) => Pinned.Remove(tensor);

    /// <inheritdoc/>
    /// <remarks>Largest first, because the point is to free a given number of bytes in as few host round-trips as
    /// possible. Each one is a real D2H transfer, so this is expensive by construction and belongs at a safe point.</remarks>
    public long OffloadActivations(long targetBytes)
    {
        if (targetBytes <= 0)
        {
            return 0;
        }
        // Snapshot before touching anything: every offload mutates the collection being walked. Buffers something
        // else owns wholesale are never candidates — paging one out would free an address inside a live arena.
        // Pinned entries sort FIRST within a size class where the backend allows them at all: they are the
        // cross-step, read-once-per-step class this is defensible for, while an unpinned transient dies at the next
        // bulk free anyway, so paging it spends a round trip on bytes that were about to be free.
        (Tensor Tensor, long Bytes)[] candidates = [.. Activations
            .Where(entry => MayOffload(entry.Key) && !IsExternallyOwned(entry.Value.Buffer))
            .Select(entry => (entry.Key, entry.Value.Bytes, Pinned: Pinned.Contains(entry.Key)))
            .OrderByDescending(entry => entry.Bytes)
            .ThenByDescending(entry => entry.Pinned)
            .Select(entry => (entry.Key, entry.Bytes))];

        long freed = 0;
        foreach ((Tensor tensor, long bytes) in candidates)
        {
            if (freed >= targetBytes)
            {
                break;
            }
            // Reading the tensor is what evicts it: the sync callback planted by CacheActivation does the transfer
            // and the release, so the tensor stays correct and simply is not resident any more.
            unsafe
            {
                _ = tensor.DataPointer;
            }
            OnActivationOffloaded(tensor);
            freed += bytes;
        }
        return freed;
    }

    /// <inheritdoc/>
    public void DrainFinalizerCleanup() => Tensor.DrainPendingFinalizerGpuCleanup(BindingKey);

    /// <inheritdoc/>
    public virtual string DiagnosticsSummary()
    {
        long weightBytes = Weights.Keys.Sum(ByteSize);
        long activationBytes = Activations.Values.Sum(entry => entry.Bytes);
        long lookups = Hits + Misses;
        double hitRate = lookups == 0 ? 0 : (double)Hits / lookups * 100.0;
        return $"{Weights.Count} weights ({weightBytes / (1024.0 * 1024.0):F1} MiB), "
            + $"{Activations.Count} activations ({activationBytes / (1024.0 * 1024.0):F1} MiB), "
            + $"{Pinned.Count} pinned, {hitRate:F1}% hit rate over {lookups} lookups, {D2hSyncCount} D2H syncs";
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        FreeAllCached();
        // Anything queued by a finalizer after this point has no cache left to run against.
        Tensor.DiscardPendingFinalizerGpuCleanup(BindingKey);
        DisposeCore();
        GC.SuppressFinalize(this);
    }

    /// <summary>Subclass cleanup, after the caches are emptied.</summary>
    protected virtual void DisposeCore() { }
}

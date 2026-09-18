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
    /// <para>The counter lives on a NON-generic holder deliberately. A static field declared inside a generic class
    /// exists once per closed type, so a counter here would give <c>GpuResidencyCache&lt;ulong&gt;</c> and
    /// <c>GpuResidencyCache&lt;VulkanBuffer&gt;</c> one each and both would hand out key 1 — the same collision this
    /// key exists to prevent, moved up a level and invisible until a second backend joins the base.</para></remarks>
    public nint BindingKey { get; } = GpuBindingKeys.Next();

    /// <summary>Weights: uploaded once, kept until explicitly freed. Reference equality, because two distinct tensors
    /// with equal contents are still two tensors, and a tensor's contents can change under it.</summary>
    protected readonly Dictionary<Tensor, TBuffer> Weights = new(ReferenceEqualityComparer.Instance);

    /// <summary>Activations: an op's output, kept so the next op reads it in place. Carries its size because
    /// offloading has to know what it is about to reclaim without asking the backend.</summary>
    protected readonly Dictionary<Tensor, (TBuffer Buffer, long Bytes)> Activations = new(ReferenceEqualityComparer.Instance);

    /// <summary>Activations exempt from <see cref="OffloadActivations"/> — cross-step state that must not move.</summary>
    protected readonly HashSet<Tensor> Pinned = new(ReferenceEqualityComparer.Instance);

    /// <summary>A resident weight's dtype conversions, keyed by weight then by target dtype name. A quantized or fp8
    /// weight is converted once and reused, instead of being converted again by every GEMM that reads it.
    ///
    /// <para>Only WEIGHTS are cached this way. An activation changes every step, so a keyed entry would never be hit
    /// and would grow without bound.</para></summary>
    /// <remarks>Each entry carries its own size: a conversion's byte count is not the weight's — that is the whole
    /// point of converting — so a backend that needs the size to free an allocation cannot recompute it from the
    /// tensor.</remarks>
    protected readonly Dictionary<Tensor, Dictionary<string, (TBuffer Buffer, long Bytes)>> WeightCasts = new(ReferenceEqualityComparer.Instance);

    /// <summary>Every buffer the cache owns. A backend frees its own op temporaries through
    /// <see cref="ReleaseIfNotCached"/>, which consults this so a temporary that has since been cached — an output
    /// bound to its tensor — is not freed out from under the tensor that now points at it.</summary>
    protected readonly HashSet<TBuffer> CachedBuffers = [];

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

    /// <summary>Allocates <paramref name="bytes"/> of device memory.</summary>
    protected abstract TBuffer AllocateDevice(long bytes);

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

    /// <summary>Called after a fresh upload that was not cached, for a backend that tracks transient buffers — to
    /// free them at its next flush, or to promote a tensor uploaded twice into a resident weight.</summary>
    protected virtual void OnTransientUploaded(TBuffer buffer, Tensor source) { }

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

        Activations[tensor] = (buffer, bytes);
        CachedBuffers.Add(buffer);

        tensor.SetGpuBinding(
            BindingKey,
            sync: () => SyncActivationToHost(tensor),
            dispose: () => ReleaseActivation(tensor));
    }

    /// <summary>Host code read the tensor: bring the value back and give up the device copy.</summary>
    private void SyncActivationToHost(Tensor tensor)
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
        if (WeightCasts.TryGetValue(weight, out Dictionary<string, (TBuffer Buffer, long Bytes)>? casts)
            && casts.TryGetValue(want.Name, out (TBuffer Buffer, long Bytes) hit))
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
        if (!WeightCasts.TryGetValue(weight, out Dictionary<string, (TBuffer Buffer, long Bytes)>? casts))
        {
            casts = new Dictionary<string, (TBuffer, long)>(StringComparer.Ordinal);
            WeightCasts[weight] = casts;
        }
        casts[want.Name] = (buffer, bytes);
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
        TBuffer buffer = AllocateDevice(bytes);
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
                CachedBuffers.Remove(buffer);
                ReleaseBuffer(buffer, ByteSize(weight));
            }
            // A conversion outlives nothing: its only purpose is to serve the weight that is going away.
            if (WeightCasts.Remove(weight, out Dictionary<string, (TBuffer Buffer, long Bytes)>? casts))
            {
                foreach ((TBuffer cast, long castBytes) in casts.Values)
                {
                    CachedBuffers.Remove(cast);
                    ReleaseBuffer(cast, castBytes);
                }
            }
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
            if (released.Add(buffer))
            {
                ReleaseBuffer(buffer, ByteSize(tensor));
            }
        }
        Weights.Clear();

        foreach (Dictionary<string, (TBuffer Buffer, long Bytes)> casts in WeightCasts.Values)
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
        (Tensor Tensor, long Bytes)[] candidates = [.. Activations
            .Where(entry => !Pinned.Contains(entry.Key))
            .Select(entry => (entry.Key, entry.Value.Bytes))
            .OrderByDescending(entry => entry.Bytes)];

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

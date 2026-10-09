using System.Runtime.InteropServices;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Moe.Residency;

namespace HartsyInference.Core.Backends;

/// <summary>Device-independent expert residency: in-flight dedup, pinning, a segmented LRU with per-layer frequency, and fence-gated eviction. A backend supplies the five transfer hooks.</summary>
/// <remarks><para>Replacement scans probation before the protected segment, then colder layers before hotter ones, then oldest use first;
/// pinned experts and the ones being requested are never victims. A prefetch also never evicts from the layers it targets or the layer last acquired;
/// <see cref="Acquire"/> does not protect its own layer, since a layer's unpinned experts must be replaceable by that layer's next request. Every hook runs under the cache lock.</para></remarks>
public abstract class ExpertCacheBase : IResidencyAwareExpertCache
{
    private const double ProtectedFraction = 0.8;
    private const long AgeEveryRequests = 4096;

    private readonly object _gate = new();

    // Scratch for the residency path, reused under _gate so a per-layer call does not allocate.
    private readonly List<ExpertKey> _scratchUnique = [];
    private readonly HashSet<ExpertKey> _scratchSeen = [];
    private readonly List<ExpertKey> _scratchResident = [];
    private readonly List<ExpertKey> _scratchAbsent = [];
    private readonly List<ExpertKey> _scratchPolicyKeys = [];
    private readonly Dictionary<ExpertLayerKey, ExpertBank> _banks = [];
    private readonly Dictionary<ExpertKey, ExpertCacheEntry> _entries = [];
    private readonly Dictionary<ExpertLayerKey, long> _layerFrequency = [];
    private readonly HashSet<ExpertLease> _live = [];
    private readonly Stack<ExpertFence> _freeFences = [];
    private long _tick;
    private long _requests;
    private long _residentBytes;
    private long _protectedBytes;
    private ExpertLayerKey? _currentLayer;
    private IAdaptiveResidencyPolicy? _residencyPolicy;
    private bool _disposed;
    private long _hits, _inFlightHits, _misses, _prefetches, _evictions, _bytesUploaded;

    /// <summary>Creates a cache of at most <paramref name="budgetBytes"/> resident expert bytes.</summary>
    protected ExpertCacheBase(long budgetBytes, IEnumerable<ExpertBank>? banks = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(budgetBytes);
        BudgetBytes = budgetBytes;
        if (banks is null) return;
        foreach (ExpertBank bank in banks) RegisterBank(bank);
    }

    /// <inheritdoc/>
    public long BudgetBytes { get; }

    /// <inheritdoc/>
    public ExpertCacheStats Stats
    {
        get
        {
            lock (_gate)
            {
                long pinnedBytes = 0;
                int pinned = 0;
                foreach (ExpertCacheEntry entry in _entries.Values)
                {
                    if (entry.PinCount == 0) continue;
                    pinnedBytes += entry.Bytes;
                    pinned++;
                }
                return new ExpertCacheStats(_hits, _inFlightHits, _misses, _prefetches, _evictions, _bytesUploaded,
                    _residentBytes, pinnedBytes, _entries.Count, pinned, BudgetBytes);
            }
        }
    }

    /// <summary>Starts copying every tensor of <paramref name="weights"/> to the device on the upload stream; returns a handle for <see cref="AwaitUpload"/>, or null when nothing is left to wait for. Must leave nothing registered if it throws.</summary>
    protected abstract object? BeginUpload(ExpertWeights weights);

    /// <summary>Orders the compute stream after an upload started by <see cref="BeginUpload"/>; consumes the handle.</summary>
    protected abstract void AwaitUpload(object pending);

    /// <summary>Makes an upload that failed to await safe to evict: no copy may still be writing the memory that is freed next.</summary>
    protected virtual void AbandonUpload(object pending) { }

    /// <summary>Frees the expert's device memory ordered after the compute stream's queued work; the upload must already be awaited.</summary>
    protected abstract void Evict(ExpertWeights weights);

    /// <summary>Records a marker on the compute stream after everything queued so far.</summary>
    protected abstract object RecordFence();

    /// <summary>True once the compute stream has passed the fence.</summary>
    protected abstract bool IsFenceDone(object fence);

    /// <summary>Blocks the host until the compute stream has passed the fence.</summary>
    protected abstract void WaitFence(object fence);

    /// <summary>Releases the fence's resources.</summary>
    protected abstract void DestroyFence(object fence);

    /// <summary>Waits out all in-flight work and returns pooled and staging memory; called once at dispose after every expert is evicted.</summary>
    protected abstract void Drain();

    /// <inheritdoc/>
    public void RegisterBank(ExpertBank bank)
    {
        ArgumentNullException.ThrowIfNull(bank);
        lock (_gate)
        {
            if (_banks.TryGetValue(bank.LayerKey, out ExpertBank? existing) && !ReferenceEquals(existing, bank))
                throw new InvalidOperationException($"Layer {bank.LayerKey} already has a different expert bank registered.");
            _banks[bank.LayerKey] = bank;
        }
    }

    /// <summary>
    /// Attaches an adaptive residency policy that chooses eviction victims among the evictable residents. Opt-in: without a policy
    /// the cache keeps its built-in order. Must be called once, before the first expert is requested or registered as resident.
    /// The policy is driven under the cache lock; the cache reports routed accesses (<see cref="Acquire"/>, <see cref="AcquireResident"/>),
    /// inserts and evictions, and falls back to the built-in order whenever the policy names no valid victim.
    /// </summary>
    public void AttachResidencyPolicy(IAdaptiveResidencyPolicy policy)
    {
        ArgumentNullException.ThrowIfNull(policy);
        lock (_gate)
        {
            ThrowIfDisposed();
            if (_residencyPolicy is not null) throw new InvalidOperationException("A residency policy is already attached.");
            if (_tick != 0 || _entries.Count != 0)
                throw new InvalidOperationException("A residency policy must be attached before the cache is used.");
            _residencyPolicy = policy;
        }
    }

    /// <inheritdoc/>
    public ExpertLease Acquire(ReadOnlySpan<ExpertKey> keys)
    {
        List<ExpertKey> unique = Distinct(keys);
        lock (_gate)
        {
            ThrowIfDisposed();
            HashSet<ExpertKey> requested = [.. unique];
            List<ExpertWeights> missing = [];
            HashSet<ExpertKey> missingKeys = [];
            long missingBytes = 0;
            int hits = 0, inFlight = 0;
            if (unique.Count > 0) _currentLayer = unique[0].LayerKey;
            foreach (ExpertKey key in unique)
            {
                _residencyPolicy?.NoteAccess(key);
                if (_entries.TryGetValue(key, out ExpertCacheEntry? entry))
                {
                    hits++;
                    if (entry.Pending is not null) inFlight++;
                    continue;
                }
                ExpertWeights weights = Resolve(key);
                missing.Add(weights);
                missingKeys.Add(key);
                missingBytes += weights.Bytes;
            }

            MakeRoom(missingBytes, requested, protectedLayers: null, mustSucceed: true);
            UploadMissing(missing, prefetch: false);

            ExpertLease lease = new();
            PinInto(lease, unique, missingKeys, hits, inFlight);
            return lease;
        }
    }

    /// <inheritdoc/>
    public int LookupResident(ReadOnlySpan<ExpertKey> keys, Span<bool> resident)
    {
        if (resident.Length < keys.Length) throw new ArgumentException("The residency mask must hold one entry per key.", nameof(resident));
        lock (_gate)
        {
            ThrowIfDisposed();
            int count = 0;
            for (int i = 0; i < keys.Length; i++)
            {
                bool present = _entries.ContainsKey(keys[i]);
                resident[i] = present;
                if (present) count++;
            }
            return count;
        }
    }

    /// <inheritdoc/>
    public ExpertLease AcquireResident(ReadOnlySpan<ExpertKey> keys, List<ExpertKey> misses)
    {
        ExpertLease lease = new();
        AcquireResident(keys, misses, lease);
        return lease;
    }

    /// <inheritdoc/>
    public void AcquireResident(ReadOnlySpan<ExpertKey> keys, List<ExpertKey> misses, ExpertLease lease)
    {
        ArgumentNullException.ThrowIfNull(misses);
        ArgumentNullException.ThrowIfNull(lease);
        lock (_gate)
        {
            ThrowIfDisposed();
            if (!lease.IsReleased) throw new InvalidOperationException("The lease is still pinned; release it before binding it again.");
            DistinctInto(keys, _scratchUnique, _scratchSeen);
            _scratchResident.Clear();
            _scratchAbsent.Clear();
            int inFlight = 0;
            foreach (ExpertKey key in _scratchUnique)
            {
                _residencyPolicy?.NoteAccess(key);
                if (_entries.TryGetValue(key, out ExpertCacheEntry? entry))
                {
                    _scratchResident.Add(key);
                    if (entry.Pending is not null) inFlight++;
                }
                else
                {
                    _scratchAbsent.Add(key);
                }
            }
            if (_scratchUnique.Count > 0) _currentLayer = _scratchUnique[0].LayerKey;
            // Only resident (or already uploading) experts are pinned; nothing is uploaded or resolved here.
            PinInto(lease, _scratchResident, missingKeys: null, _scratchResident.Count, inFlight);
            misses.AddRange(_scratchAbsent);
        }
    }

    /// <summary>Awaits in-flight uploads, then binds <paramref name="lease"/> to and pins every key in <paramref name="unique"/>. Caller holds the gate.</summary>
    private void PinInto(ExpertLease lease, List<ExpertKey> unique, HashSet<ExpertKey>? missingKeys, int hits, int inFlight)
    {
        // Awaiting can throw; do it before any pin or bind so a failure leaves the lease and the cache untouched.
        foreach (ExpertKey key in unique) AwaitPending(_entries[key]);
        ExpertWeights[] slots = lease.Bind(this, unique.Count);
        for (int i = 0; i < unique.Count; i++) slots[i] = _entries[unique[i]].Weights;
        foreach (ExpertKey key in unique)
        {
            ExpertCacheEntry entry = _entries[key];
            entry.PinCount++;
            Touch(entry, resident: missingKeys is null || !missingKeys.Contains(key));
        }
        _hits += hits;
        _inFlightHits += inFlight;
        AgeFrequencies(unique.Count);
        _live.Add(lease);
    }

    /// <inheritdoc/>
    public int Prefetch(ReadOnlySpan<ExpertKey> keys)
    {
        List<ExpertKey> unique = Distinct(keys);
        lock (_gate)
        {
            ThrowIfDisposed();
            HashSet<ExpertKey> requested = [.. unique];
            HashSet<ExpertLayerKey> protectedLayers = [.. unique.Select(static key => key.LayerKey)];
            if (_currentLayer is { } current) protectedLayers.Add(current);
            int started = 0;
            foreach (ExpertKey key in unique)
            {
                if (_entries.ContainsKey(key)) continue;
                ExpertWeights weights = Resolve(key);
                if (!MakeRoom(weights.Bytes, requested, protectedLayers, mustSucceed: false)) break;
                try
                {
                    UploadMissing([weights], prefetch: true);
                }
                catch (OutOfVramException)
                {
                    break;
                }
                started++;
            }
            return started;
        }
    }

    /// <inheritdoc/>
    public void Release(ExpertLease lease)
    {
        ArgumentNullException.ThrowIfNull(lease);
        if (!ReferenceEquals(lease.Owner, this)) throw new InvalidOperationException("Lease belongs to a different expert cache.");
        lock (_gate)
        {
            if (lease.IsReleased) return;
            if (_disposed)
            {
                lease.TryMarkReleased();
                _live.Remove(lease);
                lease.ClearWeights();
                return;
            }
            // Reclaim fences the device has already passed before adding one, so repeated release of the same experts reuses
            // holders from the pool instead of growing each entry's fence list and the device event count.
            for (int i = 0; i < lease.Count; i++)
            {
                if (_entries.TryGetValue(lease[i].Key, out ExpertCacheEntry? done)) PurgeFences(done);
            }
            // Recorded before the lease is marked or any pin drops, so a failure leaves it releasable again.
            ExpertFence fence = RentFence(RecordFence());
            lease.TryMarkReleased();
            _live.Remove(lease);
            for (int i = 0; i < lease.Count; i++)
            {
                if (!_entries.TryGetValue(lease[i].Key, out ExpertCacheEntry? entry)) continue;
                entry.PinCount--;
                fence.References++;
                entry.Fences.Add(fence);
            }
            if (fence.References == 0) ReturnFence(fence);
            lease.ClearWeights();
        }
    }

    /// <inheritdoc/>
    public long Trim(long targetResidentBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(targetResidentBytes);
        lock (_gate)
        {
            ThrowIfDisposed();
            long before = _residentBytes;
            if (before <= targetResidentBytes) return 0;
            if (_residencyPolicy is null)
            {
                foreach (ExpertCacheEntry victim in Candidates(requested: null, protectedLayers: null))
                {
                    if (_residentBytes <= targetResidentBytes) break;
                    EvictEntry(victim);
                }
                return before - _residentBytes;
            }
            List<ExpertCacheEntry> chosen = [];
            ChoosePolicyVictims(before - targetResidentBytes, requested: null, protectedLayers: null, chosen);
            foreach (ExpertCacheEntry victim in chosen) EvictEntry(victim);
            return before - _residentBytes;
        }
    }

    /// <summary>Evicts everything, drains the device and releases staging. Leases still open are closed.</summary>
    public void Dispose()
    {
        List<Exception>? failures = null;
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            foreach (ExpertLease lease in _live) lease.TryMarkReleased();
            _live.Clear();
            foreach (ExpertCacheEntry entry in _entries.Values.ToList())
            {
                entry.PinCount = 0;
                try { EvictEntry(entry, countEviction: false); }
                catch (Exception error) { (failures ??= []).Add(error); }
            }
            _entries.Clear();
            _residentBytes = 0;
            _protectedBytes = 0;
            try { Drain(); }
            catch (Exception error) { (failures ??= []).Add(error); }
        }
        if (failures is not null) throw new AggregateException("Expert cache teardown failed.", failures);
    }

    private static void DistinctInto(ReadOnlySpan<ExpertKey> keys, List<ExpertKey> unique, HashSet<ExpertKey> seen)
    {
        unique.Clear();
        seen.Clear();
        foreach (ExpertKey key in keys)
        {
            if (seen.Add(key)) unique.Add(key);
        }
    }

    private static List<ExpertKey> Distinct(ReadOnlySpan<ExpertKey> keys)
    {
        List<ExpertKey> unique = new(keys.Length);
        HashSet<ExpertKey> seen = new(keys.Length);
        foreach (ExpertKey key in keys)
        {
            if (seen.Add(key)) unique.Add(key);
        }
        return unique;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    private ExpertWeights Resolve(ExpertKey key)
    {
        if (!_banks.TryGetValue(key.LayerKey, out ExpertBank? bank))
            throw new InvalidOperationException($"No expert bank is registered for layer {key.LayerKey}.");
        return bank.Get(key.Expert);
    }

    private void AwaitPending(ExpertCacheEntry entry)
    {
        if (entry.Pending is null) return;
        object pending = entry.Pending;
        entry.Pending = null;
        try { AwaitUpload(pending); }
        catch
        {
            // The copy was never ordered before compute; a later hit must not see this entry as ready. Cleanup runs even if
            // AbandonUpload throws, and the original failure is the one reported.
            try { AbandonUpload(pending); }
            catch { /* the original failure is the one to report */ }
            try { RemoveEntry(entry, countEviction: false); }
            catch { /* the original failure is the one to report */ }
            throw;
        }
    }

    private void Touch(ExpertCacheEntry entry, bool resident)
    {
        entry.LastTick = ++_tick;
        _layerFrequency[entry.Key.LayerKey] = _layerFrequency.GetValueOrDefault(entry.Key.LayerKey) + 1;
        if (!resident || entry.Protected) return;
        entry.Protected = true;
        _protectedBytes += entry.Bytes;
        long limit = (long)(BudgetBytes * ProtectedFraction);
        while (_protectedBytes > limit)
        {
            ExpertCacheEntry? oldest = null;
            foreach (ExpertCacheEntry candidate in _entries.Values)
            {
                if (!candidate.Protected || ReferenceEquals(candidate, entry)) continue;
                if (oldest is null || candidate.LastTick < oldest.LastTick) oldest = candidate;
            }
            if (oldest is null) break;
            oldest.Protected = false;
            _protectedBytes -= oldest.Bytes;
        }
    }

    private void AgeFrequencies(int requests)
    {
        long before = _requests / AgeEveryRequests;
        _requests += requests;
        if (_requests / AgeEveryRequests == before) return;
        foreach (ExpertLayerKey layer in _layerFrequency.Keys.ToList()) _layerFrequency[layer] /= 2;
    }

    private IEnumerable<ExpertCacheEntry> Candidates(HashSet<ExpertKey>? requested, HashSet<ExpertLayerKey>? protectedLayers)
    {
        List<(ExpertCacheEntry Entry, bool Fenced)> found = [];
        foreach (ExpertCacheEntry entry in _entries.Values)
        {
            if (entry.PinCount > 0) continue;
            if (requested is not null && requested.Contains(entry.Key)) continue;
            if (protectedLayers is not null && protectedLayers.Contains(entry.Key.LayerKey)) continue;
            found.Add((entry, PurgeFences(entry)));
        }
        found.Sort((a, b) =>
        {
            int order = a.Fenced.CompareTo(b.Fenced);
            if (order != 0) return order;
            order = a.Entry.Protected.CompareTo(b.Entry.Protected);
            if (order != 0) return order;
            order = _layerFrequency.GetValueOrDefault(a.Entry.Key.LayerKey).CompareTo(_layerFrequency.GetValueOrDefault(b.Entry.Key.LayerKey));
            return order != 0 ? order : a.Entry.LastTick.CompareTo(b.Entry.LastTick);
        });
        return found.Select(static item => item.Entry);
    }

    /// <summary>Drops the fences that have passed; returns whether any are still pending.</summary>
    private bool PurgeFences(ExpertCacheEntry entry)
    {
        for (int i = entry.Fences.Count - 1; i >= 0; i--)
        {
            ExpertFence fence = entry.Fences[i];
            if (!IsFenceDone(fence.Handle)) continue;
            entry.Fences.RemoveAt(i);
            ReleaseFenceReference(fence);
        }
        return entry.Fences.Count > 0;
    }

    private void ReleaseFenceReference(ExpertFence fence)
    {
        if (--fence.References == 0) ReturnFence(fence);
    }

    private ExpertFence RentFence(object handle)
    {
        ExpertFence fence = _freeFences.Count > 0 ? _freeFences.Pop() : new ExpertFence();
        fence.Handle = handle;
        fence.References = 0;
        return fence;
    }

    /// <summary>Destroys a fence that no entry references any more, and keeps its holder for the next lease.</summary>
    private void ReturnFence(ExpertFence fence)
    {
        DestroyFence(fence.Handle);
        fence.Handle = null!;
        _freeFences.Push(fence);
    }

    private bool MakeRoom(long incomingBytes, HashSet<ExpertKey> requested, HashSet<ExpertLayerKey>? protectedLayers, bool mustSucceed)
    {
        long over = _residentBytes + incomingBytes - BudgetBytes;
        if (over <= 0) return true;
        List<ExpertCacheEntry> victims = [];
        long freed = 0;
        if (_residencyPolicy is null)
        {
            foreach (ExpertCacheEntry candidate in Candidates(requested, protectedLayers))
            {
                victims.Add(candidate);
                freed += candidate.Bytes;
                if (freed >= over) break;
            }
        }
        else
        {
            freed = ChoosePolicyVictims(over, requested, protectedLayers, victims);
        }
        if (freed < over)
        {
            if (!mustSucceed) return false;
            throw new OutOfVramException(
                $"Expert cache cannot fit {incomingBytes >> 20} MB more: budget {BudgetBytes >> 20} MB, "
                + $"{_residentBytes >> 20} MB resident, only {freed >> 20} MB evictable (pinned and current-layer experts stay).");
        }
        foreach (ExpertCacheEntry victim in victims) EvictEntry(victim);
        return true;
    }

    /// <summary>
    /// Picks victims one at a time until <paramref name="over"/> bytes are covered. Each pick asks the policy among the resident,
    /// unpinned candidates that are not loading. A pick the policy does not name, or names outside that set, falls back to the
    /// first remaining candidate in the built-in order. That fallback can be an entry still loading, exactly as it can with no policy.
    /// Cost: each pick scans the remaining candidates, so a pass that evicts k of n entries is O(n·k). The cache is small enough
    /// that this has not mattered; revisit before the cache holds thousands of experts.
    /// Returns the bytes covered. Caller holds the gate.
    /// </summary>
    private long ChoosePolicyVictims(
        long over, HashSet<ExpertKey>? requested, HashSet<ExpertLayerKey>? protectedLayers, List<ExpertCacheEntry> chosen)
    {
        List<ExpertCacheEntry> remaining = Candidates(requested, protectedLayers).ToList();
        long freed = 0;
        while (freed < over && remaining.Count > 0)
        {
            ExpertCacheEntry victim = PickPolicyVictim(remaining);
            remaining.Remove(victim);
            chosen.Add(victim);
            freed += victim.Bytes;
        }
        return freed;
    }

    /// <summary>Asks the policy for one victim among <paramref name="remaining"/>; never returns an entry outside that list.</summary>
    private ExpertCacheEntry PickPolicyVictim(List<ExpertCacheEntry> remaining)
    {
        ExpertCacheEntry fallback = remaining[0];
        _scratchPolicyKeys.Clear();
        foreach (ExpertCacheEntry candidate in remaining)
        {
            if (candidate.Pending is null) _scratchPolicyKeys.Add(candidate.Key);
        }
        if (_scratchPolicyKeys.Count == 0) return fallback;

        ExpertKey named = _residencyPolicy!.ChooseVictim(CollectionsMarshal.AsSpan(_scratchPolicyKeys));
        foreach (ExpertCacheEntry candidate in remaining)
        {
            if (candidate.Key != named) continue;
            if (candidate.PinCount == 0 && candidate.Pending is null) return candidate;
            break;
        }
        return fallback;
    }

    private void UploadMissing(List<ExpertWeights> missing, bool prefetch)
    {
        List<ExpertCacheEntry> created = [];
        try
        {
            foreach (ExpertWeights weights in missing)
            {
                object? pending = BeginUpload(weights);
                ExpertCacheEntry entry = new(weights, pending) { LastTick = ++_tick };
                _entries[weights.Key] = entry;
                _residencyPolicy?.NoteInsert(weights.Key);
                _residentBytes += weights.Bytes;
                created.Add(entry);
            }
        }
        catch
        {
            foreach (ExpertCacheEntry entry in created)
            {
                try { EvictEntry(entry, countEviction: false); }
                catch { /* the original failure is the one to report */ }
            }
            throw;
        }
        _bytesUploaded += created.Sum(static entry => entry.Bytes);
        if (prefetch) _prefetches += created.Count;
        else _misses += created.Count;
    }

    private void EvictEntry(ExpertCacheEntry entry, bool countEviction = true)
    {
        AwaitPending(entry);
        RemoveEntry(entry, countEviction);
    }

    private void RemoveEntry(ExpertCacheEntry entry, bool countEviction)
    {
        foreach (ExpertFence fence in entry.Fences.ToList())
        {
            if (!IsFenceDone(fence.Handle)) WaitFence(fence.Handle);
            ReleaseFenceReference(fence);
        }
        entry.Fences.Clear();
        Evict(entry.Weights);
        _entries.Remove(entry.Key);
        _residencyPolicy?.NoteEvict(entry.Key);
        _residentBytes -= entry.Bytes;
        if (entry.Protected) _protectedBytes -= entry.Bytes;
        if (countEviction) _evictions++;
    }
}

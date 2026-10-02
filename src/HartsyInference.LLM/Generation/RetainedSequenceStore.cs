namespace HartsyInference.LLM.Generation;

/// <summary>Keyed store of <see cref="RetainedSequence"/>s held warm for prefix-KV reuse across generation calls
/// that share a key (one phone call's conversation, one chat session, …). Bounded by entry count AND total bytes,
/// least-recently-used eviction; disposes whatever it evicts or is disposed with.</summary>
/// <remarks>Checkout/checkin, not get/set: <see cref="Checkout"/> REMOVES the entry, so a second concurrent caller
/// on the same key finds nothing and falls back to an uncached generation (the busy-key rule) instead of two
/// callers mutating one <see cref="ISequenceState"/> at once. <see cref="CheckIn"/> puts an entry back (or stores a
/// new one) and is also where bounds are enforced. A key with nothing stored is indistinguishable from one that is
/// merely checked out right now — both read as a miss — which is intentional: the caller treats either the same
/// way (create a <see cref="RetainedSequence"/> and generate uncached).</remarks>
public sealed class RetainedSequenceStore : IDisposable
{
    private readonly object _gate = new();
    private readonly Dictionary<string, RetainedSequence> _entries = new(StringComparer.Ordinal);
    private readonly List<string> _lru = []; // index 0 = least recently used, end = most recently used
    private readonly int _maxEntries;
    private readonly long _maxBytes;
    private long _bytes;
    private bool _disposed;

    /// <summary>Bounds the store to at most <paramref name="maxEntries"/> retained sequences and
    /// <paramref name="maxBytes"/> of their combined <see cref="RetainedSequence.Bytes"/>.</summary>
    public RetainedSequenceStore(int maxEntries, long maxBytes)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxEntries, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxBytes, 1);
        _maxEntries = maxEntries;
        _maxBytes = maxBytes;
    }

    /// <summary>Entries currently held (for observability/tests — not load-bearing).</summary>
    public int Count { get { lock (_gate) { return _entries.Count; } } }

    /// <summary>Combined <see cref="RetainedSequence.Bytes"/> of every entry currently held.</summary>
    public long BytesUsed { get { lock (_gate) { return _bytes; } } }

    /// <summary>Removes and returns the entry for <paramref name="key"/>; null when nothing is stored under it OR
    /// another caller has it checked out right now. The caller owns the returned instance exclusively until it
    /// calls <see cref="CheckIn"/> (or disposes it directly to drop the entry for good).</summary>
    public RetainedSequence? Checkout(string key)
    {
        ArgumentException.ThrowIfNullOrEmpty(key);
        lock (_gate)
        {
            if (_disposed || !_entries.Remove(key, out RetainedSequence? seq))
            {
                return null;
            }
            _lru.Remove(key);
            _bytes -= seq.Bytes;
            return seq;
        }
    }

    /// <summary>Stores <paramref name="sequence"/> under <paramref name="key"/>, evicting least-recently-used
    /// entries first until the store fits the configured count/byte bounds. Disposes every entry it evicts. A
    /// sequence heavier than the whole byte budget, or one holding nothing, is disposed instead of stored, and
    /// leaves the other entries alone.</summary>
    public void CheckIn(string key, RetainedSequence sequence)
    {
        ArgumentException.ThrowIfNullOrEmpty(key);
        ArgumentNullException.ThrowIfNull(sequence);
        lock (_gate)
        {
            if (_disposed || sequence.Cache is null || sequence.Bytes > _maxBytes)
            {
                sequence.Dispose();
                return;
            }
            // A stale same-key entry can only arrive here if some other path stored one for this key while this
            // sequence was checked out (not reachable via TextService's own checkout/checkin pairing today, but
            // the store does not assume a single caller discipline it cannot itself enforce).
            if (_entries.Remove(key, out RetainedSequence? stale) && !ReferenceEquals(stale, sequence))
            {
                _lru.Remove(key);
                _bytes -= stale.Bytes;
                stale.Dispose();
            }
            while (_lru.Count > 0 && (_entries.Count >= _maxEntries || _bytes + sequence.Bytes > _maxBytes))
            {
                string evictKey = _lru[0];
                _lru.RemoveAt(0);
                if (_entries.Remove(evictKey, out RetainedSequence? evicted))
                {
                    _bytes -= evicted.Bytes;
                    evicted.Dispose();
                }
            }
            _entries[key] = sequence;
            _lru.Add(key);
            _bytes += sequence.Bytes;
        }
    }

    /// <summary>Disposes every retained entry and empties the store. Idempotent; a <see cref="CheckIn"/> after
    /// disposal disposes its argument instead of storing it.</summary>
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            foreach (RetainedSequence seq in _entries.Values)
            {
                seq.Dispose();
            }
            _entries.Clear();
            _lru.Clear();
            _bytes = 0;
        }
    }
}

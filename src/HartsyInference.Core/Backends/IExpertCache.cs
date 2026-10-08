namespace HartsyInference.Core.Backends;

/// <summary>Keeps routed experts resident on a device within a byte budget, whole expert at a time.</summary>
public interface IExpertCache : IDisposable
{
    /// <summary>The byte budget the resident experts stay within.</summary>
    long BudgetBytes { get; }

    /// <summary>Current counters.</summary>
    ExpertCacheStats Stats { get; }

    /// <summary>Makes a bank's experts resolvable by key; one bank per layer.</summary>
    void RegisterBank(ExpertBank bank);

    /// <summary>Pins these experts resident: duplicates collapse, misses upload once, and the compute stream is ordered after the uploads.</summary>
    /// <exception cref="Exceptions.OutOfVramException">The pinned set cannot fit the budget.</exception>
    ExpertLease Acquire(ReadOnlySpan<ExpertKey> keys);

    /// <summary>
    /// Reports which keys are resident or already uploading, without changing any state. Returns how many are.
    /// <paramref name="resident"/> must hold one entry per key.
    /// </summary>
    int LookupResident(ReadOnlySpan<ExpertKey> keys, Span<bool> resident);

    /// <summary>
    /// Pins only the experts already resident or uploading, and never uploads or resolves anything. Keys that are not resident
    /// are appended to <paramref name="misses"/> (only when the call succeeds), so the caller can run them elsewhere.
    /// </summary>
    ExpertLease AcquireResident(ReadOnlySpan<ExpertKey> keys, List<ExpertKey> misses);

    /// <summary>Starts uploading experts nobody has asked for yet, best effort within the budget; returns how many uploads started.</summary>
    int Prefetch(ReadOnlySpan<ExpertKey> keys);

    /// <summary>Unpins a lease; its experts become evictable once work already queued on the compute stream has passed.</summary>
    void Release(ExpertLease lease);

    /// <summary>Evicts unpinned experts, least valuable first, until at most <paramref name="targetResidentBytes"/> remain; returns the bytes freed.</summary>
    long Trim(long targetResidentBytes);
}

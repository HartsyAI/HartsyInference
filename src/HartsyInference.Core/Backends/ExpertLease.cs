using System.Collections;

namespace HartsyInference.Core.Backends;

/// <summary>
/// Experts pinned resident by <see cref="IExpertCache.Acquire"/> or <see cref="IResidencyAwareExpertCache.AcquireResident(ReadOnlySpan{ExpertKey}, List{ExpertKey}, ExpertLease)"/>
/// until released; disposing releases them. A lease created with the parameterless constructor is reusable: the planner binds it
/// to a new pin once the previous one has been released, so a steady-state layer loop allocates no lease.
/// </summary>
public sealed class ExpertLease : IDisposable, IReadOnlyList<ExpertWeights>
{
    private ExpertWeights[] _weights;
    private int _count;
    private int _released = 1;

    /// <summary>Creates an unbound lease, ready for <see cref="IResidencyAwareExpertCache.AcquireResident(ReadOnlySpan{ExpertKey}, List{ExpertKey}, ExpertLease)"/>.</summary>
    public ExpertLease()
    {
        _weights = [];
    }

    /// <summary>The cache that pinned this lease; null until the lease is first bound.</summary>
    internal ExpertCacheBase? Owner { get; private set; }

    /// <summary>The leased experts, deduplicated, in request order.</summary>
    public IReadOnlyList<ExpertWeights> Weights => this;

    /// <summary>Number of leased experts.</summary>
    public int Count => _count;

    /// <summary>True once the lease has been released.</summary>
    public bool IsReleased => Volatile.Read(ref _released) != 0;

    /// <inheritdoc/>
    public ExpertWeights this[int index]
    {
        get
        {
            if ((uint)index >= (uint)_count) throw new ArgumentOutOfRangeException(nameof(index), index, $"The lease holds {_count} experts.");
            return _weights[index];
        }
    }

    /// <summary>The leased weights of <paramref name="key"/>.</summary>
    public ExpertWeights Get(ExpertKey key)
    {
        for (int i = 0; i < _count; i++)
        {
            if (_weights[i].Key == key) return _weights[i];
        }
        throw new KeyNotFoundException($"{key} is not part of this lease.");
    }

    /// <summary>Releases the lease; safe to call more than once. An unbound lease has nothing to release.</summary>
    public void Dispose() => Owner?.Release(this);

    /// <inheritdoc/>
    public IEnumerator<ExpertWeights> GetEnumerator()
    {
        for (int i = 0; i < _count; i++) yield return _weights[i];
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>Binds the lease to <paramref name="owner"/> with room for <paramref name="count"/> experts; the caller fills the slots.</summary>
    /// <remarks>Caller holds the owner's gate. Grows the slot array only when a larger pin than any before arrives.</remarks>
    internal ExpertWeights[] Bind(ExpertCacheBase owner, int count)
    {
        if (_weights.Length < count) _weights = new ExpertWeights[count];
        Owner = owner;
        _count = count;
        Volatile.Write(ref _released, 0);
        return _weights;
    }

    /// <summary>Drops the weights so a released lease does not keep them reachable.</summary>
    internal void ClearWeights()
    {
        Array.Clear(_weights, 0, _count);
        _count = 0;
    }

    internal bool TryMarkReleased() => Interlocked.Exchange(ref _released, 1) == 0;
}

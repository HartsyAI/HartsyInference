namespace HartsyInference.Core.Backends;

/// <summary>Experts pinned resident by <see cref="IExpertCache.Acquire"/> until released; disposing releases them.</summary>
public sealed class ExpertLease : IDisposable
{
    private readonly ExpertWeights[] _weights;
    private int _released;

    internal ExpertLease(ExpertCacheBase owner, ExpertWeights[] weights)
    {
        Owner = owner;
        _weights = weights;
    }

    internal ExpertCacheBase Owner { get; }

    /// <summary>The leased experts, deduplicated, in request order.</summary>
    public IReadOnlyList<ExpertWeights> Weights => _weights;

    /// <summary>True once the lease has been released.</summary>
    public bool IsReleased => Volatile.Read(ref _released) != 0;

    /// <summary>The leased weights of <paramref name="key"/>.</summary>
    public ExpertWeights Get(ExpertKey key)
    {
        foreach (ExpertWeights weights in _weights)
        {
            if (weights.Key == key) return weights;
        }
        throw new KeyNotFoundException($"{key} is not part of this lease.");
    }

    internal bool TryMarkReleased() => Interlocked.Exchange(ref _released, 1) == 0;

    /// <summary>Releases the lease; safe to call more than once.</summary>
    public void Dispose() => Owner.Release(this);
}

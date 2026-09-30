namespace HartsyInference.Core.Backends;

/// <summary>Bookkeeping for one resident expert.</summary>
internal sealed class ExpertCacheEntry
{
    public ExpertCacheEntry(ExpertWeights weights, object? pending)
    {
        Weights = weights;
        Pending = pending;
    }

    public ExpertWeights Weights { get; }

    public ExpertKey Key => Weights.Key;

    public long Bytes => Weights.Bytes;

    /// <summary>Backend upload handle the compute stream has not waited on yet.</summary>
    public object? Pending { get; set; }

    public int PinCount { get; set; }

    public bool Protected { get; set; }

    public long LastTick { get; set; }

    /// <summary>Fences of released leases that may still be reading.</summary>
    public List<ExpertFence> Fences { get; } = [];
}

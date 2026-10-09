namespace HartsyInference.Core.Moe.Residency;

/// <summary>Outcome of replaying one trace through one policy at one slot budget.</summary>
public readonly record struct ReplayResult(
    string Policy,
    int SlotBudget,
    long Accesses,
    long Hits,
    long Misses,
    long Evictions,
    long BytesMoved,
    ulong Digest)
{
    /// <summary>Hits divided by accesses; zero for an empty trace.</summary>
    public double HitRate => Accesses == 0 ? 0.0 : (double)Hits / Accesses;
}

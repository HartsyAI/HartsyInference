namespace HartsyInference.Core.Backends.Attention;

/// <summary>A bounded host-only evidence buffer allocated before inference; overflow invalidates the capture.</summary>
public sealed class AttentionEvidenceSession : IDisposable
{
    private readonly AttentionEvidenceEvent[] _events;
    private readonly int[] _published;
    private long _attempted;
    private int _disposed;

    internal AttentionEvidenceSession(int capacity)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        _events = new AttentionEvidenceEvent[capacity];
        _published = new int[capacity];
    }

    /// <summary>Number of attempted observations, including any that did not fit.</summary>
    public long Attempted => Interlocked.Read(ref _attempted);

    /// <summary>Observations that exceeded the fixed host capacity.</summary>
    public long Dropped => Math.Max(0, Attempted - _events.Length);

    /// <summary>Copies completed observations outside inference; callers must first finish all observed work.</summary>
    public AttentionEvidenceEvent[] Snapshot()
    {
        long attempted = Attempted;
        if (attempted > _events.Length)
            throw new InvalidOperationException("Attention evidence overflowed; the capture is incomplete.");
        AttentionEvidenceEvent[] result = new AttentionEvidenceEvent[(int)attempted];
        for (int i = 0; i < result.Length; i++)
        {
            if (Volatile.Read(ref _published[i]) == 0)
                throw new InvalidOperationException("Attention evidence still has an in-flight writer.");
            result[i] = _events[i];
        }
        return result;
    }

    internal void Record(in AttentionEvidenceEvent observation)
    {
        long index = Interlocked.Increment(ref _attempted) - 1;
        if (index >= _events.Length) return;
        _events[(int)index] = observation;
        Volatile.Write(ref _published[(int)index], 1);
    }

    /// <summary>Disables collection without waiting, synchronizing a device, or formatting the capture.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0) AttentionEvidence.End(this);
    }
}

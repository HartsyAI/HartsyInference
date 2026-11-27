namespace HartsyInference.Engine.Worlds;

/// <summary>Counts the resources acquired and released by the phases that report to it. Run phases report here, so
/// <see cref="Outstanding"/> returns to zero once every run has finished; a non-zero value after a run is a leak.</summary>
public sealed class WorldPhaseLedger
{
    private long _acquired;
    private long _released;

    /// <summary>Resources acquired by the reporting phases.</summary>
    public long Acquired => Interlocked.Read(ref _acquired);

    /// <summary>Resources released by the reporting phases.</summary>
    public long Released => Interlocked.Read(ref _released);

    /// <summary>Resources acquired but not yet released.</summary>
    public long Outstanding => Acquired - Released;

    internal void RecordAcquire() => Interlocked.Increment(ref _acquired);

    internal void RecordRelease() => Interlocked.Increment(ref _released);
}

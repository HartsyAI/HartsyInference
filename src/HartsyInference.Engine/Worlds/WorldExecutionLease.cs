namespace HartsyInference.Engine.Worlds;

/// <summary>The engine's execution lease for batch world generation. A run holds it from sampling through the eviction of its
/// phase, and later runs queue behind it. <see cref="SemaphoreSlim.WaitAsync(CancellationToken)"/> removes a cancelled waiter
/// from the queue itself, so a cancelled request neither holds the lease nor blocks the runs behind it.</summary>
internal sealed class WorldExecutionLease
{
    private readonly SemaphoreSlim _gate = new(1, 1);
    private int _queued;

    /// <summary>Runs waiting for the lease, not counting the holder.</summary>
    public int Queued => Volatile.Read(ref _queued);

    /// <summary>Whether a run holds the lease right now.</summary>
    public bool IsHeld => _gate.CurrentCount == 0;

    /// <summary>Waits for the lease. Dispose the returned token to release it; a wait that <paramref name="cancel"/> ends throws
    /// <see cref="OperationCanceledException"/> and leaves the lease untouched.</summary>
    public async Task<IDisposable> AcquireAsync(CancellationToken cancel)
    {
        Interlocked.Increment(ref _queued);
        try
        {
            await _gate.WaitAsync(cancel).ConfigureAwait(false);
        }
        finally
        {
            Interlocked.Decrement(ref _queued);
        }
        return new Holder(_gate);
    }

    private sealed class Holder(SemaphoreSlim gate) : IDisposable
    {
        private SemaphoreSlim? _gate = gate;

        public void Dispose() => Interlocked.Exchange(ref _gate, null)?.Release();
    }
}

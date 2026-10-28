namespace HartsyInference.Engine.Audio;

/// <summary>The lifetime both runner leases share: one pin on the runner's cache key, a call lock, and revocation when
/// the engine releases its audio models. Runner calls hold the lock, so a revocation or a Dispose waits for the call in
/// flight instead of letting the runner be disposed underneath it.</summary>
internal abstract class AudioRunnerLease : IDisposable
{
    private const string RevokedReason =
        "The engine released its audio models (Dispose, FreeMemory, or a backend or placement change); open a new lease.";

    private readonly AudioRuntime _runtime;
    private readonly IDisposable _pin;
    private readonly CancellationTokenSource _closing = new();
    private string? _closedReason;
    private int _pinReleased;

    /// <summary>Pins <paramref name="key"/> in <paramref name="cache"/> for the lease's lifetime.</summary>
    protected AudioRunnerLease(AudioRuntime runtime, IAudioRunnerCache cache, string key)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(cache);
        _runtime = runtime;
        _pin = cache.Pin(key);
        ModelKey = new AudioJob(cache, key).ModelKey;
    }

    /// <summary>The prefixed model key, e.g. <c>tts:hexgrad/Kokoro-82M</c>, for messages.</summary>
    internal string ModelKey { get; }

    /// <summary>Held around every runner call.</summary>
    protected object CallLock { get; } = new();

    /// <summary>Cancelled when the lease closes, so a call made of several runner steps stops between them instead of running on against a released runner.</summary>
    protected CancellationToken Closing => _closing.Token;

    /// <summary>Closes the lease for an engine release, waiting up to <paramref name="wait"/> for a call in flight.
    /// Returns false when the wait ran out and the lease was closed with that call still running.</summary>
    internal bool Revoke(TimeSpan wait)
    {
        bool held = Monitor.TryEnter(CallLock, wait);
        try
        {
            Close(RevokedReason);
        }
        finally
        {
            if (held)
            {
                Monitor.Exit(CallLock);
            }
        }
        return held;
    }

    /// <summary>Throws once the lease is disposed or revoked; authoritative only under <see cref="CallLock"/>.</summary>
    protected void ThrowIfClosed()
    {
        string? reason = Volatile.Read(ref _closedReason);
        if (reason is not null)
        {
            throw new ObjectDisposedException(GetType().Name, reason);
        }
    }

    /// <summary>Records why the lease closed and cancels <see cref="Closing"/>; the first reason wins. Takes no lock, so a Dispose can stop a long call before it waits for it.</summary>
    private void Signal(string reason)
    {
        if (Interlocked.CompareExchange(ref _closedReason, reason, null) is null)
        {
            _closing.Cancel();
        }
    }

    /// <summary>Signals the close and releases the pin once, so a revocation racing a Dispose does not release it twice.</summary>
    private void Close(string reason)
    {
        Signal(reason);
        if (Interlocked.Exchange(ref _pinReleased, 1) == 0)
        {
            _pin.Dispose();
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        string reason = $"The lease on '{ModelKey}' was disposed.";
        Signal(reason);
        lock (CallLock)
        {
            Close(reason);
        }
        _runtime.Unregister(this);
    }
}

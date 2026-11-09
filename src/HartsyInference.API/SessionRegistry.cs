using System.Collections.Concurrent;

namespace HartsyInference.API;

/// <summary>Tracks HTTP-visible long-lived sessions by a server-assigned id, so a client can find one again on a later request (an action or
/// frame POST, a reconnecting stream) and the server can reclaim it when the client vanishes. Sessions idle for longer than the timeout are
/// closed by a periodic sweep.</summary>
/// <typeparam name="TSession">The session type; <paramref name="close"/> releases one.</typeparam>
public class SessionRegistry<TSession> : IDisposable
    where TSession : class
{
    private readonly ConcurrentDictionary<string, Entry> _sessions = new(StringComparer.Ordinal);
    private readonly Func<TSession, ValueTask> _close;
    private readonly TimeSpan _idleTimeout;
    private readonly Timer _sweepTimer;
    private int _disposed;

    /// <summary>Creates a registry that closes sessions idle for longer than <paramref name="idleTimeout"/>, checked on that same interval.</summary>
    /// <param name="close">Releases a session the registry drops (closed, evicted, or still open at shutdown).</param>
    public SessionRegistry(TimeSpan idleTimeout, Func<TSession, ValueTask> close)
    {
        ArgumentNullException.ThrowIfNull(close);
        if (idleTimeout <= TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(idleTimeout), "The idle timeout must be positive.");
        _idleTimeout = idleTimeout;
        _close = close;
        _sweepTimer = new Timer(_ => Sweep(), null, idleTimeout, idleTimeout);
    }

    /// <summary>How many sessions are open.</summary>
    public int Count => _sessions.Count;

    /// <summary>Registers a newly opened session and returns its id.</summary>
    public string Register(TSession session)
    {
        ArgumentNullException.ThrowIfNull(session);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        string id = Guid.NewGuid().ToString("N");
        _sessions[id] = new Entry(session);
        return id;
    }

    /// <summary>Looks up a session by id and marks it active, so it does not idle out mid-use. Null when the id is unknown, closed or evicted.</summary>
    public TSession? Get(string id)
    {
        if (!_sessions.TryGetValue(id, out Entry? entry))
            return null;
        entry.Touch();
        return entry.Session;
    }

    /// <summary>Marks a session active without returning it, for a long-running reader (a stream) that should keep it from idling out.</summary>
    public void Touch(string id)
    {
        if (_sessions.TryGetValue(id, out Entry? entry)) entry.Touch();
    }

    /// <summary>Removes and releases the session, if present. Returns whether one was found.</summary>
    public async ValueTask<bool> CloseAsync(string id)
    {
        if (!_sessions.TryRemove(id, out Entry? entry))
            return false;
        await _close(entry.Session).ConfigureAwait(false);
        return true;
    }

    /// <summary>As <see cref="CloseAsync"/>, for sessions whose release is synchronous.</summary>
    public bool Close(string id) => CloseAsync(id).AsTask().GetAwaiter().GetResult();

    /// <summary>Closes sessions idle past the timeout. Runs on the sweep timer; internal so tests can drive it.</summary>
    internal void Sweep(DateTime? now = null)
    {
        DateTime cutoff = (now ?? DateTime.UtcNow) - _idleTimeout;
        foreach ((string id, Entry entry) in _sessions)
        {
            if (entry.LastActivity < cutoff && _sessions.TryRemove(id, out Entry? removed))
                ReleaseQuietly(removed.Session);
        }
    }

    /// <summary>Stops the sweep timer and releases every session still open, at host shutdown (the registry is a singleton).</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _sweepTimer.Dispose();
        foreach (Entry entry in _sessions.Values)
            ReleaseQuietly(entry.Session);
        _sessions.Clear();
        GC.SuppressFinalize(this);
    }

    private void ReleaseQuietly(TSession session)
    {
        try
        {
            _close(session).AsTask().GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            Core.Logging.Logs.Warning($"[SessionRegistry] Releasing a {typeof(TSession).Name} failed: {ex.Message}");
        }
    }

    private sealed class Entry(TSession session)
    {
        private long _lastActivityTicks = DateTime.UtcNow.Ticks;

        public TSession Session { get; } = session;

        public DateTime LastActivity => new(Interlocked.Read(ref _lastActivityTicks), DateTimeKind.Utc);

        public void Touch() => Interlocked.Exchange(ref _lastActivityTicks, DateTime.UtcNow.Ticks);
    }
}

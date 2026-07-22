using System.Collections.Concurrent;
using HartsyInference.Engine.Requests;

namespace HartsyInference.API;

/// <summary>Tracks HTTP-visible interactive-world sessions by a server-assigned id. Genuinely new infrastructure —
/// the CLI never needs this because it holds one <see cref="IWorldSession"/> for the life of a single process
/// invocation; an HTTP server can have many concurrent sessions across many clients and must be able to find one
/// again on a later request (an action POST, a reconnecting stream) and reclaim it if the client vanishes.</summary>
public sealed class WorldSessionRegistry : IDisposable
{
    private readonly ConcurrentDictionary<string, Entry> _sessions = new(StringComparer.Ordinal);
    private readonly TimeSpan _idleTimeout;
    private readonly Timer _sweepTimer;
    private int _disposed;

    /// <summary>Creates a registry that evicts sessions idle for longer than <paramref name="idleTimeout"/>,
    /// checked on that same interval.</summary>
    public WorldSessionRegistry(TimeSpan idleTimeout)
    {
        _idleTimeout = idleTimeout;
        _sweepTimer = new Timer(_ => Sweep(), null, idleTimeout, idleTimeout);
    }

    /// <summary>Registers a newly-opened session and returns its id.</summary>
    public string Register(IWorldSession session)
    {
        string id = Guid.NewGuid().ToString("N");
        _sessions[id] = new Entry(session);
        return id;
    }

    /// <summary>Looks up a session by id, touching its last-activity time so it doesn't idle-time-out mid-use.
    /// Null when the id is unknown or was already evicted/closed.</summary>
    public IWorldSession? Get(string id)
    {
        if (!_sessions.TryGetValue(id, out Entry? entry))
            return null;
        entry.Touch();
        return entry.Session;
    }

    /// <summary>Removes and disposes the session, if present. Returns whether one was actually found.</summary>
    public bool Close(string id)
    {
        if (!_sessions.TryRemove(id, out Entry? entry))
            return false;
        entry.Session.Dispose();
        return true;
    }

    private void Sweep()
    {
        DateTime cutoff = DateTime.UtcNow - _idleTimeout;
        foreach ((string id, Entry entry) in _sessions)
        {
            if (entry.LastActivity < cutoff && _sessions.TryRemove(id, out Entry? removed))
                removed.Session.Dispose();
        }
    }

    /// <summary>Stops the sweep timer and disposes every still-open session — called by the DI container at host
    /// shutdown since this is registered as a singleton.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _sweepTimer.Dispose();
        foreach (Entry entry in _sessions.Values)
            entry.Session.Dispose();
        _sessions.Clear();
    }

    private sealed class Entry(IWorldSession session)
    {
        public IWorldSession Session { get; } = session;
        public DateTime LastActivity { get; private set; } = DateTime.UtcNow;
        public void Touch() => LastActivity = DateTime.UtcNow;
    }
}

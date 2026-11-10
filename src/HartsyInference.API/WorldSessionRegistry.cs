using System.Runtime.CompilerServices;
using HartsyInference.Engine.Requests;

namespace HartsyInference.API;

/// <summary>Open interactive-world sessions by id (see <see cref="SessionRegistry{TSession}"/>). The CLI never needs this, since it holds one
/// <see cref="IWorldSession"/> for one process; an HTTP server serves many clients and finds a session again on each request. Each session
/// also remembers the frame rate it was opened at, which stamps every frame the stream sends.</summary>
public sealed class WorldSessionRegistry(TimeSpan idleTimeout)
    : SessionRegistry<IWorldSession>(idleTimeout, static session =>
    {
        session.Dispose();
        return ValueTask.CompletedTask;
    })
{
    // Keyed by the session object, so a rate leaves with its session however the registry drops it (closed, idled out, or at shutdown).
    private readonly ConditionalWeakTable<IWorldSession, StrongBox<int>> _frameRates = new();

    /// <summary>Registers a newly opened session at its frame rate and returns its id.</summary>
    public string Register(IWorldSession session, int fps)
    {
        ArgumentNullException.ThrowIfNull(session);
        _frameRates.AddOrUpdate(session, new StrongBox<int>(fps));
        return Register(session);
    }

    /// <summary>The frame rate a session was opened at; null when the id is unknown or was closed or evicted. Marks the session active, like
    /// <see cref="SessionRegistry{TSession}.Get"/>.</summary>
    public int? FrameRate(string id) =>
        Get(id) is { } session && _frameRates.TryGetValue(session, out StrongBox<int>? fps) ? fps.Value : null;
}

/// <summary>Open live vision-language sessions by id. Closing one stops its loop and releases the device slot it holds.</summary>
public sealed class RealtimeSessionRegistry(TimeSpan idleTimeout)
    : SessionRegistry<IRealtimeVlmSession>(idleTimeout, static session => session.DisposeAsync());

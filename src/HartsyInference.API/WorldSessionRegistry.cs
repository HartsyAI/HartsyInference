using HartsyInference.Engine.Requests;

namespace HartsyInference.API;

/// <summary>Open interactive-world sessions by id (see <see cref="SessionRegistry{TSession}"/>). The CLI never needs this, since it holds one
/// <see cref="IWorldSession"/> for one process; an HTTP server serves many clients and finds a session again on each request.</summary>
public sealed class WorldSessionRegistry(TimeSpan idleTimeout)
    : SessionRegistry<IWorldSession>(idleTimeout, static session =>
    {
        session.Dispose();
        return ValueTask.CompletedTask;
    });

/// <summary>Open live vision-language sessions by id. Closing one stops its loop and releases the device slot it holds.</summary>
public sealed class RealtimeSessionRegistry(TimeSpan idleTimeout)
    : SessionRegistry<IRealtimeVlmSession>(idleTimeout, static session => session.DisposeAsync());

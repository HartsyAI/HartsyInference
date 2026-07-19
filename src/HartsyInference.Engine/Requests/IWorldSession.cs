namespace HartsyInference.Engine.Requests;

/// <summary>A stateful interactive-world session: send actions and pull the frames the world produces in response.
/// The session owns its model/device state until disposed.</summary>
public interface IWorldSession : IDisposable
{
    /// <summary>Feeds an action string (host-defined, e.g. a control token) into the world's next step.</summary>
    void SendAction(string action);

    /// <summary>Streams generated frames as the world advances; ends when the session is disposed or cancelled.</summary>
    IAsyncEnumerable<VideoFrame> StreamAsync(CancellationToken cancel);
}

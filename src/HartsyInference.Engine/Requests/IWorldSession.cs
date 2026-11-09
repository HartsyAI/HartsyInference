namespace HartsyInference.Engine.Requests;

/// <summary>A stateful interactive-world session: send actions and pull the frames the world produces in response. The session owns its model/device state until disposed.</summary>
public interface IWorldSession : IDisposable
{
    /// <summary>Feeds an action string (host-defined, e.g. a control token) into the world's next step. A batch session takes
    /// every action before its stream starts and throws <see cref="InvalidOperationException"/> for a later one.</summary>
    void SendAction(string action);

    /// <summary>Streams generated frames. A progressive session continues as the world advances and ends when it is disposed or
    /// cancelled. A batch session renders its clip when the stream starts and delivers the completed frames; a cancel before or
    /// during that run ends the stream with <see cref="OperationCanceledException"/> and no frames. A batch session streams once,
    /// so a second enumeration throws <see cref="InvalidOperationException"/>.</summary>
    IAsyncEnumerable<VideoFrame> StreamAsync(CancellationToken cancel);
}

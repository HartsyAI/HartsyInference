using HartsyInference.World.Sessions;

namespace HartsyInference.Engine.Worlds;

/// <summary>A streaming world model: it opens a stream that generates a chunk per call. The engine wraps the stream in a
/// session that runs it on one compute thread under the execution lease, delivers each chunk's frames as it completes, and
/// passes on the actions the caller sends in between, so the model only has to open the stream and generate chunks.</summary>
public interface IChunkedWorldPipeline
{
    /// <summary>Opens the stream <paramref name="request"/>'s plan describes: loads what stays resident for the whole stream and
    /// encodes its conditioning. The engine disposes the returned stream when the session ends, on the thread that opened it.
    /// Checks <paramref name="cancel"/> between stages; a load that observes it throws <see cref="OperationCanceledException"/>
    /// with everything it had acquired released.</summary>
    IChunkedWorldStream OpenStream(ChunkedWorldRequest request, CancellationToken cancel);
}

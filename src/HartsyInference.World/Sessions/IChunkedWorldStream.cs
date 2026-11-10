using HartsyInference.Diffusion.Requests;
using HartsyInference.Video;

namespace HartsyInference.World.Sessions;

/// <summary>A world model generating a stream chunk by chunk: each <see cref="GenerateChunk"/> advances the world by one
/// chunk and returns its finished frames, and actions sent in between steer the chunks that follow. This is the contract a
/// chunk-causal world model implements to be driven live; the model keeps its own temporal state between chunks, and whoever
/// holds the stream calls it from one compute thread.</summary>
public interface IChunkedWorldStream : IDisposable
{
    /// <summary>Output frame width in pixels.</summary>
    int FrameWidth { get; }

    /// <summary>Output frame height in pixels.</summary>
    int FrameHeight { get; }

    /// <summary>Chunks generated so far.</summary>
    int ChunksGenerated { get; }

    /// <summary>Frames delivered so far.</summary>
    int FramesGenerated { get; }

    /// <summary>Frames the next chunk will deliver.</summary>
    int NextChunkFrames { get; }

    /// <summary>Whether a bounded stream has delivered its last chunk. Always false for an unbounded stream.</summary>
    bool Completed { get; }

    /// <summary>Feeds an action in the model's action language. It steers the chunks generated after it arrives; a chunk
    /// already being generated is not affected. Safe to call from any thread.</summary>
    /// <exception cref="FormatException">The action is not valid in the model's language.</exception>
    /// <exception cref="InvalidOperationException">The stream takes no live actions (its path was fixed when it opened).</exception>
    void SendAction(string action);

    /// <summary>Generates the next chunk and returns its frames in order, each carrying its index in the whole stream. Checks
    /// <paramref name="cancel"/> between denoising passes; a chunk that fails or is cancelled leaves the stream at the
    /// previous chunk, so the call can be repeated.</summary>
    /// <exception cref="InvalidOperationException">The stream is <see cref="Completed"/>.</exception>
    IReadOnlyList<VideoFrame> GenerateChunk(CancellationToken cancel, Action<GenerationProgress>? progress = null);
}

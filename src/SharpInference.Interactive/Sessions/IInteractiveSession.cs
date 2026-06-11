using SharpInference.Conditioning;
using SharpInference.Video;

namespace SharpInference.Interactive.Sessions;

/// <summary>A live, bidirectional generation session: the application pushes one <see cref="ActionInput"/> per frame in
/// and reads <see cref="VideoFrame"/>s out, indefinitely. Unlike the offline video pipelines (fixed-length
/// <c>IAsyncEnumerable</c>), a session maintains persistent rolling state (history latents, memory bank, caches) and
/// is driven by the action stream. See <c>docs/Research/INTERACTIVE_INFERENCE.md</c> § 3.</summary>
public interface IInteractiveSession : IAsyncDisposable
{
    /// <summary>Output frame width in pixels.</summary>
    int FrameWidth { get; }

    /// <summary>Output frame height in pixels.</summary>
    int FrameHeight { get; }

    /// <summary>Nominal output rate the session paces toward.</summary>
    int TargetFps { get; }

    /// <summary>Monotonic index of the most recently emitted frame (-1 before the first frame).</summary>
    long CurrentFrameIndex { get; }

    /// <summary>Queues one frame's action. Blocks (asynchronously) when the bounded action queue is full.</summary>
    ValueTask SubmitActionAsync(ActionInput action, CancellationToken ct = default);

    /// <summary>Streams generated frames. When the application falls behind, the oldest unread frame is dropped to
    /// keep latency bounded (see <see cref="InteractiveSessionStats.DroppedFrames"/>).</summary>
    IAsyncEnumerable<VideoFrame> ReadFramesAsync(CancellationToken ct = default);

    /// <summary>Snapshot of step-latency percentiles and queue health.</summary>
    InteractiveSessionStats GetStats();

    /// <summary>Applies a speed/quality trade-off (denoise steps, guidance) without recreating the session.</summary>
    void SetQualityProfile(QualityProfile profile);
}

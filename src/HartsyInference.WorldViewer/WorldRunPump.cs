using HartsyInference.Core.Diagnostics;
using HartsyInference.Engine.Requests;

namespace HartsyInference.WorldViewer;

/// <summary>Carries a steered run's frames to the caller's sink and to the window, and decides when the window goes away.</summary>
/// <remarks>
/// <list type="bullet">
/// <item>A run that ends normally leaves its window up with the buffer still playing. The caller writes its files, then calls
/// <see cref="WaitForCloseAsync"/>.</item>
/// <item>A run that fails is reported to the window, which keeps the error on screen until the user closes it. The exception is then
/// rethrown.</item>
/// <item>Any other exit, such as a cancelled run or a user stop, closes the window at once.</item>
/// </list>
/// The steering stops at the first exit of any kind, before the window is waited on, so keys cannot reach a session that has
/// failed. <see cref="Dispose"/> closes the window; the caller disposes the pump when its run is over, before it disposes the
/// cancellation source the run was given.
/// </remarks>
public sealed class WorldRunPump : IDisposable
{
    private readonly WorldSteering _steering;
    private readonly WorldPlayback? _playback;
    private readonly IWorldWindow? _window;
    private bool _disposed;

    /// <summary>Creates a pump over <paramref name="steering"/>. <paramref name="playback"/> and <paramref name="window"/> are null for
    /// a run with no window.</summary>
    public WorldRunPump(WorldSteering steering, WorldPlayback? playback, IWorldWindow? window)
    {
        ArgumentNullException.ThrowIfNull(steering);
        _steering = steering;
        _playback = playback;
        _window = window;
    }

    /// <summary>Hands each frame of <paramref name="frames"/> to <paramref name="keep"/>, then to the playback buffer, which waits while
    /// the window is behind. Returns when the stream ends, and rethrows whatever ended it.</summary>
    public async Task RunAsync(IAsyncEnumerable<VideoFrame> frames, Action<VideoFrame> keep, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(frames);
        ArgumentNullException.ThrowIfNull(keep);
        try
        {
            long waitStart = WorldStageTimings.Start();
            await foreach (VideoFrame frame in frames.ConfigureAwait(false))
            {
                WorldStageTimings.Record(WorldStage.FrameWaitStream, waitStart);
                keep(frame);
                if (_playback is not null)
                {
                    long handoffStart = WorldStageTimings.Start();
                    await _playback.WriteAsync(frame, cancel).ConfigureAwait(false);
                    WorldStageTimings.Record(WorldStage.FramePlaybackHandoff, handoffStart);
                }
                waitStart = WorldStageTimings.Start();
            }
        }
        catch (Exception ex)
        {
            _steering.Close();
            bool stopped = ex is OperationCanceledException;
            _playback?.Complete(stopped ? null : ex.Message);
            if (!stopped && !cancel.IsCancellationRequested)
            {
                await WaitForCloseAsync(cancel).ConfigureAwait(false);
            }
            Dispose();
            throw;
        }
        _steering.Close();
        _playback?.Complete();
    }

    /// <summary>Waits until the window is closed. A cancelled <paramref name="cancel"/> ends the wait and leaves the window open; the
    /// caller's files are already written at that point, so the wait's cancellation is not an error.</summary>
    public async Task WaitForCloseAsync(CancellationToken cancel)
    {
        if (_window is null)
        {
            return;
        }
        try
        {
            await _window.WaitUntilClosedAsync(cancel).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancel.IsCancellationRequested)
        {
        }
    }

    /// <summary>Closes the window. Idempotent.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        _window?.Dispose();
    }
}

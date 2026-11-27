using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Worlds;

namespace HartsyInference.WorldViewer.Tests;

/// <summary>A window that records how the run used it. The user's close is simulated by <see cref="UserCloses"/>.</summary>
internal sealed class FakeWorldWindow : IWorldWindow
{
    private readonly TaskCompletionSource _closed = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly TaskCompletionSource _waiting = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private int _waitCalls;
    private int _disposals;

    /// <summary>Completes when the pump first waits on the window.</summary>
    public Task Waiting => _waiting.Task;

    public int WaitCalls => Volatile.Read(ref _waitCalls);

    public int Disposals => Volatile.Read(ref _disposals);

    public bool Disposed => Disposals > 0;

    /// <summary>Simulates the user closing the window.</summary>
    public void UserCloses() => _closed.TrySetResult();

    public Task WaitUntilClosedAsync(CancellationToken cancel)
    {
        Interlocked.Increment(ref _waitCalls);
        _waiting.TrySetResult();
        return _closed.Task.WaitAsync(cancel);
    }

    public void Dispose() => Interlocked.Increment(ref _disposals);
}

public sealed class WorldRunPumpTests
{
    [Fact]
    public async Task A_failing_stream_is_reported_to_the_window_which_stays_up_until_it_is_closed()
    {
        ScriptedWorldSession session = new(cancel => FailAfter(2, new InvalidOperationException("boom"), cancel));
        WorldSteering steering = new(session, SanaWmControlScheme.Instance);
        WorldPlayback playback = new(fps: 16, bufferFrames: 32);
        FakeWorldWindow window = new();
        using WorldRunPump pump = new(steering, playback, window);
        List<int> kept = [];

        Task run = pump.RunAsync(session.StreamAsync(CancellationToken.None), frame => kept.Add(frame.Index), CancellationToken.None);
        await window.Waiting.WaitAsync(TimeSpan.FromSeconds(5));

        // The failure is on the window's status, and the window is still open for the user to read it.
        WorldPlaybackStatus status = playback.Status();
        Assert.Equal(WorldPlaybackPhase.Failed, status.Phase);
        Assert.Equal("boom", status.Error);
        Assert.False(window.Disposed);
        Assert.False(steering.Press("W"), "keys must not reach a session that has failed");

        window.UserCloses();
        InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(() => run);
        Assert.Equal("boom", error.Message);
        Assert.Equal([0, 1], kept);
        Assert.Equal(1, window.Disposals);
    }

    [Fact]
    public async Task A_sink_that_fails_is_reported_the_same_way_as_a_failing_stream()
    {
        ScriptedWorldSession session = new(cancel => TestFrames.Run(0, 3, cancel));
        WorldSteering steering = new(session, SanaWmControlScheme.Instance);
        WorldPlayback playback = new(fps: 16, bufferFrames: 32);
        FakeWorldWindow window = new();
        using WorldRunPump pump = new(steering, playback, window);

        // The spool's write fails on the second frame: the run is a failure, not a cancellation, so the window waits for the user.
        Task run = pump.RunAsync(session.StreamAsync(CancellationToken.None), frame =>
        {
            if (frame.Index == 1)
            {
                throw new IOException("disk full");
            }
        }, CancellationToken.None);
        await window.Waiting.WaitAsync(TimeSpan.FromSeconds(5));

        Assert.Equal("disk full", playback.Status().Error);
        window.UserCloses();
        await Assert.ThrowsAsync<IOException>(() => run);
        Assert.Equal(1, window.Disposals);
    }

    [Fact]
    public async Task A_run_cancelled_mid_stream_closes_the_window_at_once_without_waiting_for_the_user()
    {
        using CancellationTokenSource stop = new();
        ScriptedWorldSession session = new(_ => StopAfterOne(stop));
        WorldSteering steering = new(session, SanaWmControlScheme.Instance);
        WorldPlayback playback = new(fps: 16, bufferFrames: 32);
        FakeWorldWindow window = new();
        using WorldRunPump pump = new(steering, playback, window);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            pump.RunAsync(session.StreamAsync(stop.Token), _ => { }, stop.Token));

        Assert.Equal(0, window.WaitCalls);
        Assert.Equal(1, window.Disposals);
        Assert.Null(playback.Status().Error);
    }

    [Fact]
    public async Task A_normal_end_leaves_the_window_open_until_the_caller_waits_for_it()
    {
        ScriptedWorldSession session = new(cancel => TestFrames.Run(0, 3, cancel));
        WorldSteering steering = new(session, SanaWmControlScheme.Instance);
        WorldPlayback playback = new(fps: 16, bufferFrames: 32);
        FakeWorldWindow window = new();
        WorldRunPump pump = new(steering, playback, window);
        List<int> kept = [];

        await pump.RunAsync(session.StreamAsync(CancellationToken.None), frame => kept.Add(frame.Index), CancellationToken.None);

        Assert.Equal([0, 1, 2], kept);
        Assert.Null(playback.Status().Error);
        Assert.False(window.Disposed);
        Task wait = pump.WaitForCloseAsync(CancellationToken.None);
        Assert.False(wait.IsCompleted);
        window.UserCloses();
        await wait.WaitAsync(TimeSpan.FromSeconds(5));

        pump.Dispose();
        pump.Dispose();
        Assert.Equal(1, window.Disposals);
    }

    [Fact]
    public async Task Ctrl_C_while_waiting_for_the_window_returns_so_the_written_files_are_kept()
    {
        ScriptedWorldSession session = new(cancel => TestFrames.Run(0, 1, cancel));
        WorldSteering steering = new(session, SanaWmControlScheme.Instance);
        FakeWorldWindow window = new();
        using WorldRunPump pump = new(steering, new WorldPlayback(fps: 16, bufferFrames: 32), window);
        using CancellationTokenSource ctrlC = new();

        Task wait = pump.WaitForCloseAsync(ctrlC.Token);
        ctrlC.Cancel();

        await wait.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.False(window.Disposed, "the caller decides when the window goes; cancelling the wait only ends the wait");
    }

    [Fact]
    public async Task A_pre_cancelled_wait_with_no_window_returns_at_once()
    {
        using WorldRunPump pump = new(new WorldSteering(new ScriptedWorldSession(cancel => TestFrames.Run(0, 1, cancel)), null),
            playback: null, window: null);
        using CancellationTokenSource cancelled = new();
        cancelled.Cancel();

        await pump.WaitForCloseAsync(cancelled.Token).WaitAsync(TimeSpan.FromSeconds(5));
    }

    private static async IAsyncEnumerable<VideoFrame> FailAfter(int frames, Exception error, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancel)
    {
        for (int i = 0; i < frames; i++)
        {
            cancel.ThrowIfCancellationRequested();
            yield return TestFrames.Frame(i);
        }
        throw error;
    }

    private static async IAsyncEnumerable<VideoFrame> StopAfterOne(CancellationTokenSource stop)
    {
        yield return TestFrames.Frame(0);
        stop.Cancel();
        await Task.Yield();
        stop.Token.ThrowIfCancellationRequested();
    }
}

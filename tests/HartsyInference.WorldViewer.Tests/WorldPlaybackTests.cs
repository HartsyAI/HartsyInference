using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Worlds;

namespace HartsyInference.WorldViewer.Tests;

public sealed class WorldPlaybackTests
{
    [Fact]
    public async Task The_delivered_rate_stops_at_the_end_of_the_run_not_when_the_window_is_read()
    {
        WorldPlayback playback = new(fps: 16, bufferFrames: 32);
        await playback.WriteAsync(TestFrames.Frame(0), CancellationToken.None);
        await playback.WriteAsync(TestFrames.Frame(1), CancellationToken.None);
        playback.Complete();
        double atEnd = playback.Status().DeliveredPerSecond;

        await Task.Delay(300);
        double later = playback.Status().DeliveredPerSecond;

        Assert.True(atEnd > 0);
        Assert.Equal(atEnd, later, precision: 6);
    }

    [Fact]
    public void A_per_step_stream_gets_a_two_frame_buffer_and_any_other_gets_two_seconds()
    {
        Assert.Equal(WorldPlayback.PerStepBufferFrames, WorldPlayback.DefaultBufferFrames(24, WorldControlCadence.PerStep));
        Assert.Equal(2, WorldPlayback.PerStepBufferFrames);
        Assert.Equal(32, WorldPlayback.DefaultBufferFrames(16, WorldControlCadence.OnChange));
        Assert.Equal(32, WorldPlayback.DefaultBufferFrames(16, null));
    }

    [Fact]
    public async Task A_two_frame_buffer_shows_its_first_frame_once_two_frames_are_in()
    {
        WorldPlayback playback = new(fps: 24, bufferFrames: WorldPlayback.PerStepBufferFrames);
        await playback.WriteAsync(TestFrames.Frame(0), CancellationToken.None);
        Assert.Equal(WorldPaceStep.Buffering, playback.Step(TimeSpan.Zero, out _));
        await playback.WriteAsync(TestFrames.Frame(1), CancellationToken.None);
        Assert.Equal(WorldPaceStep.Show, playback.Step(TimeSpan.Zero, out VideoFrame? first));
        Assert.Equal(0, first!.Index);
    }

    [Fact]
    public async Task A_full_buffer_holds_the_writer_until_a_frame_is_shown()
    {
        WorldPlayback playback = new(fps: 2, bufferFrames: 2);
        await playback.WriteAsync(TestFrames.Frame(0), CancellationToken.None);
        await playback.WriteAsync(TestFrames.Frame(1), CancellationToken.None);

        Task third = playback.WriteAsync(TestFrames.Frame(2), CancellationToken.None).AsTask();
        Task finished = await Task.WhenAny(third, Task.Delay(TimeSpan.FromMilliseconds(200)));
        Assert.NotSame(third, finished);

        Assert.Equal(WorldPaceStep.Show, playback.Step(TimeSpan.Zero, out VideoFrame? first));
        Assert.Equal(0, first!.Index);
        await third.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(2, playback.Status().BufferedFrames);
    }

    [Fact]
    public async Task Frames_come_out_in_order_and_the_status_follows_the_phase()
    {
        WorldPlayback playback = new(fps: 4, bufferFrames: 8);
        Assert.Equal(WorldPlaybackPhase.Generating, playback.Status().Phase);

        for (int i = 0; i < 4; i++)
        {
            await playback.WriteAsync(TestFrames.Frame(i), CancellationToken.None);
        }
        List<int> shown = [];
        TimeSpan now = TimeSpan.Zero;
        while (shown.Count < 4)
        {
            if (playback.Step(now, out VideoFrame? frame) == WorldPaceStep.Show)
            {
                shown.Add(frame!.Index);
            }
            now += TimeSpan.FromMilliseconds(5);
        }

        Assert.Equal([0, 1, 2, 3], shown);
        Assert.Equal(WorldPlaybackPhase.Playing, playback.Status().Phase);
        playback.Complete();
        Assert.Equal(WorldPlaybackPhase.Finished, playback.Status().Phase);
        Assert.Equal(4, playback.Status().FramesShown);
    }

    [Fact]
    public async Task A_failed_stream_is_reported_and_keeps_the_frames_it_delivered()
    {
        WorldPlayback playback = new(fps: 4, bufferFrames: 8);
        await playback.WriteAsync(TestFrames.Frame(0), CancellationToken.None);
        playback.Complete("GPU fault");

        WorldPlaybackStatus status = playback.Status();
        Assert.Equal(WorldPlaybackPhase.Failed, status.Phase);
        Assert.Equal("GPU fault", status.Error);
        Assert.Equal(1, status.BufferedFrames);
        Assert.Equal(WorldPaceStep.Show, playback.Step(TimeSpan.Zero, out _));
    }

    [Fact]
    public async Task Delivered_rate_is_zero_before_frames_and_positive_once_they_arrive()
    {
        WorldPlayback playback = new(fps: 16, bufferFrames: 32);
        Assert.Equal(0, playback.Status().DeliveredPerSecond);

        await playback.WriteAsync(TestFrames.Frame(0), CancellationToken.None);
        await Task.Delay(TimeSpan.FromMilliseconds(20));
        await playback.WriteAsync(TestFrames.Frame(1), CancellationToken.None);

        WorldPlaybackStatus status = playback.Status();
        Assert.Equal(2, status.FramesReceived);
        Assert.True(status.DeliveredPerSecond > 0, "a rate is reported once frames arrive");
    }

    [Fact]
    public async Task Writing_after_completion_is_refused()
    {
        WorldPlayback playback = new(fps: 4, bufferFrames: 4);
        playback.Complete();

        await Assert.ThrowsAsync<InvalidOperationException>(() => playback.WriteAsync(TestFrames.Frame(0), CancellationToken.None).AsTask());
    }
}

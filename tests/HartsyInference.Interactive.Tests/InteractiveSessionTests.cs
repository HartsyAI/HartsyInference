using Xunit;
using HartsyInference.Conditioning;
using HartsyInference.Interactive.Sessions;
using HartsyInference.Video;

namespace HartsyInference.Interactive.Tests;

/// <summary>Queue/backpressure behavior of <see cref="BackgroundComputeSession"/> with a stub stepper: round-trip,
/// drop-oldest on a slow consumer, repeat-last on action underrun, quality-profile handoff, and clean disposal.</summary>
public class InteractiveSessionTests
{
    private sealed class StubStepper : IFrameStepper
    {
        private int _frameIndex;
        public readonly List<long> SeenActionFrames = new();
        public volatile QualityProfile? Applied;

        public int FrameWidth => 4;
        public int FrameHeight => 4;

        public IReadOnlyList<VideoFrame> Step(in ActionInput action)
        {
            lock (SeenActionFrames) SeenActionFrames.Add(action.FrameIndex);
            return [new VideoFrame(_frameIndex++, 4, 4, new byte[4 * 4 * 3])];
        }

        public void ApplyQualityProfile(QualityProfile profile) => Applied = profile;
    }

    private static ActionInput Action(long frameIndex) => new(new byte[] { 1, 2 }, frameIndex, 0);

    [Fact]
    public async Task SubmitAndRead_RoundTripsFramesInOrder()
    {
        StubStepper stepper = new();
        await using BackgroundComputeSession session = new(stepper, targetFps: 100);
        for (int i = 0; i < 3; i++) await session.SubmitActionAsync(Action(i));

        List<VideoFrame> frames = new();
        using CancellationTokenSource cts = new(TimeSpan.FromSeconds(10));
        await foreach (VideoFrame f in session.ReadFramesAsync(cts.Token))
        {
            frames.Add(f);
            if (frames.Count == 3) break;
        }
        // The compute thread runs free (repeat-last on underrun) and the bounded output queue
        // drop-oldests under load, so the exact set [0,1,2] is not guaranteed. The real invariant this
        // test pins is ORDER: frames arrive strictly increasing and begin at frame 0.
        List<long> indices = frames.Select(f => f.Index).ToList();
        Assert.Equal(0, indices[0]);
        for (int i = 1; i < indices.Count; i++)
            Assert.True(indices[i] > indices[i - 1], $"frame order regressed: [{string.Join(", ", indices)}]");
        Assert.True(session.CurrentFrameIndex >= 2);
    }

    [Fact]
    public async Task SlowConsumer_DropsOldestFrames()
    {
        StubStepper stepper = new();
        await using BackgroundComputeSession session = new(stepper, targetFps: 1000, frameQueueDepth: 2);
        await session.SubmitActionAsync(Action(0));

        // The session repeats the last action on underrun, producing frames nobody reads — the bounded
        // output queue must drop the oldest instead of stalling the compute thread.
        DateTime deadline = DateTime.UtcNow + TimeSpan.FromSeconds(10);
        while (session.GetStats().DroppedFrames == 0 && DateTime.UtcNow < deadline)
            await Task.Delay(10);

        InteractiveSessionStats stats = session.GetStats();
        Assert.True(stats.DroppedFrames > 0, "expected drop-oldest to engage");
        Assert.True(stats.FramesEmitted > stats.DroppedFrames);
        Assert.True(stats.P50StepMs >= 0);
    }

    [Fact]
    public async Task ActionUnderrun_RepeatsLastActionWithAdvancingFrameIndex()
    {
        StubStepper stepper = new();
        await using BackgroundComputeSession session = new(stepper, targetFps: 200);
        await session.SubmitActionAsync(Action(7));

        DateTime deadline = DateTime.UtcNow + TimeSpan.FromSeconds(10);
        while (DateTime.UtcNow < deadline)
        {
            lock (stepper.SeenActionFrames)
                if (stepper.SeenActionFrames.Count >= 3) break;
            await Task.Delay(10);
        }

        lock (stepper.SeenActionFrames)
        {
            Assert.True(stepper.SeenActionFrames.Count >= 3, "expected repeat-last to keep stepping");
            Assert.Equal(7, stepper.SeenActionFrames[0]);
            Assert.Equal(8, stepper.SeenActionFrames[1]);   // repeated action advances the frame index
            Assert.Equal(9, stepper.SeenActionFrames[2]);
        }
    }

    [Fact]
    public async Task SetQualityProfile_ReachesTheStepperOnTheComputeThread()
    {
        StubStepper stepper = new();
        await using BackgroundComputeSession session = new(stepper, targetFps: 200);
        session.SetQualityProfile(QualityProfile.Medium);
        await session.SubmitActionAsync(Action(0));

        DateTime deadline = DateTime.UtcNow + TimeSpan.FromSeconds(10);
        while (stepper.Applied is null && DateTime.UtcNow < deadline)
            await Task.Delay(10);
        Assert.Equal(QualityProfile.Medium, stepper.Applied);
    }

    [Fact]
    public async Task DisposeAsync_StopsTheLoopAndCompletesTheFrameStream()
    {
        StubStepper stepper = new();
        BackgroundComputeSession session = new(stepper, targetFps: 200);
        await session.SubmitActionAsync(Action(0));
        await session.DisposeAsync();
        await session.DisposeAsync();   // idempotent

        await Assert.ThrowsAsync<ObjectDisposedException>(async () => await session.SubmitActionAsync(Action(1)));

        // The frame channel is completed — enumeration ends rather than hanging.
        await foreach (VideoFrame _ in session.ReadFramesAsync())
        {
        }
    }

    [Fact]
    public async Task OversizedPayload_IsRejected()
    {
        StubStepper stepper = new();
        await using BackgroundComputeSession session = new(stepper, maxPayloadBytes: 4);
        await Assert.ThrowsAsync<ArgumentException>(async () =>
            await session.SubmitActionAsync(new ActionInput(new byte[8], 0, 0)));
    }
}

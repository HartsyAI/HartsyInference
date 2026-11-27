using System.Diagnostics;
using System.Runtime.CompilerServices;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Worlds;

namespace HartsyInference.WorldViewer.Tests;

public sealed class WorldFrameStreamTests
{
    [Fact]
    public async Task On_change_session_is_enumerated_once_and_no_action_is_sent_without_a_key()
    {
        ScriptedWorldSession session = new(cancel => TestFrames.Run(0, 3, cancel));
        WorldSteering steering = new(session, SanaWmControlScheme.Instance);
        WorldFrameStream stream = new(session, steering, maxFrames: 0);

        List<int> indices = [];
        await foreach (VideoFrame frame in stream.StreamAsync(CancellationToken.None))
        {
            indices.Add(frame.Index);
        }

        Assert.Equal([0, 1, 2], indices);
        Assert.Equal(1, session.StreamCalls);
        Assert.Empty(session.Actions);
    }

    [Fact]
    public async Task Per_step_session_gets_the_current_action_before_each_step_and_follows_key_changes()
    {
        PerStepWorldSession session = new();
        WorldSteering steering = new(session, OasisControlScheme.Instance);
        WorldFrameStream stream = new(session, steering, maxFrames: 3);
        steering.Press("W");

        List<int> indices = [];
        await foreach (VideoFrame frame in stream.StreamAsync(CancellationToken.None))
        {
            indices.Add(frame.Index);
            if (frame.Index == 1)
            {
                steering.Press("Space");
            }
        }

        Assert.Equal([0, 1, 2], indices);
        Assert.Equal(["forward", "forward", "forward+jump"], session.Taken);
    }

    [Fact]
    public async Task Per_step_session_that_yields_nothing_for_its_action_is_an_error()
    {
        ScriptedWorldSession session = new(_ => TestFrames.Run(0, 0));
        WorldSteering steering = new(session, DiamondControlScheme.Instance);
        WorldFrameStream stream = new(session, steering, maxFrames: 0);

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (VideoFrame _ in stream.StreamAsync(CancellationToken.None))
            {
            }
        });
    }

    [Fact]
    public async Task A_failing_stream_surfaces_its_error_to_the_consumer()
    {
        ScriptedWorldSession session = new(_ => FailAfterOne());
        WorldSteering steering = new(session, SanaWmControlScheme.Instance);
        WorldFrameStream stream = new(session, steering, maxFrames: 0);

        InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (VideoFrame _ in stream.StreamAsync(CancellationToken.None))
            {
            }
        });
        Assert.Equal("the GPU fell over", error.Message);
    }

    [Fact]
    public async Task Cancelling_ends_the_stream_as_cancelled()
    {
        using CancellationTokenSource cancel = new();
        ScriptedWorldSession session = new(token => TestFrames.Run(0, 100, token));
        WorldSteering steering = new(session, SanaWmControlScheme.Instance);
        WorldFrameStream stream = new(session, steering, maxFrames: 0);

        int seen = 0;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (VideoFrame _ in stream.StreamAsync(cancel.Token))
            {
                seen++;
                if (seen == 2)
                {
                    cancel.Cancel();
                }
            }
        });
        Assert.Equal(2, seen);
    }

    [Fact]
    public async Task A_chunk_burst_is_buffered_and_plays_back_in_order()
    {
        ScriptedWorldSession session = new(ChunkedStream);
        WorldSteering steering = new(session, SanaWmControlScheme.Instance);
        WorldPlayback playback = new(fps: 16, bufferFrames: 32);
        WorldFrameStream stream = new(session, steering, maxFrames: 48);

        Task producer = Task.Run(async () =>
        {
            await foreach (VideoFrame frame in stream.StreamAsync(CancellationToken.None))
            {
                await playback.WriteAsync(frame, CancellationToken.None);
            }
            playback.Complete();
        });

        List<int> shown = [];
        Stopwatch clock = Stopwatch.StartNew();
        while (shown.Count < 48 && clock.Elapsed < TimeSpan.FromSeconds(20))
        {
            if (playback.Step(clock.Elapsed, out VideoFrame? frame) == WorldPaceStep.Show)
            {
                shown.Add(frame!.Index);
            }
            await Task.Delay(TimeSpan.FromMilliseconds(2));
        }
        await producer.WaitAsync(TimeSpan.FromSeconds(5));

        Assert.Equal(Enumerable.Range(0, 48), shown);
        Assert.Equal(48, playback.Status().FramesReceived);
    }

    private static async IAsyncEnumerable<VideoFrame> ChunkedStream([EnumeratorCancellation] CancellationToken cancel)
    {
        for (int chunk = 0; chunk < 2; chunk++)
        {
            await Task.Delay(TimeSpan.FromMilliseconds(40), cancel);
            for (int i = 0; i < 24; i++)
            {
                yield return TestFrames.Frame(chunk * 24 + i);
            }
        }
    }

    private static async IAsyncEnumerable<VideoFrame> FailAfterOne()
    {
        yield return TestFrames.Frame(0);
        await Task.Yield();
        throw new InvalidOperationException("the GPU fell over");
    }
}

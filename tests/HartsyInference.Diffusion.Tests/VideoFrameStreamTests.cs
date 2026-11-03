using System.Runtime.CompilerServices;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The lease a streaming video holds is released by its producer however the stream ends: completed, cancelled before the
/// producer starts, or abandoned by a consumer that stops reading. Each case would leave the lease held, and every later model switch
/// would then wait out the lease wait, if its release were lost.</summary>
public sealed class VideoFrameStreamTests
{
    [Fact]
    public async Task CompletedStream_ReleasesTheLeaseOnce()
    {
        FramePipeline pipeline = new(frames: 3);
        CountingLease lease = new();

        int count = 0;
        await foreach (VideoFrame _ in VideoFrameStream.Produce(pipeline, Request(), null!, null, lease))
        {
            count++;
        }

        Assert.Equal(3, count);
        Assert.Equal(1, lease.Disposals);
    }

    [Fact]
    public async Task CancelledBeforeTheProducerStarts_StillReleasesTheLease()
    {
        FramePipeline pipeline = new(frames: 8);
        CountingLease lease = new();
        using CancellationTokenSource cancel = new();
        cancel.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (VideoFrame _ in VideoFrameStream.Produce(pipeline, Request(), null!, null, lease, cancel.Token))
            {
            }
        });

        await WaitForAsync(() => lease.Disposals > 0);
        Assert.Equal(1, lease.Disposals);
    }

    [Fact]
    public async Task ConsumerThatStopsEarly_StopsTheProducerAndReleasesTheLease()
    {
        // More frames than the channel holds, so an unstopped producer would block on a full channel with the lease held.
        FramePipeline pipeline = new(frames: 64);
        CountingLease lease = new();
        IAsyncEnumerator<VideoFrame> frames = VideoFrameStream.Produce(pipeline, Request(), null!, null, lease).GetAsyncEnumerator();
        Assert.True(await frames.MoveNextAsync());

        await frames.DisposeAsync();

        await WaitForAsync(() => lease.Disposals > 0);
        Assert.Equal(1, lease.Disposals);
    }

    private static VideoRequest Request() => new VideoRequest { Prompt = "stream" };

    private static async Task WaitForAsync(Func<bool> condition)
    {
        for (int attempt = 0; attempt < 500 && !condition(); attempt++)
        {
            await Task.Delay(10);
        }

        Assert.True(condition(), "The lease was not released in time.");
    }

    /// <summary>Counts how many times its holder released it.</summary>
    private sealed class CountingLease : IDisposable
    {
        private int _disposals;

        public int Disposals => Volatile.Read(ref _disposals);

        public void Dispose() => Interlocked.Increment(ref _disposals);
    }

    /// <summary>Yields a fixed number of frames, honouring cancellation between them.</summary>
    private sealed class FramePipeline(int frames) : IVideoRecipePipeline
    {
        public bool SupportsStreaming => true;

        public VideoGenerationResult Generate(VideoRequest request, VideoEffectiveSettings settings, IProgress<StepPreview>? progress,
            CancellationToken cancel) => throw new NotSupportedException("Streaming tests never run the buffered path.");

        public async IAsyncEnumerable<VideoFrame> GenerateFramesAsync(VideoRequest request, VideoEffectiveSettings settings,
            IProgress<StepPreview>? progress, [EnumeratorCancellation] CancellationToken cancel)
        {
            for (int index = 0; index < frames; index++)
            {
                cancel.ThrowIfCancellationRequested();
                await Task.Yield();
                yield return new VideoFrame { Rgb = [0, 0, 0], Width = 1, Height = 1, Index = index };
            }
        }

        public void Dispose()
        {
        }
    }
}

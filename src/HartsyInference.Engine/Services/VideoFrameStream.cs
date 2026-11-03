using System.Runtime.CompilerServices;
using System.Threading.Channels;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Streams a pipeline's frames to the caller through a background producer and a bounded channel. The producer owns
/// <paramref name="lease"/> and releases it when it finishes, however the stream ends, so eviction waits for the whole stream.</summary>
internal static class VideoFrameStream
{
    /// <summary>Yields the frames of <paramref name="pipeline"/>. Its first step blocks synchronously through the whole denoise loop, so
    /// the producer runs on a background thread, and the calling thread (an ASP.NET request handler when the extension awaits this) is
    /// not pinned for the denoise duration.</summary>
    public static async IAsyncEnumerable<VideoFrame> Produce(IVideoRecipePipeline pipeline, VideoRequest request,
        VideoEffectiveSettings settings, IProgress<StepPreview>? progress, IDisposable lease,
        [EnumeratorCancellation] CancellationToken cancel = default)
    {
        // Stops the producer when this stream ends early, so a consumer that walks away cannot leave the producer blocked on a full
        // channel with the lease held. Not disposed: the producer may still observe it after the stream ends.
        CancellationTokenSource producerStop = CancellationTokenSource.CreateLinkedTokenSource(cancel);
        Channel<VideoFrame> channel;
        Task producer;
        try
        {
            channel = Channel.CreateBounded<VideoFrame>(
                new BoundedChannelOptions(4) { SingleReader = true, SingleWriter = true, FullMode = BoundedChannelFullMode.Wait });
            // No token on Task.Run: a task cancelled before it starts never runs its finally, so the lease would never be released.
            // Cancellation reaches the producer through its own linked token instead.
            producer = Task.Run(async () =>
            {
                try
                {
                    await foreach (VideoFrame frame in pipeline.GenerateFramesAsync(request, settings, progress, producerStop.Token)
                        .WithCancellation(producerStop.Token).ConfigureAwait(false))
                    {
                        await channel.Writer.WriteAsync(frame, producerStop.Token).ConfigureAwait(false);
                    }
                    channel.Writer.TryComplete();
                }
                catch (Exception ex)
                {
                    channel.Writer.TryComplete(ex);
                }
                finally
                {
                    lease.Dispose();
                }
            });
        }
        catch
        {
            lease.Dispose();
            throw;
        }

        try
        {
            await foreach (VideoFrame frame in channel.Reader.ReadAllAsync(cancel).ConfigureAwait(false))
            {
                yield return frame;
            }
        }
        finally
        {
            // Ends the producer when this stream stops early or is cancelled. After a normal finish the producer has already exited.
            producerStop.Cancel();
        }
        // Observes the producer's result: rethrows what it captured after the channel completed.
        await producer.ConfigureAwait(false);
    }
}

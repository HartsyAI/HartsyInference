using System.Runtime.CompilerServices;
using System.Threading.Channels;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Runs a per-frame vision producer behind a bounded buffer, for <see cref="IVisionService.RunFramesAsync"/>
/// implementations (tracking, interpolation). The producer writes frames as it makes them and blocks once
/// <c>capacity</c> frames are waiting, so a slow consumer bounds memory. However the stream ends (completion, the caller
/// breaking out early, cancellation, or a producer error) the producer is cancelled and awaited, and the state it holds
/// (tracker memory banks, frame buffers) is disposed before the enumerator finishes.</summary>
public static class VisionFrameStream
{
    /// <summary>The default number of frames the producer may run ahead of the consumer.</summary>
    public const int DefaultCapacity = 4;

    /// <summary>Streams the frames <paramref name="produce"/> writes. <paramref name="state"/> is disposed when the stream
    /// ends, after the producer has stopped. A producer exception is rethrown to the consumer.</summary>
    public static async IAsyncEnumerable<VisionFrameResult> Run(
        Func<ChannelWriter<VisionFrameResult>, CancellationToken, Task> produce,
        IDisposable? state,
        int capacity = DefaultCapacity,
        [EnumeratorCancellation] CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(produce);
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        using CancellationTokenSource stop = CancellationTokenSource.CreateLinkedTokenSource(cancel);
        Channel<VisionFrameResult> frames = Channel.CreateBounded<VisionFrameResult>(new BoundedChannelOptions(capacity)
        {
            SingleReader = true,
            SingleWriter = true,
            FullMode = BoundedChannelFullMode.Wait,
        });
        Task producer = Task.Run(async () =>
        {
            try
            {
                await produce(frames.Writer, stop.Token).ConfigureAwait(false);
                frames.Writer.TryComplete();
            }
            catch (Exception ex)
            {
                frames.Writer.TryComplete(ex);
            }
        }, CancellationToken.None);
        try
        {
            await foreach (VisionFrameResult frame in frames.Reader.ReadAllAsync(cancel).ConfigureAwait(false))
                yield return frame;
        }
        finally
        {
            await stop.CancelAsync().ConfigureAwait(false);
            await producer.ConfigureAwait(false);
            state?.Dispose();
        }
    }
}

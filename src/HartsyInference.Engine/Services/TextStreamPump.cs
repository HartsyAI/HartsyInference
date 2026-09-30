using System.Runtime.CompilerServices;
using System.Threading.Channels;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Runs a producer on a worker and yields its chunks; the worker is always cancelled and awaited when the consumer stops early, so an abandoned stream releases whatever the producer holds.</summary>
/// <remarks>The channel is unbounded by decision: the sink runs on the decode thread inside the slot lock and the
/// device gate, so a bounded channel with a blocking writer would let a slow consumer stall decode while it holds
/// the GPU. Memory is bounded by <c>MaxTokens</c> deltas, and abandonment cancels the producer instead.</remarks>
internal static class TextStreamPump
{
    /// <summary>Streams <paramref name="produce"/>'s sink output, then the chunks it returns; failures become a trailing Cancelled or Error stop.</summary>
    public static async IAsyncEnumerable<TextChunk> Run(
        Func<Action<TextChunk>, CancellationToken, Task<IReadOnlyList<TextChunk>>> produce,
        [EnumeratorCancellation] CancellationToken cancel = default)
    {
        using CancellationTokenSource linked = CancellationTokenSource.CreateLinkedTokenSource(cancel);
        Channel<TextChunk> channel = Channel.CreateUnbounded<TextChunk>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = true,
        });
        // Started without the caller's token: a pre-cancelled request must still run the worker so the
        // terminal chunk is written and the reader below can complete.
        Task worker = Task.Run(() => WorkAsync(produce, channel.Writer, linked.Token), CancellationToken.None);
        try
        {
            // Read without the caller's token: a cancelled request still ends with its Cancelled stop chunk.
            await foreach (TextChunk chunk in channel.Reader.ReadAllAsync(CancellationToken.None).ConfigureAwait(false))
                yield return chunk;
        }
        finally
        {
            linked.Cancel();
            try
            {
                await worker.ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                Logs.Debug("Text stream worker cancelled.");
            }
        }
    }

    private static async Task WorkAsync(
        Func<Action<TextChunk>, CancellationToken, Task<IReadOnlyList<TextChunk>>> produce,
        ChannelWriter<TextChunk> writer, CancellationToken generation)
    {
        // TryWrite never blocks the decode thread; false only after completion, which nothing here does early.
        void Sink(TextChunk chunk) => writer.TryWrite(chunk);
        try
        {
            IReadOnlyList<TextChunk> finals = await produce(Sink, generation).ConfigureAwait(false);
            foreach (TextChunk chunk in finals) writer.TryWrite(chunk);
        }
        catch (OperationCanceledException)
        {
            Logs.Debug("Text stream cancelled.");
            writer.TryWrite(new TextChunk { Kind = TextChunkKind.StopReason, Stop = StopReason.Cancelled });
        }
        catch (Exception ex)
        {
            Logs.Error($"Text stream failed: {ex.Message}", ex);
            writer.TryWrite(new TextChunk { Kind = TextChunkKind.StopReason, Stop = StopReason.Error, Text = ex.Message });
        }
        finally
        {
            writer.TryComplete();
        }
    }
}

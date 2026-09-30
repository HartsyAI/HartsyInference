using System.Runtime.CompilerServices;
using System.Threading.Channels;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Runs a producer on a worker and yields its chunks; the worker is always cancelled and awaited when the consumer stops early, so an abandoned stream releases whatever the producer holds.</summary>
internal static class TextStreamPump
{
    /// <summary>Chunks buffered before a fast producer waits for a slow consumer.</summary>
    internal const int Capacity = 256;

    /// <summary>Streams <paramref name="produce"/>'s sink output, then the chunks it returns; failures become a trailing Cancelled or Error stop.</summary>
    public static async IAsyncEnumerable<TextChunk> Run(
        Func<Action<TextChunk>, CancellationToken, Task<IReadOnlyList<TextChunk>>> produce,
        [EnumeratorCancellation] CancellationToken cancel = default)
    {
        using CancellationTokenSource linked = CancellationTokenSource.CreateLinkedTokenSource(cancel);
        using CancellationTokenSource abandoned = new();
        Channel<TextChunk> channel = Channel.CreateBounded<TextChunk>(new BoundedChannelOptions(Capacity)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = true,
        });
        Task worker = Task.Run(() => WorkAsync(produce, channel.Writer, linked.Token, abandoned.Token), CancellationToken.None);
        try
        {
            // Read without the caller's token: a cancelled request still ends with its Cancelled stop chunk.
            await foreach (TextChunk chunk in channel.Reader.ReadAllAsync(CancellationToken.None).ConfigureAwait(false))
                yield return chunk;
        }
        finally
        {
            abandoned.Cancel();
            linked.Cancel();
            await worker.ConfigureAwait(false);
        }
    }

    private static async Task WorkAsync(
        Func<Action<TextChunk>, CancellationToken, Task<IReadOnlyList<TextChunk>>> produce,
        ChannelWriter<TextChunk> writer, CancellationToken generation, CancellationToken abandoned)
    {
        // Sink runs on the generation thread; blocking here is the backpressure that keeps a slow consumer from ballooning memory.
        void Sink(TextChunk chunk) => writer.WriteAsync(chunk, generation).AsTask().GetAwaiter().GetResult();
        async Task Terminal(TextChunk chunk)
        {
            try { await writer.WriteAsync(chunk, abandoned).ConfigureAwait(false); }
            catch (OperationCanceledException) { }
        }
        try
        {
            IReadOnlyList<TextChunk> finals = await produce(Sink, generation).ConfigureAwait(false);
            foreach (TextChunk chunk in finals) await Terminal(chunk).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            Logs.Debug("Text stream cancelled.");
            await Terminal(new TextChunk { Kind = TextChunkKind.StopReason, Stop = StopReason.Cancelled }).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            Logs.Error($"Text stream failed: {ex.Message}", ex);
            await Terminal(new TextChunk { Kind = TextChunkKind.StopReason, Stop = StopReason.Error }).ConfigureAwait(false);
        }
        finally
        {
            writer.TryComplete();
        }
    }
}

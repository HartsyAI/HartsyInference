using System.Diagnostics;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>The generation worker must never outlive a consumer that walked away, and must never buffer without bound.</summary>
public sealed class TextStreamPumpTests
{
    private static TextChunk Piece(int i) => new() { Kind = TextChunkKind.Chunk, Text = i.ToString() };

    [Fact]
    public async Task AbandonedConsumerReleasesTheSlotWithinOneSecond()
    {
        using SemaphoreSlim slot = new(0);
        Task<IReadOnlyList<TextChunk>> Produce(Action<TextChunk> sink, CancellationToken ct)
        {
            IReadOnlyList<TextChunk> Loop()
            {
                try
                {
                    for (int i = 0; ; i++)
                    {
                        ct.ThrowIfCancellationRequested();
                        sink(Piece(i));
                    }
                }
                finally { slot.Release(); }
            }
            return Task.Run(Loop);
        }
        IAsyncEnumerator<TextChunk> e = TextStreamPump.Run(Produce).GetAsyncEnumerator();
        Assert.True(await e.MoveNextAsync());
        Stopwatch watch = Stopwatch.StartNew();
        await e.DisposeAsync();
        Assert.True(await slot.WaitAsync(TimeSpan.FromSeconds(1)), "producer still holding its slot after the consumer was disposed");
        Assert.True(watch.Elapsed < TimeSpan.FromSeconds(1));
    }

    [Fact]
    public async Task ProducerBlockedOnAFullChannelIsReleasedWhenTheConsumerStops()
    {
        using SemaphoreSlim slot = new(0);
        int written = 0;
        Task<IReadOnlyList<TextChunk>> Produce(Action<TextChunk> sink, CancellationToken ct)
        {
            return Task.Run<IReadOnlyList<TextChunk>>(() =>
            {
                try
                {
                    for (int i = 0; i < 100_000; i++)
                    {
                        sink(Piece(i));
                        Interlocked.Increment(ref written);
                    }
                    return [];
                }
                finally { slot.Release(); }
            });
        }
        IAsyncEnumerator<TextChunk> e = TextStreamPump.Run(Produce).GetAsyncEnumerator();
        Assert.True(await e.MoveNextAsync());
        await Task.Delay(200);
        Assert.InRange(Volatile.Read(ref written), 1, TextStreamPump.Capacity + 8);
        await e.DisposeAsync();
        Assert.True(await slot.WaitAsync(TimeSpan.FromSeconds(1)));
    }

    [Fact]
    public async Task EveryChunkIsDeliveredInOrderThenTheFinals()
    {
        Task<IReadOnlyList<TextChunk>> Produce(Action<TextChunk> sink, CancellationToken ct)
        {
            for (int i = 0; i < 1000; i++) sink(Piece(i));
            return Task.FromResult<IReadOnlyList<TextChunk>>([new TextChunk { Kind = TextChunkKind.StopReason, Stop = StopReason.Stop }]);
        }
        List<TextChunk> got = [];
        await foreach (TextChunk c in TextStreamPump.Run(Produce)) got.Add(c);
        Assert.Equal(1001, got.Count);
        Assert.Equal(Enumerable.Range(0, 1000).Select(i => i.ToString()), got.Take(1000).Select(c => c.Text));
        Assert.Equal(StopReason.Stop, got[^1].Stop);
    }

    [Fact]
    public async Task CancelledRequestEndsWithACancelledStop()
    {
        using CancellationTokenSource cts = new();
        Task<IReadOnlyList<TextChunk>> Produce(Action<TextChunk> sink, CancellationToken ct)
        {
            sink(Piece(0));
            cts.Cancel();
            ct.ThrowIfCancellationRequested();
            return Task.FromResult<IReadOnlyList<TextChunk>>([]);
        }
        List<TextChunk> got = [];
        await foreach (TextChunk c in TextStreamPump.Run(Produce, cts.Token)) got.Add(c);
        Assert.Equal(StopReason.Cancelled, got[^1].Stop);
    }

    [Fact]
    public async Task ProducerFailureEndsWithAnErrorStop()
    {
        Task<IReadOnlyList<TextChunk>> Produce(Action<TextChunk> sink, CancellationToken ct) => throw new InvalidOperationException("boom");
        List<TextChunk> got = [];
        await foreach (TextChunk c in TextStreamPump.Run(Produce)) got.Add(c);
        Assert.Equal(StopReason.Error, Assert.Single(got).Stop);
    }
}

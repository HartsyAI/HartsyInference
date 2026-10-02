using System.Diagnostics;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>The generation worker must never outlive a consumer that walked away, and must never be stalled by one that reads slowly.</summary>
/// <remarks>Two tests hold the pump to a 100 ms budget, so the class runs on its own rather than beside other classes
/// that keep the thread pool busy.</remarks>
[Collection(TimingSensitiveCollection.Name)]
public sealed class TextStreamPumpTests
{
    private static readonly TimeSpan Budget = TimeSpan.FromMilliseconds(100);

    private static TextChunk Piece(int i) => new() { Kind = TextChunkKind.Chunk, Text = i.ToString() };

    [Fact]
    public async Task AbandonedConsumerCancelsTheWorkerWithin100Ms()
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
                        Thread.Sleep(1);
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
        Assert.True(watch.Elapsed < TimeSpan.FromSeconds(1), $"worker took {watch.ElapsedMilliseconds} ms to stop");
        Assert.True(watch.Elapsed < Budget, $"worker observed cancellation after {watch.ElapsedMilliseconds} ms, budget {Budget.TotalMilliseconds} ms");
    }

    [Fact]
    public async Task ProducerIsNeverBlockedByASlowConsumerAndStillStopsOnAbandonment()
    {
        // Unbounded by decision: the sink runs on the decode thread under the slot lock and device gate, so a
        // slow reader must never stall it. The producer finishes all of its writes while the consumer has read one.
        using SemaphoreSlim slot = new(0);
        const int total = 20_000;
        int written = 0;
        Task<IReadOnlyList<TextChunk>> Produce(Action<TextChunk> sink, CancellationToken ct)
        {
            return Task.Run<IReadOnlyList<TextChunk>>(() =>
            {
                try
                {
                    for (int i = 0; i < total; i++)
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
        Assert.True(await slot.WaitAsync(TimeSpan.FromSeconds(5)), "producer was blocked by the unread channel");
        Assert.Equal(total, Volatile.Read(ref written));
        await e.DisposeAsync();
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
    public async Task PreCancelledTokenStillCompletesWithACancelledStopWithin100Ms()
    {
        using CancellationTokenSource cts = new();
        cts.Cancel();
        Task<IReadOnlyList<TextChunk>> Produce(Action<TextChunk> sink, CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested();
            return Task.FromResult<IReadOnlyList<TextChunk>>([]);
        }
        async Task<List<TextChunk>> Drain()
        {
            List<TextChunk> chunks = [];
            await foreach (TextChunk c in TextStreamPump.Run(Produce, cts.Token)) chunks.Add(c);
            return chunks;
        }
        // The first stream in the process pays for compiling the pump, the channel and the cancellation logging; the
        // budget is for the stream, so that is spent before the clock starts.
        Assert.Equal(StopReason.Cancelled, Assert.Single(await Drain()).Stop);

        Stopwatch watch = Stopwatch.StartNew();
        List<TextChunk> got = await Drain();
        Assert.True(watch.Elapsed < Budget, $"stream took {watch.ElapsedMilliseconds} ms to complete");
        Assert.Equal(StopReason.Cancelled, Assert.Single(got).Stop);
    }

    [Fact]
    public async Task ProducerFailureEndsWithAnErrorStopCarryingTheMessage()
    {
        Task<IReadOnlyList<TextChunk>> Produce(Action<TextChunk> sink, CancellationToken ct) => throw new InvalidOperationException("boom");
        List<TextChunk> got = [];
        await foreach (TextChunk c in TextStreamPump.Run(Produce)) got.Add(c);
        TextChunk only = Assert.Single(got);
        Assert.Equal(StopReason.Error, only.Stop);
        Assert.Equal("boom", only.Text);
    }
}

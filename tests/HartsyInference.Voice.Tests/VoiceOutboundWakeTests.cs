using System.Collections.Concurrent;
using System.Diagnostics;
using HartsyInference.Voice.Turns;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Voice.Tests;

/// <summary>What the reader (the host's 20 ms sender, a thread of its own) pays for a waiting producer: nothing per read
/// while a cancellable playback wait is pending, exactly one wake on the read that reaches the position, one wake per
/// refill for a writer waiting for space, nothing allocated for any of those wakes, and cancellation that still ends a
/// wait at once and withdraws it. Allocation is measured with <see cref="GC.GetAllocatedBytesForCurrentThread"/> on the
/// reader thread only; every producer-side call runs elsewhere, since a wait allocates its waiter by design.</summary>
public sealed class VoiceOutboundWakeTests
{
    private const int Frame = 320;
    private const int Reads = 1_000;

    private readonly ITestOutputHelper _output;

    public VoiceOutboundWakeTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task AnAwaitedPositionCostsTheReaderNothingUntilOneWakeOnTheReadThatReachesIt()
    {
        VoiceOutbound outbound = new(1 << 20, new VoiceTurnSignals());
        float[] reply = new float[Frame * (Reads + 2)];
        long epoch = await outbound.WaitFlushesAppliedAsync(CancellationToken.None);
        Assert.True(await outbound.WriteAsync(reply, 0, reply.Length, epoch, CancellationToken.None));
        using CancellationTokenSource cancel = new();
        using ReaderThread reader = new();
        float[] buffer = new float[Frame];

        // One awaited frame read and woken on the reader thread first, so the measured reads run warm code.
        Task warm = outbound.WaitPlayedAsync(outbound.Consumed + Frame, cancel.Token).AsTask();
        await reader.RunAsync(() => outbound.Read(buffer));
        await warm.WaitAsync(TimeSpan.FromSeconds(5));
        long wakes = outbound.Wakes;

        Task played = outbound.WaitPlayedAsync(outbound.Consumed + (long)Frame * (Reads + 1), cancel.Token).AsTask();
        long pending = await reader.RunAsync(() =>
        {
            for (int i = 0; i < Reads; i++)
            {
                outbound.Read(buffer);
            }
        });
        Assert.False(played.IsCompleted, "the wait ended before the reader reached its position.");
        Assert.Equal(wakes, outbound.Wakes);

        long crossing = await reader.RunAsync(() => outbound.Read(buffer));
        await played.WaitAsync(TimeSpan.FromSeconds(5));
        _output.WriteLine($"reader: {pending} bytes over {Reads} reads with the wait pending, {crossing} bytes on the read that woke it");
        Assert.Equal(wakes + 1, outbound.Wakes);
        Assert.Equal(0, pending);
        Assert.Equal(0, crossing);
    }

    [Fact]
    public async Task AWriterWaitingForSpaceIsWokenOncePerRefillAndTheReaderAllocatesNothing()
    {
        const int Capacity = 4_096;
        VoiceOutbound outbound = new(Capacity, new VoiceTurnSignals());
        float[] reply = new float[Capacity * 4 + Frame * Reads];
        long epoch = await outbound.WaitFlushesAppliedAsync(CancellationToken.None);
        using CancellationTokenSource cancel = new();
        Task<bool> writing = Task.Run(() => outbound.WriteAsync(reply, 0, reply.Length, epoch, cancel.Token).AsTask());
        using ReaderThread reader = new();
        float[] buffer = new float[Frame];
        int readsPerRefill = (outbound.RefillSamples + Frame - 1) / Frame;
        int stalls = 0;

        // Reads only while the writer is parked waiting for space, so each refill is the same number of reads.
        void ReadWhileTheWriterWaits(int count)
        {
            long deadline = Stopwatch.GetTimestamp() + Stopwatch.Frequency * 20;
            for (int i = 0; i < count; i++)
            {
                while (!outbound.ProducerWaiting)
                {
                    if (Stopwatch.GetTimestamp() > deadline)
                    {
                        stalls++;
                        return;
                    }
                    Thread.SpinWait(32);
                }
                outbound.Read(buffer);
            }
        }

        await reader.RunAsync(() => ReadWhileTheWriterWaits(4 * readsPerRefill));
        long wakes = outbound.Wakes;
        long allocated = await reader.RunAsync(() => ReadWhileTheWriterWaits(Reads));
        long woken = outbound.Wakes - wakes;
        _output.WriteLine($"reader: {allocated} bytes over {Reads} reads; {woken} wakes, refill every {outbound.RefillSamples} samples");

        Assert.Equal(0, stalls);
        Assert.Equal(Reads / readsPerRefill, woken);
        Assert.Equal(0, allocated);
        Assert.False(writing.IsCompleted, "the writer ran out of reply before the reads ended; the test proved less than it claims.");
        cancel.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => writing.WaitAsync(TimeSpan.FromSeconds(1)));
    }

    [Fact]
    public async Task AFlushWaitIsWokenOnceOnTheReadThatAppliesIt()
    {
        VoiceTurnSignals signals = new();
        VoiceOutbound outbound = new(256, signals);
        using CancellationTokenSource cancel = new();
        long first = await outbound.WaitFlushesAppliedAsync(cancel.Token);
        signals.RequestFlush();
        Task<long> next = outbound.WaitFlushesAppliedAsync(cancel.Token).AsTask();
        Assert.False(next.IsCompleted);

        float[] buffer = new float[16];
        outbound.Read(buffer);
        Assert.Equal(first + 1, await next.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Equal(1, outbound.Wakes);
        outbound.Read(buffer);
        Assert.Equal(1, outbound.Wakes);
    }

    [Fact]
    public async Task CancellingAnAwaitedPositionEndsTheWaitAtOnceAndWithdrawsIt()
    {
        VoiceOutbound outbound = new(1_024, new VoiceTurnSignals());
        float[] reply = new float[640];
        long epoch = await outbound.WaitFlushesAppliedAsync(CancellationToken.None);
        Assert.True(await outbound.WriteAsync(reply, 0, reply.Length, epoch, CancellationToken.None));
        using CancellationTokenSource cancel = new();
        Task played = outbound.WaitPlayedAsync(outbound.Written, cancel.Token).AsTask();
        Assert.False(played.IsCompleted);

        cancel.Cancel();
        OperationCanceledException cancelled = await Assert.ThrowsAnyAsync<OperationCanceledException>(() => played.WaitAsync(TimeSpan.FromSeconds(1)));
        Assert.Equal(cancel.Token, cancelled.CancellationToken);

        // The withdrawn wait is never completed: reading past its position wakes nothing.
        outbound.Read(new float[reply.Length]);
        Assert.Equal(reply.Length, outbound.Consumed);
        Assert.Equal(0, outbound.Wakes);
    }

    /// <summary>A thread of its own, as the host's sender is: runs one piece of work at a time and reports the managed
    /// bytes that work allocated on it.</summary>
    private sealed class ReaderThread : IDisposable
    {
        private readonly BlockingCollection<(Action Work, TaskCompletionSource<long> Done)> _work = new();
        private readonly Thread _thread;

        public ReaderThread()
        {
            _thread = new Thread(Run) { IsBackground = true, Name = "voice-test-reader" };
            _thread.Start();
        }

        public Task<long> RunAsync(Action work)
        {
            TaskCompletionSource<long> done = new(TaskCreationOptions.RunContinuationsAsynchronously);
            _work.Add((work, done));
            return done.Task;
        }

        public void Dispose()
        {
            _work.CompleteAdding();
            _thread.Join();
            _work.Dispose();
        }

        private void Run()
        {
            foreach ((Action work, TaskCompletionSource<long> done) in _work.GetConsumingEnumerable())
            {
                try
                {
                    long before = GC.GetAllocatedBytesForCurrentThread();
                    work();
                    long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
                    done.SetResult(allocated);
                }
                catch (Exception e)
                {
                    done.SetException(e);
                }
            }
        }
    }
}

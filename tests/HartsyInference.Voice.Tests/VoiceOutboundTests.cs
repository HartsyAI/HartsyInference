using HartsyInference.Voice.Turns;
using Xunit;

namespace HartsyInference.Voice.Tests;

/// <summary>The reply queue's flush protocol on a tiny ring: the reader zero-fills and never blocks, the writer waits for
/// space instead of dropping, a new turn cannot start writing until the reader has applied a pending flush (so the flush
/// cannot eat the next reply's opening), and under concurrent flushes the reader never hears an older turn after a
/// newer one.</summary>
public sealed class VoiceOutboundTests
{
    [Fact]
    public void TheReaderZeroFillsAndCountsOnlyRealSamples()
    {
        VoiceOutbound outbound = new(64, new VoiceTurnSignals());
        float[] buffer = new float[16];
        Array.Fill(buffer, 5f);
        Assert.Equal(0, outbound.Read(buffer));
        Assert.All(buffer, sample => Assert.Equal(0f, sample));
    }

    [Fact]
    public async Task TheWriterWaitsForSpaceInsteadOfDropping()
    {
        VoiceOutbound outbound = new(64, new VoiceTurnSignals());
        float[] reply = new float[1_000];
        for (int i = 0; i < reply.Length; i++)
        {
            reply[i] = i;
        }
        long epoch = await outbound.WaitFlushesAppliedAsync(CancellationToken.None);
        Task<bool> writing = outbound.WriteAsync(reply, 0, reply.Length, epoch, CancellationToken.None).AsTask();
        List<float> heard = [];
        float[] buffer = new float[24];
        DateTime deadline = DateTime.UtcNow.AddSeconds(10);
        while (heard.Count < reply.Length && DateTime.UtcNow < deadline)
        {
            int real = outbound.Read(buffer);
            heard.AddRange(buffer.AsSpan(0, real));
            await Task.Yield();
        }
        Assert.True(await writing.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Equal(reply, heard);
        Assert.Equal(reply.Length, outbound.Written);
        Assert.Equal(reply.Length, outbound.Consumed);
    }

    [Fact]
    public async Task ANewTurnWaitsUntilThePendingFlushIsApplied()
    {
        VoiceTurnSignals signals = new();
        VoiceOutbound outbound = new(256, signals);
        float[] stale = [1f, 1f, 1f, 1f];
        float[] fresh = [2f, 2f, 2f, 2f];
        long first = await outbound.WaitFlushesAppliedAsync(CancellationToken.None);
        Assert.True(await outbound.WriteAsync(stale, 0, stale.Length, first, CancellationToken.None));

        signals.RequestFlush();
        Task<long> next = outbound.WaitFlushesAppliedAsync(CancellationToken.None).AsTask();
        await Task.Delay(50);
        Assert.False(next.IsCompleted, "a new turn started writing before the reader applied the flush.");
        Assert.False(await outbound.WriteAsync(stale, 0, stale.Length, first, CancellationToken.None));

        float[] buffer = new float[8];
        Assert.Equal(0, outbound.Read(buffer));
        long second = await next.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(first + 1, second);
        Assert.True(await outbound.WriteAsync(fresh, 0, fresh.Length, second, CancellationToken.None));
        Assert.Equal(fresh.Length, outbound.Read(buffer));
        Assert.Equal(fresh, buffer[..fresh.Length]);
        Assert.Equal(stale.Length, outbound.DiscardedSamples);
    }

    [Fact]
    public async Task PlaybackCompletesWhenTheReaderPassesThePosition()
    {
        VoiceOutbound outbound = new(64, new VoiceTurnSignals());
        float[] reply = new float[40];
        long epoch = await outbound.WaitFlushesAppliedAsync(CancellationToken.None);
        Assert.True(await outbound.WriteAsync(reply, 0, reply.Length, epoch, CancellationToken.None));
        Task played = outbound.WaitPlayedAsync(outbound.Written, CancellationToken.None).AsTask();
        float[] buffer = new float[16];
        outbound.Read(buffer);
        Assert.False(played.IsCompleted);
        outbound.Read(buffer);
        outbound.Read(buffer);
        await played.WaitAsync(TimeSpan.FromSeconds(5));
    }

    [Fact]
    public async Task UnderConcurrentFlushesTheReaderNeverHearsAnOlderTurnAfterANewerOne()
    {
        VoiceTurnSignals signals = new();
        VoiceOutbound outbound = new(128, signals);
        using CancellationTokenSource stop = new(TimeSpan.FromSeconds(30));
        const int Turns = 300;
        bool producerDone = false;

        Task producer = Task.Run(async () =>
        {
            float[] frame = new float[32];
            for (int turn = 1; turn <= Turns; turn++)
            {
                Array.Fill(frame, turn);
                long epoch = await outbound.WaitFlushesAppliedAsync(stop.Token);
                for (int i = 0; i < 20; i++)
                {
                    if (!await outbound.WriteAsync(frame, 0, frame.Length, epoch, stop.Token))
                    {
                        break;
                    }
                }
            }
            Volatile.Write(ref producerDone, true);
        });
        Task flusher = Task.Run(() =>
        {
            Random random = new(1234);
            while (!Volatile.Read(ref producerDone) && !stop.IsCancellationRequested)
            {
                signals.RequestFlush();
                Thread.SpinWait(random.Next(200, 20_000));
            }
        });
        List<float> heard = [];
        float[] buffer = new float[40];
        while (!producer.IsCompleted || outbound.Queued > 0)
        {
            int real = outbound.Read(buffer);
            heard.AddRange(buffer.AsSpan(0, real));
            stop.Token.ThrowIfCancellationRequested();
        }
        await producer;
        await flusher;

        for (int i = 1; i < heard.Count; i++)
        {
            Assert.True(heard[i] >= heard[i - 1], $"turn {heard[i - 1]} was heard before turn {heard[i]} at sample {i}.");
        }
        Assert.NotEmpty(heard);
        Assert.True(outbound.DiscardedSamples > 0, "the flusher never overlapped a turn; the test proved nothing.");
    }
}

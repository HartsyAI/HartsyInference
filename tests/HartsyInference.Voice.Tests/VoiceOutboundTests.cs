using HartsyInference.Voice.Turns;
using Xunit;

namespace HartsyInference.Voice.Tests;

/// <summary>The reply queue's flush protocol on a tiny ring: the reader zero-fills and never blocks, the writer waits for
/// space instead of dropping, a new turn cannot start writing until the reader has applied a pending flush (so the flush
/// cannot eat the next reply's opening), and under concurrent flushes the reader never hears an older turn after a
/// newer one. The tagged read returns one turn's samples at a time, each labelled with the turn that wrote it.</summary>
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

    [Fact]
    public async Task ATaggedReadStopsWhereTheNextTurnBegins()
    {
        VoiceOutbound outbound = new(256, new VoiceTurnSignals());
        await WriteTurnAsync(outbound, turnId: 1, count: 10);
        await WriteTurnAsync(outbound, turnId: 2, count: 12);
        float[] buffer = new float[64];

        Assert.Equal(10, outbound.Read(buffer, out int first));
        Assert.Equal(1, first);
        Assert.All(buffer[..10], sample => Assert.Equal(1f, sample));
        Assert.All(buffer[10..], sample => Assert.Equal(0f, sample));
        Assert.Equal(12, outbound.Read(buffer, out int second));
        Assert.Equal(2, second);
        Assert.All(buffer[..12], sample => Assert.Equal(2f, sample));
        Assert.Equal(0, outbound.Read(buffer, out int none));
        Assert.Equal(0, none);
    }

    [Fact]
    public async Task TheUntaggedReadStillCrossesTurnBoundaries()
    {
        VoiceOutbound outbound = new(256, new VoiceTurnSignals());
        await WriteTurnAsync(outbound, turnId: 1, count: 10);
        await WriteTurnAsync(outbound, turnId: 2, count: 12);
        float[] buffer = new float[64];

        Assert.Equal(22, outbound.Read(buffer));
        Assert.Equal([.. Enumerable.Repeat(1f, 10), .. Enumerable.Repeat(2f, 12)], buffer[..22]);
    }

    [Fact]
    public async Task AFlushAcrossATurnBoundaryDropsBothAndTheNextTurnIsTaggedAsItself()
    {
        VoiceTurnSignals signals = new();
        VoiceOutbound outbound = new(256, signals);
        await WriteTurnAsync(outbound, turnId: 1, count: 10);
        await WriteTurnAsync(outbound, turnId: 2, count: 12);
        signals.RequestFlush();
        Task third = WriteTurnAsync(outbound, turnId: 3, count: 8);
        float[] buffer = new float[64];

        Assert.Equal(0, outbound.Read(buffer, out int none));
        Assert.Equal(0, none);
        await third.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(8, outbound.Read(buffer, out int turn));
        Assert.Equal(3, turn);
        Assert.All(buffer[..8], sample => Assert.Equal(3f, sample));
        Assert.Equal(22, outbound.DiscardedSamples);
    }

    [Fact]
    public async Task ATurnThatWroteNothingLeavesNoTag()
    {
        VoiceOutbound outbound = new(256, new VoiceTurnSignals());
        await outbound.BeginTurnAsync(1, CancellationToken.None);
        await WriteTurnAsync(outbound, turnId: 2, count: 6);
        float[] buffer = new float[16];

        Assert.Equal(6, outbound.Read(buffer, out int turn));
        Assert.Equal(2, turn);
    }

    [Fact]
    public async Task MoreTurnsThanMarkSlotsWaitForTheReaderInsteadOfOverwritingTags()
    {
        VoiceOutbound outbound = new(1_024, new VoiceTurnSignals());
        Task producer = Task.Run(async () =>
        {
            for (int turn = 1; turn <= 40; turn++)
            {
                await WriteTurnAsync(outbound, turn, count: 4);
            }
        });
        float[] buffer = new float[64];
        List<(int Turn, float Sample)> heard = [];
        DateTime deadline = DateTime.UtcNow.AddSeconds(10);
        while ((!producer.IsCompleted || outbound.Queued > 0) && DateTime.UtcNow < deadline)
        {
            int real = outbound.Read(buffer, out int turn);
            for (int i = 0; i < real; i++)
            {
                heard.Add((turn, buffer[i]));
            }
            await Task.Yield();
        }
        await producer.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(40 * 4, heard.Count);
        Assert.All(heard, pair => Assert.Equal(pair.Turn, (int)pair.Sample));
    }

    [Fact]
    public async Task UnderConcurrentFlushesEveryTaggedSampleCarriesTheTurnThatWroteIt()
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
                long epoch = await outbound.BeginTurnAsync(turn, stop.Token);
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
            Random random = new(4321);
            while (!Volatile.Read(ref producerDone) && !stop.IsCancellationRequested)
            {
                signals.RequestFlush();
                Thread.SpinWait(random.Next(200, 20_000));
            }
        });
        int lastTurn = 0;
        long samples = 0;
        float[] buffer = new float[40];
        while (!producer.IsCompleted || outbound.Queued > 0)
        {
            int real = outbound.Read(buffer, out int turn);
            for (int i = 0; i < real; i++)
            {
                Assert.Equal(turn, (int)buffer[i]);
            }
            if (real > 0)
            {
                Assert.True(turn >= lastTurn, $"turn {turn} was read after turn {lastTurn}.");
                lastTurn = turn;
                samples += real;
            }
            stop.Token.ThrowIfCancellationRequested();
        }
        await producer;
        await flusher;

        Assert.True(samples > 0);
        Assert.True(outbound.DiscardedSamples > 0, "the flusher never overlapped a turn; the test proved nothing.");
    }

    private static async Task WriteTurnAsync(VoiceOutbound outbound, int turnId, int count)
    {
        float[] samples = new float[count];
        Array.Fill(samples, turnId);
        long epoch = await outbound.BeginTurnAsync(turnId, CancellationToken.None);
        Assert.True(await outbound.WriteAsync(samples, 0, samples.Length, epoch, CancellationToken.None));
    }
}

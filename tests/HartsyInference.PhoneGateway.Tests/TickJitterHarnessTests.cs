using HartsyInference.Core.Numerics;
using HartsyInference.PhoneGateway.Media;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.PhoneGateway.Tests;

/// <summary>Sixty seconds of the RTP clock with every core burning. The p99 &lt; 2 ms / max &lt; 10 ms bound is the
/// plan's claim for the SCHED_FIFO configuration only, so it is asserted only when FIFO was granted; without it
/// (an <c>rtprio</c> limit of 0, as on a plain developer login) the numbers are logged and the test returns, in the
/// early-return style of <c>BackendGate.TryOpen</c>.</summary>
[Trait("Category", "Integration")]
public sealed class TickJitterHarnessTests
{
    private const int DurationMs = 60_000;

    private readonly ITestOutputHelper _output;
    public TickJitterHarnessTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task SixtySecondsUnderFullLoad_HoldsTheCadence()
    {
        int burnerCount = Environment.ProcessorCount;
        Burner burner = new(burnerCount);
        using ClockedAudioSource source = new(new ClockedAudioSourceOptions { FifoPriority = 50 });
        long frames = 0;
        source.OnAudioSourceEncodedSample += (_, _) => Interlocked.Increment(ref frames);
        burner.Start();
        try
        {
            source.Start();
            await Task.Delay(DurationMs);
            source.Stop();
        }
        finally
        {
            burner.Stop();
        }
        LatencyHistogram.Summary late = source.Lateness.Snapshot();
        long expected = source.ElapsedNs / ClockedAudioSource.PeriodNs;
        _output.WriteLine(
            $"burners={burnerCount} fifo={source.FifoActive} frames={Interlocked.Read(ref frames)} expected={expected} " +
            $"catchUp={source.CatchUpFrames} resyncs={source.Resyncs} allocated={source.TickThreadAllocatedBytes} " +
            $"lateness mean={late.MeanUs}us p50={late.P50Us}us p99={late.P99Us}us max={late.MaxUs}us");
        Assert.False(source.Faulted);
        Assert.Equal(0, source.TickThreadAllocatedBytes);
        Assert.InRange(Interlocked.Read(ref frames), expected - 1, expected + 1 + source.CatchUpFrames);
        if (!source.FifoActive)
        {
            _output.WriteLine($"SKIPPED the FIFO bound: {source.FifoReason}");
            return;
        }
        Assert.True(late.P99Us < 2000, $"p99 {late.P99Us} us under FIFO");
        Assert.True(late.MaxUs < 10_000, $"max {late.MaxUs} us under FIFO");
    }

    /// <summary>One spinning thread per core, ordinary priority: the load the tick thread must out-schedule.</summary>
    private sealed class Burner(int count)
    {
        private readonly Thread[] _threads = new Thread[count];
        private volatile bool _run = true;

        public void Start()
        {
            for (int i = 0; i < _threads.Length; i++)
            {
                _threads[i] = new Thread(Spin) { IsBackground = true, Name = "burner-" + i };
                _threads[i].Start();
            }
        }

        public void Stop()
        {
            _run = false;
            foreach (Thread thread in _threads)
            {
                thread.Join();
            }
        }

        private void Spin()
        {
            uint x = 1;
            while (_run)
            {
                x = x * 1103515245u + 12345u;
            }
            GC.KeepAlive(x);
        }
    }
}

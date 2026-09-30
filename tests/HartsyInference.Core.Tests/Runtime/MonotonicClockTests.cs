using System.Diagnostics;
using HartsyInference.Core.Runtime;
using Xunit;

namespace HartsyInference.Core.Tests.Runtime;

/// <summary>The clock a tick thread schedules against: it must not run backwards, a sleep to a deadline must not
/// return early, and neither call may allocate, because a garbage collection is the one stall a real-time thread
/// cannot be scheduled around.</summary>
public sealed class MonotonicClockTests
{
    [Fact]
    public void NowNs_NeverDecreases()
    {
        long previous = MonotonicClock.NowNs();
        for (int i = 0; i < 10_000; i++)
        {
            long now = MonotonicClock.NowNs();
            Assert.True(now >= previous, $"clock went backwards: {previous} -> {now}");
            previous = now;
        }
    }

    [Fact]
    public void NowNs_TracksWallElapsedTime()
    {
        long startNs = MonotonicClock.NowNs();
        Stopwatch clock = Stopwatch.StartNew();
        Thread.SpinWait(5_000_000);
        long elapsedNs = MonotonicClock.NowNs() - startNs;
        double stopwatchNs = clock.Elapsed.TotalMilliseconds * 1_000_000;
        // Both clocks are monotonic; they can only differ by the few microseconds between the two reads.
        Assert.InRange(elapsedNs, stopwatchNs * 0.5, stopwatchNs * 2 + 2_000_000);
    }

    [Fact]
    public void SleepUntil_DoesNotReturnBeforeTheDeadline()
    {
        const long WaitNs = 3_000_000;
        long deadline = MonotonicClock.NowNs() + WaitNs;
        MonotonicClock.SleepUntil(deadline);
        long now = MonotonicClock.NowNs();
        Assert.True(now >= deadline, $"woke {deadline - now} ns early");
        Assert.True(now - deadline < 500_000_000, $"woke {(now - deadline) / 1_000_000} ms late");
    }

    [Fact]
    public void SleepUntil_PastDeadline_ReturnsAtOnce()
    {
        long before = MonotonicClock.NowNs();
        MonotonicClock.SleepUntil(before - 1_000_000_000);
        Assert.True(MonotonicClock.NowNs() - before < 100_000_000);
    }

    [Fact]
    public void NowNs_And_SleepUntil_DoNotAllocate()
    {
        // Warm up: the first call through a P/Invoke stub allocates its marshalling state once.
        MonotonicClock.SleepUntil(MonotonicClock.NowNs() + 100_000);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1_000; i++)
        {
            MonotonicClock.NowNs();
        }
        for (int i = 0; i < 10; i++)
        {
            MonotonicClock.SleepUntil(MonotonicClock.NowNs() + 50_000);
        }
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }
}

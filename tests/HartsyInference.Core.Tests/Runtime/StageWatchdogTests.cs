using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Runtime;
using Xunit;

namespace HartsyInference.Core.Tests.Runtime;

/// <summary>Unit lane: the watchdog with a fake clock and no timer, so every transition is driven by the test.</summary>
public sealed class StageWatchdogTests
{
    private sealed class Clock
    {
        public double Now;
    }

    private static StageWatchdog Make(Clock clock, Action<StageStalledException>? onStall = null, double minimum = 120) =>
        new("flow", () => (3L << 30, 12L << 30), 10, TimeSpan.FromSeconds(minimum), onStall, TimeSpan.Zero, () => clock.Now);

    [Fact]
    public void SilenceUnderTheLimitIsNotAStall()
    {
        Clock clock = new();
        using StageWatchdog watchdog = Make(clock);
        clock.Now = 119;

        watchdog.Poll();
        watchdog.Step();
    }

    [Fact]
    public void ALongStepRaisesTheStallOnceAndTheNextStepThrowsItTyped()
    {
        Clock clock = new();
        List<StageStalledException> raised = [];
        using StageWatchdog watchdog = Make(clock, raised.Add);
        clock.Now = 12;
        watchdog.Step();
        clock.Now = 24;
        watchdog.Step();

        clock.Now = 24 + 121;
        watchdog.Poll();
        watchdog.Poll();
        StageStalledException thrown = Assert.Throws<StageStalledException>(watchdog.Step);

        Assert.Single(raised);
        Assert.Same(raised[0], thrown);
        Assert.Equal("flow", thrown.Stage);
        Assert.Equal(121, thrown.StalledSeconds);
        Assert.Equal(12, thrown.ExpectedStepSeconds);
        Assert.Equal((3L << 30, 12L << 30), (thrown.FreeBytes, thrown.TotalBytes));
        Assert.Contains("numerics.flowF16Weights", thrown.Message);
    }

    [Fact]
    public void TheLimitFollowsTheMedianStepWhenThatIsLongerThanTheFloor()
    {
        Clock clock = new();
        using StageWatchdog watchdog = Make(clock, minimum: 5);
        foreach (double end in new[] { 30.0, 60.0, 90.0 })
        {
            clock.Now = end;
            watchdog.Step();
        }

        clock.Now = 90 + 299;
        watchdog.Poll();
        watchdog.Step();
        clock.Now += 301;
        watchdog.Poll();

        Assert.Throws<StageStalledException>(watchdog.Step);
    }

    [Fact]
    public void ADisposedWatchdogRecordsNothing()
    {
        Clock clock = new();
        List<StageStalledException> raised = [];
        StageWatchdog watchdog = Make(clock, raised.Add);
        watchdog.Dispose();
        clock.Now = 10_000;

        watchdog.Poll();

        Assert.Empty(raised);
    }

    [Fact]
    public void AFactorBelowOneIsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new StageWatchdog("flow", stallFactor: 0.5));
    }
}

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

    private static StageWatchdog Make(Clock clock, Action<StageStalledException>? onStall = null, double minimum = 120, Func<(long, long)>? memory = null) =>
        new("flow", memory ?? (() => (3L << 30, 12L << 30)), 10, TimeSpan.FromSeconds(minimum), "Try X.", onStall, TimeSpan.Zero, () => clock.Now);

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
        Assert.Same(raised[0], thrown.InnerException);
        Assert.Equal("flow", thrown.Stage);
        Assert.Equal(121, thrown.StalledSeconds);
        Assert.Equal(12, thrown.ExpectedStepSeconds);
        Assert.Equal((3L << 30, 12L << 30), (thrown.FreeBytes, thrown.TotalBytes));
        Assert.EndsWith("Try X.", thrown.Message);
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
    public void ASlowFirstStepOnlyWarnsAndALaterStallStillThrows()
    {
        Clock clock = new();
        List<StageStalledException> raised = [];
        using StageWatchdog watchdog = Make(clock, raised.Add);
        clock.Now = 130;
        watchdog.Poll();

        watchdog.Step();
        foreach (double end in new[] { 131.0, 132.0, 133.0 })
        {
            clock.Now = end;
            watchdog.Step();
        }

        clock.Now = 133 + 121;
        watchdog.Poll();

        Assert.Equal(2, raised.Count);
        Assert.Throws<StageStalledException>(watchdog.Step);
    }

    [Fact]
    public void ASlowFirstStepDoesNotStretchTheLimitOfTheStepsAfter()
    {
        Clock clock = new();
        List<StageStalledException> raised = [];
        using StageWatchdog watchdog = Make(clock, raised.Add);
        clock.Now = 300;
        watchdog.Step();
        clock.Now = 310;
        watchdog.Step();

        clock.Now = 310 + 121;
        watchdog.Poll();

        Assert.Single(raised);
        Assert.Equal(10, raised[0].ExpectedStepSeconds);
    }

    [Fact]
    public void ADeviceThatReportsNoMemoryIsNotWatched()
    {
        Clock clock = new();
        List<StageStalledException> raised = [];
        using StageWatchdog watchdog = Make(clock, raised.Add, memory: () => (0, 0));
        clock.Now = 10_000;

        watchdog.Poll();
        watchdog.Step();

        Assert.Empty(raised);
    }

    [Fact]
    public void PollNeverCallsTheMemoryDelegateSoABackendLockCannotBlockIt()
    {
        Clock clock = new();
        int calls = 0;
        using StageWatchdog watchdog = Make(clock, memory: () =>
        {
            calls++;
            return (3L << 30, 12L << 30);
        });
        int afterConstruction = calls;
        clock.Now = 500;

        watchdog.Poll();

        Assert.Equal(afterConstruction, calls);
    }

    [Fact]
    public void AnEvenNumberOfStepsUsesTheMeanOfTheMiddleTwo()
    {
        Clock clock = new();
        List<StageStalledException> raised = [];
        using StageWatchdog watchdog = Make(clock, raised.Add, minimum: 1);
        foreach (double end in new[] { 100.0, 110.0, 130.0, 140.0, 160.0 })
        {
            clock.Now = end;
            watchdog.Step();
        }

        clock.Now = 160 + 149;
        watchdog.Poll();
        Assert.Empty(raised);
        clock.Now = 160 + 151;
        watchdog.Poll();

        Assert.Single(raised);
        Assert.Equal(15, raised[0].ExpectedStepSeconds);
    }

    [Fact]
    public void AStallCallbackThatThrowsDoesNotEscapeThePollAndTheStallStillEndsTheRun()
    {
        Clock clock = new();
        int calls = 0;
        using StageWatchdog watchdog = Make(clock, _ =>
        {
            calls++;
            throw new InvalidOperationException("boom");
        });
        clock.Now = 12;
        watchdog.Step();
        clock.Now = 24;
        watchdog.Step();

        clock.Now = 24 + 121;
        watchdog.Poll();

        Assert.Equal(1, calls);
        Assert.Throws<StageStalledException>(watchdog.Step);
    }

    [Fact]
    public void AFactorBelowOneIsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new StageWatchdog("flow", stallFactor: 0.5));
    }
}

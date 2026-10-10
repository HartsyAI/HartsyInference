using HartsyInference.Core.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Runtime;

/// <summary>The scheduling calls must never throw and must always explain a refusal, because the one thing a
/// service operator has to do about a missed real-time cadence is read that reason. Every call here runs on a
/// throwaway thread: a success (root, or an <c>rtprio</c> limit that has been granted) would otherwise leave an
/// xunit worker thread in <c>SCHED_FIFO</c> or pinned to one core for the rest of the run.</summary>
public sealed class RealtimeSchedulingTests
{
    private readonly ITestOutputHelper _output;
    public RealtimeSchedulingTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void TryEnterFifo_OnItsOwnThread_EitherSucceedsOrExplains()
    {
        (bool ok, string reason) = OnThrowawayThread(() =>
        {
            bool result = RealtimeScheduling.TryEnterFifo(50, out string why);
            return (result, why);
        });
        _output.WriteLine($"TryEnterFifo(50) -> {ok}: {reason}");
        if (ok)
        {
            Assert.Equal("", reason);
            return;
        }
        Assert.False(string.IsNullOrWhiteSpace(reason));
        if (OperatingSystem.IsLinux())
        {
            Assert.Contains("LimitRTPRIO", reason, StringComparison.Ordinal);
            Assert.Contains("limits.d", reason, StringComparison.Ordinal);
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(100)]
    public void TryEnterFifo_RejectsPrioritiesOutsideTheLinuxRange(int priority)
    {
        bool ok = RealtimeScheduling.TryEnterFifo(priority, out string reason);
        Assert.False(ok);
        Assert.Contains(priority.ToString(), reason, StringComparison.Ordinal);
    }

    [Fact]
    public void TryPinToCpu_RejectsAnIndexOutsideTheMask()
    {
        Assert.False(RealtimeScheduling.TryPinToCpu(-1, out string low));
        Assert.Contains("-1", low, StringComparison.Ordinal);
        Assert.False(RealtimeScheduling.TryPinToCpu(RealtimeScheduling.MaxCpu + 1, out string high));
        Assert.Contains((RealtimeScheduling.MaxCpu + 1).ToString(), high, StringComparison.Ordinal);
    }

    private static T OnThrowawayThread<T>(Func<T> work)
    {
        T result = default!;
        Exception? failure = null;
        Thread thread = new(() =>
        {
            try
            {
                result = work();
            }
            catch (Exception ex)
            {
                failure = ex;
            }
        })
        {
            IsBackground = true,
            Name = "realtime-scheduling-test",
        };
        thread.Start();
        Assert.True(thread.Join(TimeSpan.FromSeconds(10)), "the scheduling call did not return");
        if (failure is not null)
        {
            throw new InvalidOperationException("the scheduling call threw", failure);
        }
        return result;
    }
}

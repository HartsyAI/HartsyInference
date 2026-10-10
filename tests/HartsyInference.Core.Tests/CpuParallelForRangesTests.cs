using System.Collections.Concurrent;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tests.MemoryManagement;
using Xunit;
using static HartsyInference.Tests.Common.CpuSchedules;

namespace HartsyInference.Core.Tests;

/// <summary><see cref="CpuParallel.ForRanges(long, long, long, Action{long, long})"/> cuts a count into ranges that
/// depend on the count and the range length alone: they tile it exactly with the last one short, and they are the same
/// ranges whether the work fans out over every core, runs under a <c>numerics.cpuThreads</c> cap or runs inline, which
/// is what keeps the casts and reductions built on it byte-identical at any cap. Serialized with the other classes
/// that change process-wide knobs.</summary>
[Collection(EnvironmentSensitiveCollection.Name)]
public sealed class CpuParallelForRangesTests
{
    private const long RangeLength = 1000;
    private const long Count = 37 * RangeLength + 123;

    /// <summary>Enough work per element that every call fans out whenever the schedule lets it.</summary>
    private const long HeavyWork = CpuParallel.MinWorkForParallel;

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void TheRangesTileTheCountExactly_WithTheLastOneShort_AtEveryCap(int cap)
    {
        List<(long Start, long Length)> ranges = WithCpuThreads(cap, () => Record(Count, RangeLength));

        Assert.Equal(38, ranges.Count);
        long next = 0;
        foreach ((long start, long length) in ranges)
        {
            Assert.Equal(next, start);
            Assert.Equal(start + RangeLength <= Count ? RangeLength : Count - start, length);
            next = start + length;
        }
        Assert.Equal(Count, next);
        Assert.Equal(123L, ranges[^1].Length);
    }

    [Fact]
    public void TheRangesAreTheSame_UnderEverySchedule()
    {
        (List<(long, long)> parallel, List<(long, long)> capped, List<(long, long)> inline) =
            UnderEverySchedule(() => Record(Count, RangeLength));

        Assert.Equal(parallel, capped);
        Assert.Equal(parallel, inline);
    }

    [Fact]
    public void ACountOfZero_RunsNothing_AndACountUnderOneRange_IsOneCallOnTheCallingThread()
    {
        Assert.Empty(Record(0, RangeLength));

        int caller = Environment.CurrentManagedThreadId;
        int ranOn = -1, calls = 0;
        (long Start, long Length) only = (-1, -1);
        CpuParallel.ForRanges(RangeLength - 1, RangeLength, HeavyWork, (start, length) =>
        {
            calls++;
            only = (start, length);
            ranOn = Environment.CurrentManagedThreadId;
        });
        Assert.Equal(1, calls);
        Assert.Equal((0L, RangeLength - 1), only);
        Assert.Equal(caller, ranOn);
    }

    [Fact]
    public void BadArguments_AreRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => CpuParallel.ForRanges(-1, RangeLength, 1, static (_, _) => { }));
        Assert.Throws<ArgumentOutOfRangeException>(() => CpuParallel.ForRanges(Count, 0, 1, static (_, _) => { }));
        Assert.Throws<ArgumentOutOfRangeException>(() => CpuParallel.ForRanges(Count, -5, 1, static (_, _) => { }));
        Assert.Throws<ArgumentNullException>(() => CpuParallel.ForRanges(Count, RangeLength, 1, null!));
        Assert.Throws<ArgumentNullException>(() => CpuParallel.ForRanges(Count, RangeLength, 1, 0, (Action<long, long, int>)null!));
    }

    [Fact]
    public void AFailingRange_IsRethrownAsItself_NotWrapped()
    {
        InvalidOperationException thrown = WithCpuThreads(0, () => Assert.Throws<InvalidOperationException>(() =>
            CpuParallel.ForRanges(Count, RangeLength, HeavyWork, static (start, _) =>
            {
                if (start == 17 * RangeLength) throw new InvalidOperationException("range 17");
            })));

        Assert.Equal("range 17", thrown.Message);
    }

    private static List<(long Start, long Length)> Record(long count, long rangeLength)
    {
        ConcurrentBag<(long Start, long Length)> calls = [];
        CpuParallel.ForRanges(count, rangeLength, HeavyWork, (start, length) => calls.Add((start, length)));
        return [.. calls.OrderBy(c => c.Start)];
    }
}

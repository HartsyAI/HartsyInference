using HartsyInference.Core.Numerics;
using Xunit;

namespace HartsyInference.Core.Tests.Numerics;

/// <summary>The lateness histogram is what a tick thread's "p99 under 2 ms" claim rests on, so the bucket edges
/// and the percentile rule are checked against arrays worked out by hand rather than against the class itself.</summary>
public sealed class LatencyHistogramTests
{
    [Fact]
    public void Edges_AreTheFixedMicrosecondBounds()
    {
        Assert.Equal(new long[] { 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000 },
            LatencyHistogram.UpperBoundsUs.ToArray());
        Assert.Equal(11, LatencyHistogram.BucketCount);
    }

    [Fact]
    public void Record_LandsEachSampleInTheBucketWhoseUpperBoundItDoesNotExceed()
    {
        LatencyHistogram histogram = new();
        histogram.Record(40_000);        // 40 µs  -> ≤50
        histogram.Record(50_000);        // 50 µs  -> ≤50 (inclusive upper bound)
        histogram.Record(50_001);        // just over -> ≤100
        histogram.Record(150_000);       // 150 µs -> ≤200
        histogram.Record(3_000_000);     // 3 ms   -> ≤5000
        histogram.Record(80_000_000);    // 80 ms  -> open-ended
        histogram.Record(-7);            // negative counts as zero -> ≤50

        long[] counts = new long[LatencyHistogram.BucketCount];
        histogram.CopyCounts(counts);
        Assert.Equal(new long[] { 3, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1 }, counts);
        Assert.Equal(7, histogram.Count);
        Assert.Equal(80_000_000, histogram.MaxNs);
    }

    [Fact]
    public void Percentiles_ReportTheUpperBoundOfTheBucketThatReachesTheRank()
    {
        LatencyHistogram histogram = new();
        long[] samplesNs = [40_000, 60_000, 150_000, 3_000_000, 80_000_000];
        foreach (long ns in samplesNs)
        {
            histogram.Record(ns);
        }

        // Ranks: p50 of 5 = ceil(2.5) = 3rd sample -> 150 µs -> bucket ≤200.
        Assert.Equal(200, histogram.PercentileUs(0.50));
        // p99 of 5 = ceil(4.95) = 5th sample -> open-ended bucket -> the max itself (80 000 µs).
        Assert.Equal(80_000, histogram.PercentileUs(0.99));
        Assert.Equal(50, histogram.PercentileUs(0.20));
        Assert.Equal(100, histogram.PercentileUs(0.40));

        LatencyHistogram.Summary summary = histogram.Snapshot();
        Assert.Equal(5, summary.Count);
        Assert.Equal(200, summary.P50Us);
        Assert.Equal(80_000, summary.P99Us);
        Assert.Equal(80_000, summary.MaxUs);
        Assert.Equal((40_000 + 60_000 + 150_000 + 3_000_000 + 80_000_000) / 5 / 1_000, summary.MeanUs);
    }

    [Fact]
    public void Percentiles_OfAnEmptyHistogram_AreZero()
    {
        LatencyHistogram histogram = new();
        Assert.Equal(0, histogram.PercentileUs(0.5));
        LatencyHistogram.Summary summary = histogram.Snapshot();
        Assert.Equal(new LatencyHistogram.Summary(0, 0, 0, 0, 0), summary);
    }

    [Fact]
    public void Percentile_RejectsFractionsOutsideZeroOne()
    {
        LatencyHistogram histogram = new();
        Assert.Throws<ArgumentOutOfRangeException>(() => histogram.PercentileUs(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => histogram.PercentileUs(1.5));
    }

    [Fact]
    public void Reset_ClearsEverything()
    {
        LatencyHistogram histogram = new();
        histogram.Record(1_000_000);
        histogram.Record(9_000_000);
        histogram.Reset();
        Assert.Equal(0, histogram.Count);
        Assert.Equal(0, histogram.MaxNs);
        long[] counts = new long[LatencyHistogram.BucketCount];
        histogram.CopyCounts(counts);
        Assert.All(counts, c => Assert.Equal(0, c));
    }

    /// <summary>The least any of a few identical passes allocated. An allocation inside <see cref="LatencyHistogram.Record"/>
    /// recurs on every call, so it shows in every pass; a one-off the runtime makes on this thread while the test runs
    /// lands in one pass only and is not the histogram's.</summary>
    [Fact]
    public void Record_DoesNotAllocate()
    {
        LatencyHistogram histogram = new();
        histogram.Record(1);
        long least = long.MaxValue;
        for (int pass = 0; pass < 3; pass++)
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 100_000; i++)
            {
                histogram.Record(i * 1_000L);
            }
            least = Math.Min(least, GC.GetAllocatedBytesForCurrentThread() - before);
        }
        Assert.Equal(0, least);
    }
}

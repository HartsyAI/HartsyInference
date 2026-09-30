namespace HartsyInference.Core.Numerics;

/// <summary>Fixed-bucket lateness histogram for a periodic thread, recorded without allocation.</summary>
/// <remarks>Buckets are bounded above (in microseconds) at 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000 and
/// 50000, with an eleventh bucket for everything later. <see cref="Record"/> is written for exactly one thread — the
/// tick thread itself — and touches nothing but its own counters; a percentile read from another thread while it is
/// recording is a diagnostic snapshot, not a transaction. Percentiles are reported as the upper bound of the bucket
/// that reaches them, so "p99 = 2000" means the 99th percentile was late by at most two milliseconds; when the
/// percentile lands in the open-ended bucket the maximum observed value is reported instead.</remarks>
public sealed class LatencyHistogram
{
    /// <summary>Number of buckets, including the open-ended last one.</summary>
    public const int BucketCount = 11;

    private const long NsPerMicrosecond = 1_000L;

    private static readonly long[] _upperBoundsUs = [50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000];

    private readonly long[] _counts = new long[BucketCount];
    private long _count;
    private long _sumNs;
    private long _maxNs;

    /// <summary>Upper bound in microseconds of each bounded bucket; the eleventh bucket has none.</summary>
    public static ReadOnlySpan<long> UpperBoundsUs => _upperBoundsUs;

    /// <summary>Samples recorded since construction or the last <see cref="Reset"/>.</summary>
    public long Count => Volatile.Read(ref _count);

    /// <summary>Largest lateness recorded, in nanoseconds.</summary>
    public long MaxNs => Volatile.Read(ref _maxNs);

    /// <summary>Records one lateness; negative values count as zero.</summary>
    public void Record(long lateNs)
    {
        if (lateNs < 0)
        {
            lateNs = 0;
        }
        int bucket = 0;
        while (bucket < BucketCount - 1 && lateNs > _upperBoundsUs[bucket] * NsPerMicrosecond)
        {
            bucket++;
        }
        _counts[bucket]++;
        _sumNs += lateNs;
        if (lateNs > _maxNs)
        {
            _maxNs = lateNs;
        }
        Volatile.Write(ref _count, _count + 1);
    }

    /// <summary>Copies the per-bucket counts into <paramref name="destination"/>, which needs <see cref="BucketCount"/> slots.</summary>
    public void CopyCounts(Span<long> destination)
    {
        if (destination.Length < BucketCount)
        {
            throw new ArgumentException($"destination needs {BucketCount} slots, got {destination.Length}.", nameof(destination));
        }
        _counts.AsSpan().CopyTo(destination);
    }

    /// <summary>Upper bound in microseconds of the bucket that reaches <paramref name="fraction"/> (0..1] of the samples; the maximum for the open-ended bucket, zero when empty.</summary>
    public long PercentileUs(double fraction)
    {
        if (fraction <= 0 || fraction > 1 || double.IsNaN(fraction))
        {
            throw new ArgumentOutOfRangeException(nameof(fraction), fraction, "fraction must be in (0, 1]");
        }
        long total = Volatile.Read(ref _count);
        if (total == 0)
        {
            return 0;
        }
        long rank = (long)Math.Ceiling(fraction * total);
        long cumulative = 0;
        for (int bucket = 0; bucket < BucketCount - 1; bucket++)
        {
            cumulative += _counts[bucket];
            if (cumulative >= rank)
            {
                return _upperBoundsUs[bucket];
            }
        }
        return _maxNs / NsPerMicrosecond;
    }

    /// <summary>Count, mean, p50, p99 and max at this moment.</summary>
    public Summary Snapshot()
    {
        long total = Volatile.Read(ref _count);
        long meanUs = total == 0 ? 0 : _sumNs / total / NsPerMicrosecond;
        return new Summary(total, meanUs, PercentileUs(0.50), PercentileUs(0.99), _maxNs / NsPerMicrosecond);
    }

    /// <summary>Clears every counter. Call it from the recording thread, or while that thread is stopped; it is not
    /// safe against a concurrent <see cref="Record"/>.</summary>
    public void Reset()
    {
        Array.Clear(_counts);
        _sumNs = 0;
        _maxNs = 0;
        Volatile.Write(ref _count, 0);
    }

    /// <summary>One reading of the histogram, all times in microseconds.</summary>
    public readonly record struct Summary(long Count, long MeanUs, long P50Us, long P99Us, long MaxUs);
}

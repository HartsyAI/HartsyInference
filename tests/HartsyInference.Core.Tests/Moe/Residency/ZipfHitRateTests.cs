using HartsyInference.Core.Moe.Residency;
using HartsyInference.Core.Moe.Telemetry;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Moe.Residency;

/// <summary>Hit rates of each policy on seeded Zipf traces. The numbers are printed for the report.</summary>
public sealed class ZipfHitRateTests
{
    private readonly ITestOutputHelper output;

    public ZipfHitRateTests(ITestOutputHelper output)
    {
        this.output = output;
    }

    [Theory]
    [InlineData(1.0, 32)]
    [InlineData(1.2, 32)]
    [InlineData(1.0, 96)]
    public void Zipf_HitRateOrdering_LfuAboveDecayedAboveSegmentedLru(double exponent, int slots)
    {
        // Stationary Zipf traces favor perfect frequency counts. The decayed policy forgets some of that history, and
        // SLRU keys on recency only. The ordering below holds for every configuration checked when this was written.
        ExpertTraceRecord[] trace = SyntheticTraceGenerator.Zipf(42, 4, 64, 2000, 4, exponent, 1 << 20);

        ReplayResult lru = Run(trace, new SegmentedLruPolicy(protectedCapacity: slots / 2), slots, exponent);
        ReplayResult lfu = Run(trace, new LfuPolicy(), slots, exponent);
        ReplayResult decayed = Run(trace, new DecayedLfuHysteresisPolicy(0.995, 0.1), slots, exponent);

        foreach (ReplayResult result in new[] { lru, lfu, decayed })
        {
            Assert.Equal(trace.Length, result.Accesses);
            Assert.InRange(result.HitRate, 0.0, 1.0);
        }

        Assert.True(lfu.HitRate >= decayed.HitRate, $"LFU {lfu.HitRate} below decayed {decayed.HitRate}");
        Assert.True(decayed.HitRate >= lru.HitRate, $"decayed {decayed.HitRate} below SLRU {lru.HitRate}");
    }

    private ReplayResult Run(ExpertTraceRecord[] trace, IAdaptiveResidencyPolicy policy, int slots, double exponent)
    {
        ReplayResult result = TraceReplayHarness.Replay(trace, policy, slots);
        output.WriteLine(
            $"zipf s={exponent} slots={slots} {result.Policy}: hit rate {result.HitRate:F4} " +
            $"({result.Hits}/{result.Accesses}), evictions {result.Evictions}");
        return result;
    }
}

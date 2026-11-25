using HartsyInference.Engine.Placement;
using Xunit;

namespace HartsyInference.Engine.Tests.Placement;

/// <summary>The device budget an expert cache gets after the dense weights are preloaded.</summary>
public sealed class ExpertOffloadBudgetTests
{
    private const long Gib = 1L << 30;

    [Theory]
    [InlineData(0L, 24 * Gib, 0L)]
    [InlineData(24 * Gib, 0L, 0L)]
    [InlineData(-1L, 24 * Gib, 0L)]
    public void NoMemoryReported_LeavesNoBudget(long free, long total, long expected)
    {
        Assert.Equal(expected, ExpertOffloadSetup.DeviceCacheBudget(free, total));
    }

    [Fact]
    public void SmallCard_KeepsThreeGibHeadroom()
    {
        // 12 GB card, 15% of 12 GB is under 3 GiB, so the floor applies.
        Assert.Equal(5 * Gib, ExpertOffloadSetup.DeviceCacheBudget(8 * Gib, 12 * Gib));
    }

    [Fact]
    public void LargeCard_KeepsFifteenPercentHeadroom()
    {
        // 96 GB card: 15% is 14.4 GB, above the 3 GiB floor.
        long total = 96 * Gib, free = 80 * Gib;
        Assert.Equal(free - total * 15 / 100, ExpertOffloadSetup.DeviceCacheBudget(free, total));
    }

    [Fact]
    public void NothingFreeAfterHeadroom_ClampsToZero()
    {
        Assert.Equal(0L, ExpertOffloadSetup.DeviceCacheBudget(2 * Gib, 12 * Gib));
    }
}

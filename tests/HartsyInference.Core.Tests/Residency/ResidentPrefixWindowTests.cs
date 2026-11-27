using HartsyInference.Core.Memory.Vram.Streaming;
using Xunit;

namespace HartsyInference.Core.Tests.Residency;

/// <summary>The resident prefix leaves the streaming window's bytes free as well as the headroom: BlockStreamingController.Prime uploads
/// prefetchAhead + 1 streamed blocks right after the placement, outside the headroom the planner reserves.</summary>
public sealed class ResidentPrefixWindowTests
{
    private const long Gb = 1L << 30;

    [Fact]
    public void ThePrefixLeavesTheHeadroomFreeAfterTheStreamingWindowIsResident()
    {
        long[] blocks = [Gb, Gb, Gb, Gb, Gb, Gb, Gb, Gb];   // 8 GiB of blocks; 6 GiB free, 1 GiB headroom, two blocks prefetched ahead
        int prefix = ResidentPrefixSizing.SizeWithStreamingWindow(6 * Gb, Gb, blocks, prefetchAhead: 2);

        long window = (2 + 1) * Gb;
        long freeAfterPrefix = 6 * Gb - prefix * Gb;
        Assert.Equal(2, prefix);
        Assert.True(freeAfterPrefix - window >= Gb, "the window must leave the headroom free");
    }

    [Fact]
    public void TheOldSizingWouldHaveLeftTheWindowOverTheHeadroom()
    {
        long[] blocks = [Gb, Gb, Gb, Gb, Gb, Gb, Gb, Gb];
        int naive = ResidentPrefixSizing.Size(6 * Gb, Gb, blocks);

        // The naive prefix (5 blocks) leaves 1 GiB free, which the window's 3 GiB takes over the headroom.
        Assert.Equal(5, naive);
        Assert.True(6 * Gb - naive * Gb - 3 * Gb < Gb);
    }

    [Fact]
    public void WhenEveryBlockFitsThereIsNoWindowToReserve()
    {
        long[] blocks = [Gb, Gb, Gb];
        Assert.Equal(3, ResidentPrefixSizing.SizeWithStreamingWindow(10 * Gb, Gb, blocks, prefetchAhead: 2));
    }

    [Fact]
    public void NothingFitsWhenTheHeadroomAloneExceedsFree()
    {
        long[] blocks = [Gb, Gb];
        Assert.Equal(0, ResidentPrefixSizing.SizeWithStreamingWindow(Gb, 2 * Gb, blocks, prefetchAhead: 1));
    }

    [Fact]
    public void WidestBlockSetsTheWindow()
    {
        // A narrow first block among wide ones: the window is sized at the widest (2 GiB), so the prefix leaves 2 * (2 GiB) + the headroom free.
        long[] blocks = [Gb / 2, 2 * Gb, Gb / 2, Gb / 2, Gb / 2, Gb / 2, Gb / 2, Gb / 2, Gb / 2, Gb / 2, Gb / 2, Gb / 2];
        int prefix = ResidentPrefixSizing.SizeWithStreamingWindow(6 * Gb, Gb / 2, blocks, prefetchAhead: 1);
        long used = 0;
        for (int i = 0; i < prefix; i++) used += blocks[i];
        Assert.Equal(1, prefix);
        Assert.True(6 * Gb - used - 4 * Gb >= Gb / 2);
    }
}

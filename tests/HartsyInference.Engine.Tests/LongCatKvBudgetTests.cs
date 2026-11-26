using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The avatar's KV cache counts against the streaming budget, and a later run whose budget is negative is refused before its denoise.
/// The numbers are the 4090 run at 832x480: a later run's free memory after the shared weights was 14639 MB, and its cache is 12.3 GB.</summary>
public sealed class LongCatKvBudgetTests
{
    private const long Mb = 1L << 20;

    [Fact]
    public void TheCacheAt832x480IsTheF32KeysAndValuesOfFiveConditioningLatentsInEveryBlock()
    {
        LongCatConfig config = new() { HiddenSize = 4096, Depth = 48, PatchSize = (1, 2, 2) };
        // 832x480 is a 104x60 latent grid; 5 latent frames of 30x52 tokens each.
        long bytes = LongCatRecipeSupport.KvCacheBytes(config, condLatents: 5, latentHeight: 60, latentWidth: 104, offloaded: false);
        long expected = 2L * 48 * (5 * 30 * 52) * 4096 * 4;
        Assert.Equal(expected, bytes);
        Assert.Equal(12_268_339_200L, bytes);
    }

    [Fact]
    public void At640x368TheCacheIsSmallerByTheTokenRatio()
    {
        LongCatConfig config = new() { HiddenSize = 4096, Depth = 48, PatchSize = (1, 2, 2) };
        // 640x368 is a 80x46 latent grid: 23x40 tokens per frame, 4600 tokens for five frames.
        Assert.Equal(2L * 48 * (5 * 23 * 40) * 4096 * 4, LongCatRecipeSupport.KvCacheBytes(config, 5, 46, 80, offloaded: false));
    }

    [Fact]
    public void OffloadedTheDeviceHoldsOneBlockOfTheCache()
    {
        LongCatConfig config = new() { HiddenSize = 4096, Depth = 48, PatchSize = (1, 2, 2) };
        long device = LongCatRecipeSupport.KvCacheBytes(config, 5, 60, 104, offloaded: false);
        long staged = LongCatRecipeSupport.KvCacheBytes(config, 5, 60, 104, offloaded: true);
        Assert.Equal(device / 48, staged);
        Assert.Equal(255_590_400L, staged);
    }

    [Fact]
    public void AMeasuredPeakReplacesThePerTokenEstimate()
    {
        long estimate = LongCatRecipeSupport.BudgetHeadroomBytes(39_000, 4096, 11008, measuredPeakBytes: null, cacheBytes: 0);
        long measured = LongCatRecipeSupport.BudgetHeadroomBytes(39_000, 4096, 11008, measuredPeakBytes: 3_000 * Mb, cacheBytes: 0);
        Assert.Equal(LongCatRecipeSupport.FixedPoolBytes + LongCatRecipeSupport.SafetyBytes + 3_000 * Mb, measured);
        Assert.True(measured < estimate);
        Assert.Equal(measured + 256 * Mb, LongCatRecipeSupport.BudgetHeadroomBytes(39_000, 4096, 11008, measuredPeakBytes: 3_000 * Mb, cacheBytes: 256 * Mb));
    }

    [Fact]
    public void ARunWhoseCacheFitsIsAccepted()
    {
        LongCatRecipeSupport.RequireCacheRoom(freeAfterShared: 14_639 * Mb, headroomBeforeCache: 9_000 * Mb, cacheBytes: 4_000 * Mb);
    }

    [Fact]
    public void ARunWhoseBudgetIsNegativeIsRefusedWithTheNumbers()
    {
        // The 832x480 later run: 14639 MB free, about 15 GB of activation headroom, and the 12.3 GB cache.
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            LongCatRecipeSupport.RequireCacheRoom(freeAfterShared: 14_639 * Mb, headroomBeforeCache: 15_000 * Mb, cacheBytes: 12_268 * Mb));
        Assert.Contains("12268 MB of KV cache", error.Message);
        Assert.Contains("14639 MB is free", error.Message);
        Assert.Contains("MB short", error.Message);
    }

    [Fact]
    public void ExactlyZeroIsStillAccepted()
    {
        LongCatRecipeSupport.RequireCacheRoom(freeAfterShared: 10 * Mb, headroomBeforeCache: 6 * Mb, cacheBytes: 4 * Mb);
    }
}

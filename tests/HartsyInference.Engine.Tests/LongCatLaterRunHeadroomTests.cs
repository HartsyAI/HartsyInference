using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The later run's headroom with the cached step's bytes: the activations of the noisy latents, plus the staged pair and the joined copy of
/// the conditioning keys and values. The free figures are the logs: 832x480 had 14639 MiB after the shared weights, 640x368 had 14761 MiB.</summary>
public sealed class LongCatLaterRunHeadroomTests
{
    private const long Mb = 1L << 20;

    private static LongCatConfig Config() => new() { HiddenSize = 4096, Depth = 48, PatchSize = (1, 2, 2) };

    [Fact]
    public void A832x480LaterRunCountsTheJoinedCopyAndIsShortByAboutThirtyEightMiB()
    {
        LongCatConfig config = Config();
        // 20 noisy latents of 30x52 tokens; five conditioning latents of 60x104 latent grid.
        long headroom = LongCatRecipeSupport.BudgetHeadroomBytes(20 * 30 * 52, 4096, 11008, measuredPeakBytes: null,
            cacheBytes: LongCatRecipeSupport.CachedStepBytes(config, 5, 60, 104));
        long spendable = 14_639 * Mb - headroom;

        Assert.Equal(2 * 255_590_400L, LongCatRecipeSupport.CachedStepBytes(config, 5, 60, 104));
        Assert.InRange(spendable / (double)Mb, -40.0, -35.0);
    }

    [Fact]
    public void A640x368LaterRunStillFitsWithItsJoinedCopy()
    {
        LongCatConfig config = Config();
        // 20 noisy latents of 23x40 tokens; five conditioning latents of 46x80 latent grid.
        long headroom = LongCatRecipeSupport.BudgetHeadroomBytes(20 * 23 * 40, 4096, 11008, measuredPeakBytes: null,
            cacheBytes: LongCatRecipeSupport.CachedStepBytes(config, 5, 46, 80));
        long spendable = 14_761 * Mb - headroom;

        Assert.InRange(spendable / (double)Mb, 3_070.0, 3_100.0);
    }

    [Fact]
    public void TheCachedStepCountsTwoPairsOfTheConditioningTokens()
    {
        LongCatConfig config = Config();
        long pair = LongCatRecipeSupport.KvCacheBytes(config, 5, 60, 104, offloaded: true);
        Assert.Equal(2 * pair, LongCatRecipeSupport.CachedStepBytes(config, 5, 60, 104));
    }
}

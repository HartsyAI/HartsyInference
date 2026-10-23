using HartsyInference.Diffusion.Models.Vae.QwenImage;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

public sealed class QwenImageVaeBudgetTests
{
    private const long GiB = 1L << 30;
    private const long MiB = 1L << 20;

    /// <summary>A 1024² decode asks for the banded im2col cap (1 GiB) plus its activations; less free VRAM must not pick the full decode.</summary>
    [Fact]
    public void EstimateDecodeBytes_At1024_IsAtLeastTheBandCap()
    {
        long need = QwenImageVaeDecoder.EstimateDecodeBytes(128, 128, GiB);
        Assert.True(need >= GiB, $"estimate {need / MiB} MB is below the 1 GiB band cap");
    }

    [Fact]
    public void ChooseTileLatent_PicksLargestTileThatFits()
    {
        Assert.Equal(32, QwenImageVaeDecoder.ChooseTileLatent(600 * MiB, GiB));
        Assert.Equal(24, QwenImageVaeDecoder.ChooseTileLatent(352 * MiB, GiB));
        Assert.Equal(16, QwenImageVaeDecoder.ChooseTileLatent(150 * MiB, GiB));
    }

    [Fact]
    public void ChooseTileLatent_WithAmpleMemory_PicksLargestCandidate()
    {
        Assert.Equal(64, QwenImageVaeDecoder.ChooseTileLatent(64 * GiB, GiB));
    }

    [Fact]
    public void ChooseTileLatent_BelowEveryCandidate_FallsBackToSmallest()
    {
        Assert.Equal(16, QwenImageVaeDecoder.ChooseTileLatent(MiB, GiB));
    }

    [Fact]
    public void PlanTileLatent_FitsWhole_DecodesWhole()
    {
        Assert.Equal(0, QwenImageVaeDecoder.PlanTileLatent(128, 128, 64 * GiB, GiB));
    }

    [Fact]
    public void PlanTileLatent_LatentSmallerThanSmallestTile_DecodesWhole()
    {
        Assert.Equal(0, QwenImageVaeDecoder.PlanTileLatent(8, 12, MiB, GiB));
        Assert.Equal(0, QwenImageVaeDecoder.PlanTileLatent(16, 16, MiB, GiB));
    }

    [Fact]
    public void PlanTileLatent_LatentLargerThanSmallestTile_TilesWhenShort()
    {
        Assert.Equal(16, QwenImageVaeDecoder.PlanTileLatent(128, 128, 150 * MiB, GiB));
        Assert.Equal(24, QwenImageVaeDecoder.PlanTileLatent(128, 128, 352 * MiB, GiB));
    }
}

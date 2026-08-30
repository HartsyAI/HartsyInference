using HartsyInference.Diffusion.Models.Vae;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Guards the VAE tile planner against non-advancing grids and sub-latent tiles.</summary>
public sealed class MiniMaxH3VaeTilePlanTests
{
    [Fact]
    public void OverlapCannotPreventOneLatentUnitOfAdvance()
    {
        MiniMaxH3VideoVaeConfig config = new MiniMaxH3VideoVaeConfig
        {
            VaeRatio = 16,
            TileSize = 64,
            TileOverlapMin = 64,
        };

        (int[] starts, int[] lengths, int[] overlaps) = config.SplitTiles(128);

        Assert.Equal(starts.Length - 1, overlaps.Length);
        for (int i = 1; i < starts.Length; i++)
        {
            Assert.True(starts[i] - starts[i - 1] >= config.VaeRatio);
        }
        Assert.Equal(0, starts[0]);
        Assert.Equal(128, starts[^1] + lengths[^1]);
    }

    [Theory]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(24)]
    public void SubTileOrMisalignedSizesFailInsteadOfHanging(int tileSize)
    {
        MiniMaxH3VideoVaeConfig config = new MiniMaxH3VideoVaeConfig
        {
            VaeRatio = 16,
            TileSize = tileSize,
            TileOverlapMin = tileSize,
        };
        Assert.Throws<ArgumentOutOfRangeException>(() => config.SplitTiles(64));
    }
}

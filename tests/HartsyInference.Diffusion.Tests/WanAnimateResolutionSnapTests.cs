using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins Wan-Animate's pixel-size grid. The DiT patchifies <c>(1,2,2)</c> over an 8×-compressed latent, so a
/// latent grid that is merely a multiple of 8 pixels can be odd — and then <c>Unpatchify</c> emits one fewer column
/// than the scheduler's step was handed. Snapping to 8 (the VAE stride alone) is the live form of that bug.</summary>
public sealed class WanAnimateResolutionSnapTests
{
    [Fact]
    public void PatchAndVaeStridesCompoundIntoSixteen()
    {
        Assert.Equal(16, VideoRecipeUtils.PatchAlignedMultiple(vaeSpatialCompression: 8, patchSize: (1, 2, 2)));
        Assert.Equal(32, VideoRecipeUtils.PatchAlignedMultiple(vaeSpatialCompression: 16, patchSize: (1, 2, 2)));
    }

    [Theory]
    [InlineData(832, 480, 832, 480)]    // the official 480p bucket is already aligned
    [InlineData(1280, 720, 1280, 720)]  // and the 720p one
    [InlineData(844, 492, 848, 496)]
    [InlineData(840, 488, 832, 480)]    // exact midpoints round to even, per Math.Round's default
    [InlineData(1, 1, 16, 16)]          // never below one multiple
    public void ResolutionSnapsToTheSixteenGrid(int width, int height, int expectedWidth, int expectedHeight)
    {
        VideoRequest request = new VideoRequest { Prompt = "", Width = width, Height = height };
        (int w, int h) = VideoRecipeUtils.ResolveResolution(request, 16);
        Assert.Equal(expectedWidth, w);
        Assert.Equal(expectedHeight, h);
    }

    [Fact]
    public void SnappingToEightLeavesAnOddLatentGridThatSixteenNeverDoes()
    {
        bool eightEverOdd = false;
        for (int size = 1; size <= 2048; size++)
        {
            VideoRequest request = new VideoRequest { Prompt = "", Width = size, Height = size };
            (int wide, int _) = VideoRecipeUtils.ResolveResolution(request, 8);
            eightEverOdd |= (wide / 8) % 2 != 0;
            (int aligned, int _) = VideoRecipeUtils.ResolveResolution(request, 16);
            Assert.Equal(0, (aligned / 8) % 2);
        }
        Assert.True(eightEverOdd, "the 8-multiple snap must be able to produce the odd latent grid this guards against");
    }
}

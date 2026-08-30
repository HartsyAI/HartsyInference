using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

public sealed class VideoRecipeUtilsGuideFitTests
{
    [Fact]
    public void CoverFitCenterCropsTheLongAxis()
    {
        ImageData image = new ImageData
        {
            Width = 4,
            Height = 2,
            Rgb =
            [
                10, 1, 2, 20, 3, 4, 30, 5, 6, 40, 7, 8,
                50, 9, 10, 60, 11, 12, 70, 13, 14, 80, 15, 16,
            ],
        };

        byte[] fitted = VideoRecipeUtils.FitGuideFrame(image, 2, 2, VideoGuideFitMode.Cover);

        Assert.Equal<byte>([20, 3, 4, 30, 5, 6, 60, 11, 12, 70, 13, 14], fitted);
    }
}

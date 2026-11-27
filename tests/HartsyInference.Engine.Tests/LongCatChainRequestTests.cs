using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The avatar chain's request checks: each refusal carries its video.chain.* code, and a driving clip shorter than the clip is detected.</summary>
public sealed class LongCatChainRequestTests
{
    [Fact]
    public void AnOverBudgetChainCarriesTheBudgetCode()
    {
        VideoChainException error = Assert.Throws<VideoChainException>(() => LongCatAvatarRecipe.ValidateChainRequest(10_000, 93, 832, 480, 25));
        Assert.Equal("video.chain.budget", error.Code);
    }

    [Fact]
    public void ASegmentTooLongCarriesTheSegmentCode()
    {
        VideoChainException error = Assert.Throws<VideoChainException>(() => LongCatAvatarRecipe.ValidateChainRequest(null, 200, 832, 480, 25));
        Assert.Equal("video.chain.segment_frames", error.Code);
    }

    [Fact]
    public void TheWrongFrameRateCarriesTheFpsCode()
    {
        VideoChainException error = Assert.Throws<VideoChainException>(() => LongCatAvatarRecipe.ValidateChainRequest(173, 93, 832, 480, 24));
        Assert.Equal("video.chain.fps", error.Code);
    }

    [Fact]
    public void AValidChainPasses()
    {
        LongCatAvatarRecipe.ValidateChainRequest(173, 93, 832, 480, LongCatAvatarRecipe.OutputFps);
    }

    [Theory]
    [InlineData(2.0, 100, true)]
    [InlineData(4.0, 100, false)]
    [InlineData(5.0, 100, false)]
    public void ShortDrivingAudioIsDetected(double seconds, int frames, bool expected)
    {
        Assert.Equal(expected, LongCatAvatarRecipe.DrivingAudioIsShort(seconds, frames, LongCatAvatarRecipe.OutputFps));
    }
}

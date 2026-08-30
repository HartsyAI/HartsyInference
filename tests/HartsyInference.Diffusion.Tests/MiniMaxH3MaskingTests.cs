using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Unit gates for arbitrary-guide frame geometry and continuous H3 AV-mask resampling.</summary>
public sealed class MiniMaxH3MaskingTests
{
    [Theory]
    [InlineData(-1, 39, 38)]
    [InlineData(-39, 39, 0)]
    [InlineData(17, 39, 17)]
    public void SignedGuideFramesResolveFromTheAlignedTargetEnd(int input, int frames, int expected)
    {
        Assert.Equal(expected, MiniMaxH3Masking.ResolveFrameIndex(input, frames));
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(4, 1)]
    [InlineData(5, 5)]
    [InlineData(21, 5)]
    [InlineData(22, 22)]
    [InlineData(38, 22)]
    [InlineData(39, 39)]
    public void GuideVideosUseAStillBelowFiveAndOtherwiseTruncateTo17NPlus5(int input, int expected)
    {
        Assert.Equal(expected, MiniMaxH3Masking.GuideFrameCount(input));
    }

    [Theory]
    [InlineData(0, 39, 39)]
    [InlineData(17, 22, 39)]
    [InlineData(38, 1, 39)]
    public void NormalizedGuideClipsMayEndExactlyAtTheAlignedTarget(
        int anchor, int guideFrames, int targetFrames)
    {
        MiniMaxH3Masking.ValidateGuideFrameSpan(anchor, guideFrames, targetFrames);
    }

    [Theory]
    [InlineData(1, 39, 39)]
    [InlineData(18, 22, 39)]
    [InlineData(38, 5, 39)]
    public void NormalizedGuideClipsCannotExtendPastTheAlignedTarget(
        int anchor, int guideFrames, int targetFrames)
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(
            () => MiniMaxH3Masking.ValidateGuideFrameSpan(anchor, guideFrames, targetFrames));

        Assert.Equal("guideFrameCount", error.ParamName);
    }

    [Fact]
    public void VideoMaskUsesQuantizedTwoByTwoAmaxInPackedRowOrder()
    {
        float[] mask =
        [
            0f, 0.1f, 0.2f, 0.3f,
            0.4f, 0.5f, 0.6f, 0.7f,
            0.8f, 0.9f, 0.25f, 0.35f,
            0.45f, 0.55f, 0.65f, 0.75f,
        ];

        float[] rows = Assert.IsType<float[]>(MiniMaxH3Masking.PackVideoMaskRows(
            mask, latentFrames: 1, latentHeight: 4, latentWidth: 4, out float[]? featureValues));

        Assert.Equal([0.5f, 0.703125f, 0.90234375f, 0.75f], rows);
        Assert.Equal(
            [
                0f, 0.1f, 0.4f, 0.5f,
                0.2f, 0.3f, 0.6f, 0.7f,
                0.8f, 0.9f, 0.45f, 0.55f,
                0.25f, 0.35f, 0.65f, 0.75f,
            ],
            Assert.IsType<float[]>(featureValues));
    }

    [Fact]
    public void VideoMaskRetainsRawPatchValuesWhenTheTokenAmaxIsWhite()
    {
        float[] rows = Assert.IsType<float[]>(MiniMaxH3Masking.PackVideoMaskRows(
            [0f, 0.25f, 0.75f, 1f], latentFrames: 1, latentHeight: 2, latentWidth: 2,
            out float[]? featureValues));

        Assert.Equal([1f], rows);
        Assert.Equal([0f, 0.25f, 0.75f, 1f], Assert.IsType<float[]>(featureValues));
    }

    [Fact]
    public void TokenMasksRoundUpToTheOneOver256Grid()
    {
        float[] raw = [0f, 1f / 512f, 0.25f, 1f];
        float[] rows = Assert.IsType<float[]>(MiniMaxH3Masking.PackVideoMaskRows(
            raw, latentFrames: 1, latentHeight: 1, latentWidth: 4, out float[]? featureValues,
            patchHeight: 1, patchWidth: 1));

        Assert.Equal([0f, 1f / 256f, 0.25f, 1f], rows);
        Assert.Equal(raw, Assert.IsType<float[]>(featureValues));
    }

    [Fact]
    public void AudioMaskResamplesAtFortyHertzAndRepeatsChannelMajor()
    {
        float[] rows = Assert.IsType<float[]>(MiniMaxH3Masking.ResampleAudioMask(
            [0f, 1f, 0f], sourceRate: 20f, targetAudioLatentFrames: 5,
            out float[]? featureRows));

        Assert.Equal([0f, 0.5f, 1f, 0.5f, 0f, 0f, 0.5f, 1f, 0.5f, 0f], rows);
        Assert.Equal(rows, Assert.IsType<float[]>(featureRows));
    }

    [Fact]
    public void AudioMaskRetainsContinuousRowsSeparatelyFromQuantizedTokens()
    {
        float[] rows = Assert.IsType<float[]>(MiniMaxH3Masking.ResampleAudioMask(
            [0.1f], sourceRate: 40f, targetAudioLatentFrames: 1,
            out float[]? featureRows));

        Assert.Equal([0.1015625f, 0.1015625f], rows);
        Assert.Equal([0.1f, 0.1f], Assert.IsType<float[]>(featureRows));
    }

    [Fact]
    public void AllWhiteMasksAreExactNoOps()
    {
        Assert.Null(MiniMaxH3Masking.PackVideoMaskRows(
            Enumerable.Repeat(1f, 2 * 4 * 4).ToArray(), 2, 4, 4,
            out float[]? videoFeatureValues));
        Assert.Null(videoFeatureValues);
        Assert.Null(MiniMaxH3Masking.ResampleAudioMask(
            [1f, 1f], 40f, 8, out float[]? audioFeatureRows));
        Assert.Null(audioFeatureRows);
    }

    [Fact]
    public void GuideAudioCropsToTheDurationRemainingAfterItsAnchor()
    {
        Assert.Equal(66, MiniMaxH3Masking.GuideAudioLatentFrames(80, 8));
        Assert.Equal(1, MiniMaxH3Masking.GuideAudioLatentFrames(65, 38));
    }
}

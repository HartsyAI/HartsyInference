using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The continuation's source geometry against the pinned reference (meituan-longcat/LongCat-Video @ 58eb336): the demo's stride to 15 fps
/// and the 480p aspect bucket (see <see cref="LongCatContinuation"/> for the cited lines).</summary>
public sealed class LongCatContinuationTests
{
    [Theory]
    [InlineData(30.0, 2)]   // 30 / 15 = 2
    [InlineData(15.0, 1)]
    [InlineData(24.0, 2)]   // round(1.6) = 2
    [InlineData(22.5, 2)]   // round(1.5) = 2: Python's round is half-to-even
    [InlineData(37.5, 2)]   // round(2.5) = 2
    [InlineData(0.0, 1)]    // an unknown rate (cv2 reports 0) strides by 1
    public void TheStrideIsTheDemosRoundedRatioToFifteen(double fps, int stride) =>
        Assert.Equal(stride, LongCatContinuation.Stride(fps));

    [Fact]
    public void TheBucketIsTheClosestAspectKeyAndTheFirstOnATie()
    {
        Assert.Equal((480, 832), LongCatContinuation.Bucket(1280, 720));   // 0.5625 → "0.58"
        Assert.Equal((480, 832), LongCatContinuation.Bucket(832, 480));    // 0.5769 → "0.58"
        // 1.0 is equidistant from "0.95" and "1.05" in doubles; the first listed key wins, as Python's stable sorted(...)[0] does.
        Assert.Equal((608, 640), LongCatContinuation.Bucket(16, 16));
        Assert.Equal((320, 1216), LongCatContinuation.Bucket(4000, 1000)); // 0.25 → "0.26", the closest key
    }

    [Fact]
    public void ThePreparedSourceStridesAndResizesOnlyItsContext()
    {
        byte[][] decoded = Enumerable.Range(0, 40).Select(i => new byte[16 * 8 * 3]).ToArray();   // 16×8 source, ratio 0.5 → "0.52"

        LongCatContinuation.Source source = LongCatContinuation.Prepare(decoded, width: 16, height: 8, sourceFps: 30, contextFrames: 13);

        Assert.Equal(20, source.Frames.Length);                       // 40 frames at stride 2
        Assert.Equal(13, source.ContextFrames.Length);
        Assert.Equal((448, 864), (source.GenerationHeight, source.GenerationWidth));
        Assert.All(source.ContextFrames, frame => Assert.Equal(448 * 864 * 3, frame.Length));
        Assert.Equal(16, source.Width);
        Assert.Equal(8, source.Height);
    }

    [Fact]
    public void AClipShorterThanTheContextIsRefused()
    {
        byte[][] decoded = Enumerable.Range(0, 10).Select(i => new byte[16 * 16 * 3]).ToArray();

        Assert.Throws<ArgumentException>(() => LongCatContinuation.Prepare(decoded, 16, 16, sourceFps: 15, contextFrames: 13));
    }

    [Fact]
    public void ContinuationIsGatedThroughTheFeatureMappingSoOnlyTheBaseBuildDeclaresIt()
    {
        VideoRequest request = new() { Prompt = "p", VideoContinuation = new VideoClip { Data = [1] } };

        Assert.True((VideoService.RequestedFeatures(request) & VideoFeatures.VideoContinuation) != 0);
        Assert.True(new LongCatVideoRecipe().Supports.HasFlag(VideoFeatures.VideoContinuation));
        Assert.False(new LongCatVideoDistilledRecipe().Supports.HasFlag(VideoFeatures.VideoContinuation));
        Assert.False(new LongCatAvatarRecipe().Supports.HasFlag(VideoFeatures.VideoContinuation));
    }
}

using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The 1.6 chain's context is the segment's last latent frames, carried as they are (no decode-encode round trip).</summary>
public sealed class LongCatCarriedContextTests
{
    [Fact]
    public void TheContextIsTheLastLatentFramesOfEachChannel()
    {
        const int channels = 2, segmentFrames = 5, hw = 3;
        float[] segment = new float[channels * segmentFrames * hw];
        for (int i = 0; i < segment.Length; i++)
            segment[i] = i;

        float[] context = LongCatRecipeSupport.CarriedContext(segment, channels, segmentFrames, contextLatents: 2, hw);

        Assert.Equal(channels * 2 * hw, context.Length);
        // Channel 0: frames 3 and 4 (values 9..14); channel 1: frames 3 and 4 of its block (values 24..29).
        float[] expected = [9, 10, 11, 12, 13, 14, 24, 25, 26, 27, 28, 29];
        Assert.Equal(expected, context);
    }

    [Fact]
    public void AContextAsLongAsTheSegmentIsTheWholeSegment()
    {
        float[] segment = [1, 2, 3, 4];
        Assert.Equal(segment, LongCatRecipeSupport.CarriedContext(segment, channels: 1, segmentFrames: 4, contextLatents: 4, hw: 1));
    }

    [Fact]
    public void AContextLongerThanTheSegmentIsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            LongCatRecipeSupport.CarriedContext([1, 2, 3, 4], channels: 1, segmentFrames: 4, contextLatents: 5, hw: 1));
    }
}

using HartsyInference.Core.Numerics;
using Xunit;

namespace HartsyInference.Core.Tests.Runtime;

/// <summary><c>F.interpolate(mode="linear", align_corners=True)</c> on a <c>[frames, channels]</c> sequence. The
/// expectations are closed-form (linear ramps, exact end frames), not values taken from torch.</summary>
public sealed class LinearResampleTests
{
    [Fact]
    public void ThreeFramesToFive_InsertsTheMidpoints()
    {
        float[] source = [0f, 10f, 20f];
        float[] output = new float[5];

        LinearResample.AlignCorners(source, 3, 1, output, 5);

        Assert.Equal([0f, 5f, 10f, 15f, 20f], output);
    }

    [Theory]
    [InlineData(186, 93)]
    [InlineData(236, 118)]
    [InlineData(93, 186)]
    [InlineData(7, 7)]
    public void ARamp_ComesBackAsARampOverTheSameSpan(int sourceFrames, int destinationFrames)
    {
        float[] source = new float[sourceFrames];
        for (int i = 0; i < sourceFrames; i++) source[i] = i;
        float[] output = new float[destinationFrames];

        LinearResample.AlignCorners(source, sourceFrames, 1, output, destinationFrames);

        // The position is an F32 product, as in torch, so the ramp is exact only to F32 rounding.
        Assert.Equal(0f, output[0]);
        for (int i = 0; i < destinationFrames; i++)
        {
            double expected = (double)i * (sourceFrames - 1) / (destinationFrames - 1);
            Assert.InRange(output[i], expected - 2e-6 * (1 + expected), expected + 2e-6 * (1 + expected));
        }
    }

    [Fact]
    public void ChannelsAreResampledIndependently()
    {
        float[] source = [0f, 100f, 10f, 200f, 20f, 300f];
        float[] output = new float[3 * 2];

        LinearResample.AlignCorners(source, 3, 2, output, 3);

        Assert.Equal(source, output);
        float[] doubled = new float[5 * 2];
        LinearResample.AlignCorners(source, 3, 2, doubled, 5);
        Assert.Equal([0f, 100f, 5f, 150f, 10f, 200f, 15f, 250f, 20f, 300f], doubled);
    }

    [Fact]
    public void AOneFrameOutput_TakesTheFirstFrame()
    {
        float[] output = new float[2];

        LinearResample.AlignCorners([3f, 4f, 9f, 9f], 2, 2, output, 1);

        Assert.Equal([3f, 4f], output);
    }

    [Fact]
    public void ARejectedShape_Throws()
    {
        Assert.Throws<ArgumentException>(() => LinearResample.AlignCorners(new float[4], 3, 1, new float[2], 2));
        Assert.Throws<ArgumentException>(() => LinearResample.AlignCorners(new float[3], 3, 1, new float[3], 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => LinearResample.AlignCorners(new float[1], 0, 1, new float[1], 1));
    }

    [Theory]
    [InlineData(480_000, 16_000, 25.0, 750)]
    [InlineData(16_999, 16_000, 25.0, 26)]
    [InlineData(15_999, 16_000, 25.0, 24)]
    public void FramesAtRate_TruncatesTowardZero(int samples, int rate, double fps, int expected)
        => Assert.Equal(expected, LinearResample.FramesAtRate(samples, rate, fps));
}

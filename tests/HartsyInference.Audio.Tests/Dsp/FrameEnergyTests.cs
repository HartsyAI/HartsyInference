using HartsyInference.Audio.Dsp;
using Xunit;

namespace HartsyInference.Audio.Tests.Dsp;

public sealed class FrameEnergyTests
{
    [Fact]
    public void FrameRms_ShortLastFrame_IsMeasuredOverItsOwnSamples()
    {
        float[] y = [1f, 1f, 1f, 1f, 0.5f];

        double[] rms = FrameEnergy.FrameRms(y, 4, 2);

        Assert.Equal(1.0, rms[0], 12);
        Assert.Equal(0.5, rms[1], 12);
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(4, -1)]
    [InlineData(4, 4)]
    public void FrameRms_BadFrameOrCount_Throws(int frame, int count)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => FrameEnergy.FrameRms(new float[10], frame, count));
    }

    [Fact]
    public void MaxValue_EmptyArray_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => FrameEnergy.MaxValue([]));
    }
}

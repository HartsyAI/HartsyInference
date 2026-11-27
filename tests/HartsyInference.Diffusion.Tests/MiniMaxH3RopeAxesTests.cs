using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The shared H3 rope builder with two axes (the single-frame VAE decoder) and with the default three.</summary>
public sealed class MiniMaxH3RopeAxesTests
{
    [Fact]
    public void TwoAxisTablesRotateHeightAndWidthOnly()
    {
        float[] invFreq = [1f];
        double[] positions = [0.5, 0.25];

        (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables(positions, invFreq, headDim: 4, axes: 2);
        using (cos)
        using (sin)
        {
            // One token, one frequency: height in slot 0, width in slot 1, each duplicated into the pair half at +2.
            Span<float> c = cos.AsSpan<float>();
            Assert.Equal(Math.Cos(0.5), c[0], 6);
            Assert.Equal(Math.Cos(0.25), c[1], 6);
            Assert.Equal(Math.Cos(0.5), c[2], 6);
            Assert.Equal(Math.Cos(0.25), c[3], 6);

            Span<float> s = sin.AsSpan<float>();
            Assert.Equal(Math.Sin(0.5), s[0], 6);
            Assert.Equal(Math.Sin(0.25), s[1], 6);
        }
    }

    [Fact]
    public void RotaryWidthScalesWithTheAxisCount()
    {
        Assert.Equal(32, MiniMaxH3Rope.RotaryDim(invFreqLen: 8, axes: 2));
        Assert.Equal(48, MiniMaxH3Rope.RotaryDim(invFreqLen: 8));
    }

    [Fact]
    public void RefusesPositionsThatDoNotMatchTheAxisCount()
    {
        Assert.Throws<ArgumentException>(() => MiniMaxH3Rope.BuildTables([0.5, 0.25, 0.1], [1f], headDim: 4, axes: 2));
    }
}

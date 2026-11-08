using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov3;
using Xunit;

namespace HartsyInference.Vision.Dinov3.Tests;

public sealed unsafe class Dinov3RopeTests
{
    [Fact]
    public void Periods_AreBaseOneHundredRoundedThroughBf16()
    {
        float[] periods = PeriodsOf64();

        Assert.Equal(16, periods.Length);
        Assert.Equal(1f, periods[0]);
        // 100^(1/16) = 1.33352...; bf16 keeps 7 mantissa bits, so 1 + 43/128 = 1.3359375.
        Assert.Equal(1.3359375f, periods[1]);
        for (int k = 0; k < periods.Length; k++)
        {
            double exact = Math.Pow(100.0, 2.0 * k / 32.0);
            Assert.True(Math.Abs(periods[k] - exact) <= exact / 128.0, $"period {k}: {periods[k]} vs {exact}");
        }
    }

    [Theory]
    [InlineData(1.0f, 1.0f)]
    // Halfway between 1 and 1 + 2^-7: ties go to the even mantissa, 1.
    [InlineData(1.00390625f, 1.0f)]
    // Halfway between 1 + 2^-7 (odd) and 1 + 2^-6 (even): ties go to 1 + 2^-6.
    [InlineData(1.01171875f, 1.015625f)]
    [InlineData(-2.5f, -2.5f)]
    public void RoundToBf16_IsRoundToNearestEven(float input, float expected)
    {
        Assert.Equal(expected, Dinov3RopeTables.RoundToBf16(input));
    }

    [Fact]
    public void PrefixTokens_AreIdentityRotations()
    {
        (Tensor cos, Tensor sin) = Dinov3RopeTables.Build(1, prefix: 5, gridH: 2, gridW: 3, headDim: 64);
        try
        {
            float* c = (float*)cos.DataPointer, s = (float*)sin.DataPointer;
            for (int t = 0; t < 5; t++)
                for (int ch = 0; ch < 64; ch++)
                {
                    Assert.Equal(1f, c[t * 64 + ch]);
                    Assert.Equal(0f, s[t * 64 + ch]);
                }
        }
        finally
        {
            cos.Dispose();
            sin.Dispose();
        }
    }

    [Fact]
    public void Tables_HaveShapeBatchTokensAndHeadDim()
    {
        (Tensor cos, Tensor sin) = Dinov3RopeTables.Build(2, prefix: 5, gridH: 28, gridW: 28, headDim: 64);
        try
        {
            Assert.Equal(new TensorShape(2, 789, 64), cos.Shape);
            Assert.Equal(new TensorShape(2, 789, 64), sin.Shape);
        }
        finally
        {
            cos.Dispose();
            sin.Dispose();
        }
    }

    [Fact]
    public void RotateHalfPartnersShareAnAngle()
    {
        (Tensor cos, Tensor sin) = Dinov3RopeTables.Build(1, prefix: 5, gridH: 4, gridW: 4, headDim: 64);
        try
        {
            float* c = (float*)cos.DataPointer, s = (float*)sin.DataPointer;
            for (int t = 5; t < 5 + 16; t++)
                for (int i = 0; i < 32; i++)
                {
                    Assert.Equal(c[t * 64 + i], c[t * 64 + i + 32]);
                    Assert.Equal(s[t * 64 + i], s[t * 64 + i + 32]);
                }
        }
        finally
        {
            cos.Dispose();
            sin.Dispose();
        }
    }

    [Fact]
    public void PatchAngles_MatchTheReferenceFormula()
    {
        // The reference: coords_h[i] = (i + 0.5) / H mapped to [-1, 1]; angle = 2*pi*coord / period[k], with the first
        // quarter of the half head reading rows, the second quarter columns, and both tiled over the head.
        const int gridH = 28, gridW = 28;
        double[] periods = Enumerable.Range(0, 16).Select(k => (double)Dinov3RopeTables.RoundToBf16((float)Math.Pow(100.0, 2.0 * k / 32.0))).ToArray();
        (Tensor cos, Tensor sin) = Dinov3RopeTables.Build(1, prefix: 5, gridH, gridW, headDim: 64);
        try
        {
            float* c = (float*)cos.DataPointer, s = (float*)sin.DataPointer;
            double maxErr = 0;
            foreach ((int h, int w) in new[] { (0, 0), (0, 27), (13, 5), (27, 27), (27, 0) })
            {
                int token = 5 + h * gridW + w;
                for (int ch = 0; ch < 64; ch++)
                {
                    int half = ch % 32;
                    int axis = half / 16, k = half % 16;
                    double coord = 2.0 * ((axis == 0 ? h : w) + 0.5) / 28.0 - 1.0;
                    double angle = 2.0 * Math.PI * coord / periods[k];
                    maxErr = Math.Max(maxErr, Math.Abs(c[token * 64 + ch] - Math.Cos(angle)));
                    maxErr = Math.Max(maxErr, Math.Abs(s[token * 64 + ch] - Math.Sin(angle)));
                }
            }
            Assert.True(maxErr < 2e-5, $"max table error {maxErr:G6}");
        }
        finally
        {
            cos.Dispose();
            sin.Dispose();
        }
    }

    [Fact]
    public void CellCentres_AreSymmetricAboutZero()
    {
        // Centres sit at (i + 0.5) / n, so the first and last cell angles mirror each other on the first frequency.
        (Tensor cos, Tensor sin) = Dinov3RopeTables.Build(1, prefix: 0, gridH: 3, gridW: 1, headDim: 64);
        try
        {
            float* c = (float*)cos.DataPointer, s = (float*)sin.DataPointer;
            // Row axis, frequency 0 (period 1): angle = 2*pi*coord, coords -2/3, 0, 2/3.
            Assert.Equal((float)Math.Cos(2 * Math.PI * (-2.0 / 3.0)), c[0], 5);
            Assert.Equal(1f, c[1 * 64], 5);
            Assert.Equal(c[0], c[2 * 64], 5);
            Assert.Equal(-s[0], s[2 * 64], 5);
        }
        finally
        {
            cos.Dispose();
            sin.Dispose();
        }
    }

    private static float[] PeriodsOf64() => Dinov3RopeTables.Periods(64);
}

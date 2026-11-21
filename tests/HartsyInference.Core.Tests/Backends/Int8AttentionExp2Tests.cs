using HartsyInference.Core.Backends.Attention;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Backends;

/// <summary><see cref="Int8AttentionExp2.Evaluate"/> against the true <c>2^x</c>. Bit-equality with its shader twin is a
/// GPU test; this is its accuracy, which that equality says nothing about.
/// <para>The bound was fixed before the first run: within 4 units in the last place of the correctly rounded value over
/// [-125, 127], exact at every whole number, and a tile's largest score still mapping to the byte 255.</para></summary>
public sealed class Int8AttentionExp2Tests(ITestOutputHelper output)
{
    private static double UlpsOff(float actual, double x)
    {
        double exact = Math.Pow(2.0, x);
        float nearest = (float)exact;
        double ulp = (double)MathF.BitIncrement(nearest) - nearest;
        return Math.Abs(actual - exact) / ulp;
    }

    [Fact]
    public void IsWithinFourUnitsInTheLastPlace()
    {
        Random random = new(81);
        double worst = 0;
        float worstAt = 0f;
        for (int i = 0; i < 3_000_000; i++)
        {
            float x = i % 3 == 0 ? (float)(random.NextDouble() * 252 - 125)
                : i % 3 == 1 ? (float)(random.NextDouble() * 48 - 40) : (float)(random.NextDouble() - 0.5);
            double off = UlpsOff(Int8AttentionExp2.Evaluate(x), x);
            if (off > worst) (worst, worstAt) = (off, x);
        }
        output.WriteLine($"worst: {worst:F3} units in the last place at {worstAt:R}");
        Assert.True(worst <= 4.0, $"{worst} units in the last place at {worstAt:R}");
    }

    [Fact]
    public void WholeNumbersAreExact()
    {
        for (int k = -125; k <= 127; k++) Assert.Equal(MathF.ScaleB(1f, k), Int8AttentionExp2.Evaluate(k));
    }

    [Fact]
    public void TheEdgesAreAsStated()
    {
        Assert.True(float.IsNaN(Int8AttentionExp2.Evaluate(float.NaN)));
        Assert.Equal(float.PositiveInfinity, Int8AttentionExp2.Evaluate(float.PositiveInfinity));
        Assert.Equal(float.PositiveInfinity, Int8AttentionExp2.Evaluate(MathF.BitIncrement(127f)));
        Assert.Equal(0f, Int8AttentionExp2.Evaluate(MathF.BitDecrement(-125f)));
        Assert.Equal(0f, Int8AttentionExp2.Evaluate(float.NegativeInfinity));
        Assert.Equal(MathF.ScaleB(1f, -125), Int8AttentionExp2.Evaluate(-125f));
        // The smallest results are still normal floats.
        Assert.True(Int8AttentionExp2.Evaluate(-125f) >= 1.17549435e-38f);
        Assert.True(Int8AttentionExp2.Evaluate(MathF.BitDecrement(-124.5f)) >= 1.17549435e-38f);
    }

    /// <summary>u8 probabilities are taken against the tile's largest score less <see cref="Int8AttentionV2Reference.U8Offset"/>,
    /// so that score itself has to come out as 255. It does while the scores are below about 2^16 in magnitude, where
    /// their last place is still finer than the offset needs.</summary>
    [Fact]
    public void ATilesLargestScoreMapsTo255()
    {
        Random random = new(82);
        for (int i = 0; i < 1_000_000; i++)
        {
            float tileMax = (float)(random.NextDouble() * 2000 - 1000);
            float tileM = tileMax - Int8AttentionV2Reference.U8Offset;
            float p = Int8AttentionExp2.Evaluate(tileMax - tileM);
            Assert.True((int)MathF.Round(Math.Clamp(p, 0f, 255f)) == 255, $"a tile max of {tileMax:R} gives {p:R}");
        }
    }
}

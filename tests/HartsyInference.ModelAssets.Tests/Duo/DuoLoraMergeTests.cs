using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Duo;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.Duo;

public sealed class DuoLoraMergeTests
{
    private static float Bf16(ushort bits) => BitConverter.UInt32BitsToSingle((uint)bits << 16);

    private static ushort Bf16Bits(float value) => BFloat16Bits.FromFloat(value);

    private static Tensor F32(long rows, long cols, params float[] values)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static Tensor Bf16Tensor(long rows, long cols, params float[] values)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.BF16);
        for (int i = 0; i < values.Length; i++) t.AsSpan<ushort>()[i] = Bf16Bits(values[i]);
        return t;
    }

    [Fact]
    public void RoundNearestEven_MatchesTorchOnTiesAndSpecials()
    {
        Assert.Equal((ushort)0x3f80, Bf16Bits(1.0f + 1f / 256f));      // tie between 1.0 and 1.0078125 goes to even (1.0)
        Assert.Equal((ushort)0x3f82, Bf16Bits(1.0f + 3f / 256f));      // tie goes up to the even neighbour
        Assert.Equal((ushort)0x3f81, Bf16Bits(1.0f + 1f / 128f));
        Assert.Equal((ushort)0x7f80, Bf16Bits(float.PositiveInfinity));
        Assert.True(float.IsNaN(Bf16(Bf16Bits(float.NaN))));
        Assert.Equal((ushort)0x7f80, Bf16Bits(3.4028235e38f));         // rounds up into infinity, as torch does
    }

    [Fact]
    public void MergeRows_RoundsTheDeltaToBf16BeforeTheAddInsteadOfRoundingOnce()
    {
        // delta 0.003907 is just above half a BF16 step at 1.0. Rounded first it becomes exactly 2^-8, the sum is a tie and goes to 1.0;
        // a single rounding of the F32 sum would reach 1.0078125.
        using Tensor weight = Bf16Tensor(1, 1, 1.0f);
        using Tensor a = F32(1, 1, 1f);
        using Tensor b = F32(1, 1, 0.003907f);

        DuoLoraMerge.MergeRows(weight, 0, a, b, scale: 1f, swapHalves: false);

        Assert.Equal((ushort)0x3f80, weight.AsSpan<ushort>()[0]);
        Assert.NotEqual(Bf16Bits(1.0f + 0.003907f), weight.AsSpan<ushort>()[0]);
    }

    [Fact]
    public void MergeRows_AppliesTheScaleAndOnlyTouchesItsRowSlice()
    {
        using Tensor weight = Bf16Tensor(4, 2, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f);
        using Tensor a = F32(1, 2, 1f, 2f);
        using Tensor b = F32(2, 1, 4f, 8f);

        DuoLoraMerge.MergeRows(weight, rowStart: 1, a, b, scale: 0.5f, swapHalves: false);

        float[] values = weight.AsSpan<ushort>().ToArray().Select(Bf16).ToArray();
        Assert.Equal(new[] { 1f, 1f, 3f, 5f, 5f, 9f, 1f, 1f }, values);
    }

    [Fact]
    public void MergeRows_SwapsTheGateAndValueHalvesOfB()
    {
        // Weight rows are [gate; value]; B arrives as [value; gate], so weight row 0 (gate) takes B row 2 and weight row 2 (value) takes B row 0.
        using Tensor weight = F32(4, 1, 0f, 0f, 0f, 0f);
        using Tensor a = F32(1, 1, 1f);
        using Tensor b = F32(4, 1, 10f, 11f, 20f, 21f);

        DuoLoraMerge.MergeRows(weight, 0, a, b, 1f, swapHalves: true);

        Assert.Equal(new[] { 20f, 21f, 10f, 11f }, weight.AsSpan<float>().ToArray());
    }

    [Fact]
    public void MergeRows_MatchesAnIndependentDoubleComputationOnRandomData()
    {
        Random rng = new(7);
        const int Rows = 37, Rank = 6, In = 53;
        float[] av = Enumerable.Range(0, Rank * In).Select(_ => (float)rng.NextDouble() - 0.5f).ToArray();
        float[] bv = Enumerable.Range(0, Rows * Rank).Select(_ => (float)rng.NextDouble() - 0.5f).ToArray();
        float[] wv = Enumerable.Range(0, Rows * In).Select(_ => (float)rng.NextDouble() * 4 - 2).ToArray();
        using Tensor a = F32(Rank, In, av), b = F32(Rows, Rank, bv), f32Weight = F32(Rows, In, wv);
        using Tensor bf16Weight = Bf16Tensor(Rows, In, wv);

        DuoLoraMerge.MergeRows(f32Weight, 0, a, b, 0.25f, false);
        DuoLoraMerge.MergeRows(bf16Weight, 0, a, b, 0.25f, false);

        int bf16Differences = 0;
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < In; c++)
            {
                double delta = 0;
                for (int k = 0; k < Rank; k++) delta += (double)bv[r * Rank + k] * av[k * In + c];
                float expectedDelta = (float)(delta * 0.25);
                Assert.Equal(wv[r * In + c] + expectedDelta, f32Weight.AsSpan<float>()[r * In + c], 1e-5f);
                float baseValue = Bf16(Bf16Bits(wv[r * In + c]));
                ushort expected = Bf16Bits(baseValue + Bf16(Bf16Bits(expectedDelta)));
                if (expected != bf16Weight.AsSpan<ushort>()[r * In + c]) bf16Differences++;
            }
        }
        // The reference sums in double, the merge in F32; a delta next to a BF16 rounding boundary may land on the other side, which is rare.
        Assert.True(bf16Differences <= Rows * In / 200, $"{bf16Differences} of {Rows * In} BF16 results differ from the double reference.");
    }

    [Fact]
    public void MergeRows_ValidatesShapesAndDTypes()
    {
        using Tensor weight = Bf16Tensor(2, 2, 0, 0, 0, 0);
        using Tensor a = F32(1, 2, 1, 1), b = F32(2, 1, 1, 1);
        using Tensor wrongRank = F32(2, 2, 0, 0, 0, 0), halfA = a.CastTo(DType.F16);
        Assert.Throws<ArgumentException>(() => DuoLoraMerge.MergeRows(weight, 1, a, b, 1f, false));      // slice does not fit
        Assert.Throws<ArgumentException>(() => DuoLoraMerge.MergeRows(weight, 0, wrongRank, b, 1f, false)); // A rank 2 vs B rank 1
        Assert.Throws<ArgumentException>(() => DuoLoraMerge.MergeRows(weight, 0, halfA, b, 1f, false));
        using Tensor f16Weight = new(new TensorShape(2, 2), DType.F16);
        Assert.Throws<NotSupportedException>(() => DuoLoraMerge.MergeRows(f16Weight, 0, a, b, 1f, false));
    }
}

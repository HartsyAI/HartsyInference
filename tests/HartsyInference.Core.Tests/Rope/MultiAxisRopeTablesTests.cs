using System.Runtime.CompilerServices;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Rope;

/// <summary>Frequency plan validation and per-pair tables, checked against the closed-form formula and bit-for-bit
/// against <see cref="RopeFrequencyBuilder"/>.</summary>
public sealed unsafe class MultiAxisRopeTablesTests
{
    [Fact]
    public void Create_RejectsInvalidSpecs()
    {
        Assert.ThrowsAny<ArgumentException>(() => RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [], Theta = 10000 }));
        Assert.ThrowsAny<ArgumentException>(() => RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4, 5], Theta = 10000 }));
        Assert.ThrowsAny<ArgumentException>(() => RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4, 0], Theta = 10000 }));
        Assert.ThrowsAny<ArgumentException>(() => RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4], Theta = 0 }));
        Assert.ThrowsAny<ArgumentException>(() => RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4], Theta = double.NaN }));
        Assert.ThrowsAny<ArgumentException>(() =>
            RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4, 4], Theta = 10000, NtkScales = [2.0] }));
        Assert.ThrowsAny<ArgumentException>(() =>
            RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4], Theta = 10000, NtkScales = [0.0] }));
        Assert.ThrowsAny<ArgumentException>(() =>
            RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4], Theta = 10000, NtkScales = [double.NaN] }));
    }

    [Fact]
    public void Plan_ExposesHeadDimPairsAndAxisDims()
    {
        RopeFrequencyPlan plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4, 6, 2], Theta = 10000 });

        Assert.Equal(12, plan.HeadDim);
        Assert.Equal(6, plan.Pairs);
        Assert.Equal(3, plan.Axes);
        Assert.Equal([4, 6, 2], plan.AxisDims.ToArray());
    }

    [Fact]
    public void ComputePairs_MatchesFrequencyBuilderBitForBit_PerAxis()
    {
        RopeFrequencyPlan plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4, 4], Theta = 10000 });
        float[] positions = [3f, -2f];
        float[] cos = new float[4];
        float[] sin = new float[4];

        MultiAxisRopeTables.ComputePairs(plan, positions, 1, RopeAngleOrder.MultiplyByInverse, cos, sin);

        double[] inv = RopeFrequencyBuilder.ComputeInvFreq(4, 10000);
        for (int k = 0; k < 2; k++)
        {
            Assert.Equal(BitConverter.SingleToInt32Bits((float)Math.Cos(3.0 * inv[k])), BitConverter.SingleToInt32Bits(cos[k]));
            Assert.Equal(BitConverter.SingleToInt32Bits((float)Math.Sin(3.0 * inv[k])), BitConverter.SingleToInt32Bits(sin[k]));
            Assert.Equal(BitConverter.SingleToInt32Bits((float)Math.Cos(-2.0 * inv[k])), BitConverter.SingleToInt32Bits(cos[2 + k]));
            Assert.Equal(BitConverter.SingleToInt32Bits((float)Math.Sin(-2.0 * inv[k])), BitConverter.SingleToInt32Bits(sin[2 + k]));
        }
    }

    [Fact]
    public void ComputePairs_NtkScale_UsesScaledBase()
    {
        RopeFrequencyPlan plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4], Theta = 10000, NtkScales = [2.0] });
        float[] positions = [5f];
        float[] cos = new float[2];
        float[] sin = new float[2];

        MultiAxisRopeTables.ComputePairs(plan, positions, 1, RopeAngleOrder.MultiplyByInverse, cos, sin);

        double scaledBase = 10000 * RopeFrequencyBuilder.NtkThetaFactor(2.0, 4);
        double[] inv = RopeFrequencyBuilder.ComputeInvFreq(4, scaledBase);
        for (int k = 0; k < 2; k++)
            Assert.Equal(BitConverter.SingleToInt32Bits((float)Math.Cos(5.0 * inv[k])), BitConverter.SingleToInt32Bits(cos[k]));
    }

    [Fact]
    public void ComputePairs_DivideOrder_UsesTheDenominator()
    {
        RopeFrequencyPlan plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4], Theta = 10000 });
        float[] positions = [3.7f];
        float[] cos = new float[2];
        float[] sin = new float[2];

        MultiAxisRopeTables.ComputePairs(plan, positions, 1, RopeAngleOrder.DivideByDenominator, cos, sin);

        for (int k = 0; k < 2; k++)
        {
            double denominator = Math.Pow(10000, (double)(2 * k) / 4);
            Assert.Equal(BitConverter.SingleToInt32Bits((float)Math.Cos(3.7 / denominator)), BitConverter.SingleToInt32Bits(cos[k]));
        }
    }

    [Fact]
    public void NegativeZeroPosition_PropagatesSignedZeroThroughExpander()
    {
        RopeFrequencyPlan plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4], Theta = 10000 });
        float[] positions = [-0f];
        float[] cos = new float[2];
        float[] sin = new float[2];
        float[] cosRows = new float[4];
        float[] sinRows = new float[4];

        MultiAxisRopeTables.ComputePairs(plan, positions, 1, RopeAngleOrder.MultiplyByInverse, cos, sin);
        RopeTableExpander.Expand(cos, sin, 1, 4, RopeTableLayout.InterleavedPair, 0, RopeTablePadding.Identity, cosRows, sinRows);

        Assert.Equal(BitConverter.SingleToInt32Bits(-0f), BitConverter.SingleToInt32Bits(sin[0]));
        Assert.Equal(BitConverter.SingleToInt32Bits(-0f), BitConverter.SingleToInt32Bits(sinRows[0]));
        Assert.Equal(BitConverter.SingleToInt32Bits(-0f), BitConverter.SingleToInt32Bits(sinRows[1]));
    }

    [Fact]
    public void ComputePairs_RejectsNonFinitePositions()
    {
        RopeFrequencyPlan plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4], Theta = 10000 });
        float[] positions = [float.NaN];
        float[] cos = new float[2];
        float[] sin = new float[2];

        Assert.ThrowsAny<ArgumentException>(() =>
            MultiAxisRopeTables.ComputePairs(plan, positions, 1, RopeAngleOrder.MultiplyByInverse, cos, sin));
    }

    [Fact]
    public void BuildRows_PartialRotary_KeepsLeadingPairsThenExpands()
    {
        RopeFrequencyPlan plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4, 4], Theta = 10000 });
        float[] positions = [1f, 2f, -3f, 0.5f];
        float[] fullCos = new float[8];
        float[] fullSin = new float[8];
        MultiAxisRopeTables.ComputePairs(plan, positions, 2, RopeAngleOrder.MultiplyByInverse, fullCos, fullSin);
        float[] compactCos = [fullCos[0], fullCos[1], fullCos[4], fullCos[5]];
        float[] compactSin = [fullSin[0], fullSin[1], fullSin[4], fullSin[5]];
        float[] expectedCos = new float[16];
        float[] expectedSin = new float[16];
        RopeTableExpander.Expand(compactCos, compactSin, 2, 8, RopeTableLayout.InterleavedPrefix, 4, RopeTablePadding.Identity,
            expectedCos, expectedSin);
        float[] actualCos = new float[16];
        float[] actualSin = new float[16];

        MultiAxisRopeTables.BuildRows(plan, positions, 2, RopeAngleOrder.MultiplyByInverse, RopeTableLayout.InterleavedPrefix, 4,
            RopeTablePadding.Identity, actualCos, actualSin);

        Assert.Equal(expectedCos, actualCos);
        Assert.Equal(expectedSin, actualSin);
    }

    [Fact]
    public void BuildTensors_ZeroTokens_IsRejectedBeforeAllocation()
    {
        RopeFrequencyPlan plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4, 4], Theta = 10000 });

        Assert.ThrowsAny<ArgumentException>(() =>
            MultiAxisRopeTables.BuildTensors(plan, [], 0, RopeAngleOrder.MultiplyByInverse, RopeTableLayout.InterleavedPair, 0,
                RopeTablePadding.Identity));
    }

    [Fact]
    public void BuildTensors_ReturnsOwnedTensorsWithTheExpectedValues()
    {
        RopeFrequencyPlan plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4, 4], Theta = 10000 });
        float[] positions = [1f, 2f, -3f, 0.5f];
        float[] expectedCos = new float[16];
        float[] expectedSin = new float[16];
        MultiAxisRopeTables.BuildRows(plan, positions, 2, RopeAngleOrder.MultiplyByInverse, RopeTableLayout.InterleavedPair, 0,
            RopeTablePadding.Identity, expectedCos, expectedSin);

        (Tensor cos, Tensor sin) = MultiAxisRopeTables.BuildTensors(plan, positions, 2, RopeAngleOrder.MultiplyByInverse,
            RopeTableLayout.InterleavedPair, 0, RopeTablePadding.Identity);
        try
        {
            Assert.Equal(2, (int)cos.Shape[0]);
            Assert.Equal(8, (int)cos.Shape[1]);
            ReadOnlySpan<float> actualCos = new((void*)cos.DataPointer, 16);
            ReadOnlySpan<float> actualSin = new((void*)sin.DataPointer, 16);
            for (int i = 0; i < 16; i++)
            {
                Assert.Equal(BitConverter.SingleToInt32Bits(expectedCos[i]), BitConverter.SingleToInt32Bits(actualCos[i]));
                Assert.Equal(BitConverter.SingleToInt32Bits(expectedSin[i]), BitConverter.SingleToInt32Bits(actualSin[i]));
            }
        }
        finally
        {
            cos.Dispose();
            sin.Dispose();
        }
    }
}

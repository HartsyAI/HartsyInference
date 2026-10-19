using HartsyInference.Core.Rope;
using Xunit;

namespace HartsyInference.Core.Tests.Rope;

/// <summary>Layout, padding and storage rules of <see cref="RopeTableExpander"/>, checked on hand-written rows.</summary>
public sealed class RopeTableExpanderTests
{
    [Fact]
    public void RotateHalf_FullWidth_RepeatsEachPairAcrossBothHalves()
    {
        float[] cos = [0.1f, 0.2f, 0.3f, 0.4f];
        float[] sin = [-0.1f, -0.2f, -0.3f, -0.4f];
        float[] cosRows = new float[8];
        float[] sinRows = new float[8];

        RopeTableExpander.Expand(cos, sin, 2, 4, RopeTableLayout.RotateHalf, 0, RopeTablePadding.Identity, cosRows, sinRows);

        Assert.Equal([0.1f, 0.2f, 0.1f, 0.2f, 0.3f, 0.4f, 0.3f, 0.4f], cosRows);
        Assert.Equal([-0.1f, -0.2f, -0.1f, -0.2f, -0.3f, -0.4f, -0.3f, -0.4f], sinRows);
    }

    [Fact]
    public void RotateHalf_PartialRotary_RotatesLeadingChannelsAndPadsTheRest()
    {
        float[] cos = [0.5f];
        float[] sin = [0.25f];
        float[] cosRows = new float[4];
        float[] sinRows = new float[4];

        RopeTableExpander.Expand(cos, sin, 1, 4, RopeTableLayout.RotateHalf, 2, RopeTablePadding.Identity, cosRows, sinRows);

        Assert.Equal([0.5f, 0.5f, 1f, 1f], cosRows);
        Assert.Equal([0.25f, 0.25f, 0f, 0f], sinRows);
    }

    [Fact]
    public void RotateHalf_PartialRotary_ZeroPaddingWritesZeros()
    {
        float[] cos = [0.5f];
        float[] sin = [0.25f];
        float[] cosRows = new float[4];
        float[] sinRows = new float[4];

        RopeTableExpander.Expand(cos, sin, 1, 4, RopeTableLayout.RotateHalf, 2, RopeTablePadding.Zero, cosRows, sinRows);

        Assert.Equal([0.5f, 0.5f, 0f, 0f], cosRows);
        Assert.Equal([0.25f, 0.25f, 0f, 0f], sinRows);
    }

    [Fact]
    public void InterleavedPrefix_FullWidth_WritesFirstHalfAndPadsTheRest()
    {
        float[] cos = [0.1f, 0.2f];
        float[] sin = [0.3f, 0.4f];
        float[] cosRows = new float[4];
        float[] sinRows = new float[4];

        RopeTableExpander.Expand(cos, sin, 1, 4, RopeTableLayout.InterleavedPrefix, 0, RopeTablePadding.Identity, cosRows, sinRows);

        Assert.Equal([0.1f, 0.2f, 1f, 1f], cosRows);
        Assert.Equal([0.3f, 0.4f, 0f, 0f], sinRows);
    }

    [Fact]
    public void InterleavedPrefix_PartialRotary_UsesLeadingPairsOnly()
    {
        float[] cos = [0.1f];
        float[] sin = [0.3f];
        float[] cosRows = new float[4];
        float[] sinRows = new float[4];

        RopeTableExpander.Expand(cos, sin, 1, 4, RopeTableLayout.InterleavedPrefix, 2, RopeTablePadding.Zero, cosRows, sinRows);

        Assert.Equal([0.1f, 0f, 0f, 0f], cosRows);
        Assert.Equal([0.3f, 0f, 0f, 0f], sinRows);
    }

    [Fact]
    public void InterleavedPair_DuplicatesEachPair()
    {
        float[] cos = [0.1f, 0.2f];
        float[] sin = [0.3f, 0.4f];
        float[] cosRows = new float[4];
        float[] sinRows = new float[4];

        RopeTableExpander.Expand(cos, sin, 1, 4, RopeTableLayout.InterleavedPair, 0, RopeTablePadding.Identity, cosRows, sinRows);

        Assert.Equal([0.1f, 0.1f, 0.2f, 0.2f], cosRows);
        Assert.Equal([0.3f, 0.3f, 0.4f, 0.4f], sinRows);
    }

    [Fact]
    public void InterleavedPair_PartialRotary_IsRejected()
    {
        float[] cos = [0.1f];
        float[] sin = [0.3f];
        float[] cosRows = new float[4];
        float[] sinRows = new float[4];

        Assert.ThrowsAny<ArgumentException>(() =>
            RopeTableExpander.Expand(cos, sin, 1, 4, RopeTableLayout.InterleavedPair, 2, RopeTablePadding.Identity, cosRows, sinRows));
    }

    [Fact]
    public void ZeroTokens_WithEmptyDestinations_IsANoOp()
    {
        float[] cosRows = [];
        float[] sinRows = [];

        RopeTableExpander.Expand([], [], 0, 4, RopeTableLayout.RotateHalf, 0, RopeTablePadding.Identity, cosRows, sinRows);

        Assert.Empty(cosRows);
        Assert.Empty(sinRows);
    }

    [Fact]
    public void Destinations_AreFullyOverwritten_RegardlessOfPriorContents()
    {
        float[] cos = [0.1f, 0.2f];
        float[] sin = [0.3f, 0.4f];
        float[] cosRows = Enumerable.Repeat(float.NaN, 4).ToArray();
        float[] sinRows = Enumerable.Repeat(float.NaN, 4).ToArray();

        RopeTableExpander.Expand(cos, sin, 1, 4, RopeTableLayout.InterleavedPrefix, 0, RopeTablePadding.Zero, cosRows, sinRows);

        Assert.All(cosRows, value => Assert.False(float.IsNaN(value)));
        Assert.All(sinRows, value => Assert.False(float.IsNaN(value)));
    }

    [Fact]
    public void NegativeZero_IsPreservedBitwise()
    {
        float[] cos = [-0f];
        float[] sin = [-0f];
        float[] cosRows = new float[2];
        float[] sinRows = new float[2];

        RopeTableExpander.Expand(cos, sin, 1, 2, RopeTableLayout.RotateHalf, 0, RopeTablePadding.Identity, cosRows, sinRows);

        Assert.Equal(BitConverter.SingleToInt32Bits(-0f), BitConverter.SingleToInt32Bits(cosRows[0]));
        Assert.Equal(BitConverter.SingleToInt32Bits(-0f), BitConverter.SingleToInt32Bits(sinRows[1]));
    }

    [Fact]
    public void Overlap_BetweenInputAndDestination_IsRejected()
    {
        float[] shared = new float[6];
        float[] sin = [0.3f, 0.4f];
        float[] sinRows = new float[4];

        Assert.ThrowsAny<ArgumentException>(() =>
            RopeTableExpander.Expand(shared.AsSpan(0, 2), sin, 1, 4, RopeTableLayout.InterleavedPrefix, 0,
                RopeTablePadding.Identity, shared.AsSpan(1, 4), sinRows));
    }

    [Fact]
    public void Overlap_BetweenCosineAndSineDestinations_IsRejected()
    {
        float[] cos = [0.1f, 0.2f];
        float[] sin = [0.3f, 0.4f];
        float[] rows = new float[8];

        Assert.ThrowsAny<ArgumentException>(() =>
            RopeTableExpander.Expand(cos, sin, 1, 4, RopeTableLayout.InterleavedPrefix, 0,
                RopeTablePadding.Identity, rows.AsSpan(0, 4), rows.AsSpan(2, 4)));
    }

    [Fact]
    public void InputLengthMismatch_IsRejected()
    {
        float[] cos = [0.1f];
        float[] sin = [0.3f, 0.4f];
        float[] cosRows = new float[4];
        float[] sinRows = new float[4];

        Assert.ThrowsAny<ArgumentException>(() =>
            RopeTableExpander.Expand(cos, sin, 1, 4, RopeTableLayout.InterleavedPrefix, 0, RopeTablePadding.Identity, cosRows, sinRows));
    }

    [Fact]
    public void EffectiveRotaryDim_AcceptsZeroAsFullWidth_AndRejectsInvalidWidths()
    {
        Assert.Equal(4, RopeTableExpander.EffectiveRotaryDim(4, 0, RopeTableLayout.RotateHalf));
        Assert.Equal(2, RopeTableExpander.EffectiveRotaryDim(4, 2, RopeTableLayout.RotateHalf));
        Assert.ThrowsAny<ArgumentException>(() => RopeTableExpander.EffectiveRotaryDim(5, 0, RopeTableLayout.RotateHalf));
        Assert.ThrowsAny<ArgumentException>(() => RopeTableExpander.EffectiveRotaryDim(4, 3, RopeTableLayout.RotateHalf));
        Assert.ThrowsAny<ArgumentException>(() => RopeTableExpander.EffectiveRotaryDim(4, 6, RopeTableLayout.RotateHalf));
        Assert.ThrowsAny<ArgumentException>(() => RopeTableExpander.EffectiveRotaryDim(4, -2, RopeTableLayout.RotateHalf));
    }
}

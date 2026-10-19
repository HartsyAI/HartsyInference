using HartsyInference.Core.Rope;
using Xunit;

namespace HartsyInference.Core.Tests.Rope;

/// <summary>Multimodal RoPE: pair-to-axis maps in both layouts (transformers' <c>apply_interleaved_mrope</c> slice bounds
/// included), <c>get_rope_index</c> positions around an image, and the cos/sin rows, which must equal the plain
/// partial-rotary table when every axis carries the same position.</summary>
public sealed class MRopeTests
{
    [Fact]
    public void Interleaved_Qwen35Sections_ClaimEveryPair()
    {
        byte[] axes = MRopePairAxes.Build([11, 11, 10, 0], 32, MRopeLayout.Interleaved);
        for (int j = 0; j < 32; j++)
            Assert.Equal(j % 3, axes[j]);
    }

    [Fact]
    public void Interleaved_SectionBound_SendsLatePairsToTheTemporalAxis()
    {
        byte[] narrowWidth = MRopePairAxes.Build([5, 4, 3], 12, MRopeLayout.Interleaved);
        Assert.Equal(new byte[] { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0 }, narrowWidth);
        byte[] qwen3Vl = MRopePairAxes.Build([24, 20, 20], 64, MRopeLayout.Interleaved);
        Assert.Equal(new byte[] { 0, 0, 0, 0 }, qwen3Vl[60..]);
        Assert.Equal(1, qwen3Vl[58]);
        Assert.Equal(2, qwen3Vl[59]);
    }

    [Fact]
    public void Sectioned_LaysAxesOutContiguously()
    {
        byte[] axes = MRopePairAxes.Build([16, 24, 24], 64, MRopeLayout.Sectioned);
        Assert.All(axes[..16], a => Assert.Equal(0, a));
        Assert.All(axes[16..40], a => Assert.Equal(1, a));
        Assert.All(axes[40..], a => Assert.Equal(2, a));
        Assert.Throws<ArgumentException>(() => MRopePairAxes.Build([16, 24, 23], 64, MRopeLayout.Sectioned));
    }

    [Fact]
    public void Positions_PlaceTheImageGridAndResumeTextPastIt()
    {
        const int P = 9;
        int[] ids = [1, 2, P, P, P, P, P, P, 3];
        int[] positions = MRopePositions.Build(ids, P, [new MRopeImageGrid(1, 2, 3)], out int next);
        int seq = ids.Length;
        Assert.Equal(new[] { 0, 1, 2, 2, 2, 2, 2, 2, 5 }, positions[..seq]);
        Assert.Equal(new[] { 0, 1, 2, 2, 2, 3, 3, 3, 5 }, positions[seq..(2 * seq)]);
        Assert.Equal(new[] { 0, 1, 2, 3, 4, 2, 3, 4, 5 }, positions[(2 * seq)..]);
        Assert.Equal(6, next);
    }

    [Fact]
    public void Positions_RefusePlaceholderRunsThatDoNotFitTheirGrids()
    {
        const int P = 9;
        Assert.Throws<ArgumentException>(() => MRopePositions.Build([1, P, P, P, 2], P, [new MRopeImageGrid(1, 2, 2)], out _));
        Assert.Throws<ArgumentException>(() => MRopePositions.Build([1, P, P, P, P, P, 2], P, [new MRopeImageGrid(1, 2, 2)], out _));
        Assert.Throws<ArgumentException>(() => MRopePositions.Build([1, 2], P, [new MRopeImageGrid(1, 1, 1)], out _));
    }

    [Fact]
    public void Table_WithEqualAxes_IsThePartialRotaryTable()
    {
        const int tokens = 5, headDim = 16, rotaryDim = 8;
        int[] positions = new int[3 * tokens];
        for (int a = 0; a < 3; a++)
            for (int s = 0; s < tokens; s++) positions[a * tokens + s] = 7 + s;
        (double[] invFreq, double scaling) = RopeFrequencyBuilder.Build(rotaryDim, 10000.0, RopeScaling.None, 12);
        byte[] axes = MRopePairAxes.Build([2, 1, 1], rotaryDim / 2, MRopeLayout.Interleaved);
        float[] cos = new float[tokens * headDim], sin = new float[tokens * headDim];
        MRopeTable.BuildRows(positions, tokens, 1, 3, invFreq, scaling, axes, headDim, rotaryDim, cos, sin);
        for (int r = 0; r < 3; r++)
        {
            for (int j = 0; j < rotaryDim / 2; j++)
            {
                double angle = (7 + 1 + r) * invFreq[j];
                Assert.Equal((float)Math.Cos(angle), cos[r * headDim + j]);
                Assert.Equal((float)Math.Cos(angle), cos[r * headDim + j + rotaryDim / 2]);
                Assert.Equal((float)Math.Sin(angle), sin[r * headDim + j + rotaryDim / 2]);
            }
            Assert.Equal(0f, cos[r * headDim + rotaryDim]);
        }
    }
}

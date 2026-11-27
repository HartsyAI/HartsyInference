using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Sam3dObjects;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamPositionEmbeddingTests
{
    [Fact]
    public void Absolute_LaysOutSinThenCosPerCoordinateAndZeroPads()
    {
        // 8 channels over 3 coordinates: freqDim = 8 / 3 / 2 = 1, so each coordinate takes [sin, cos] and two columns are padding.
        using Tensor embedding = SamPositionEmbedding.Absolute(8, 3, [2, 0, 5]);

        Assert.Equal(new TensorShape(1, 8), embedding.Shape);
        float[] expected = [MathF.Sin(2f), MathF.Cos(2f), 0f, 1f, MathF.Sin(5f), MathF.Cos(5f), 0f, 0f];
        float[] got = embedding.AsReadOnlySpan<float>().ToArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], got[i], 5);
    }

    [Fact]
    public void Absolute_UsesGeometricFrequencies()
    {
        // 12 channels over 3 coordinates: freqDim 2, frequencies 1 and 1 / 10000^(1/2) = 0.01.
        using Tensor embedding = SamPositionEmbedding.Absolute(12, 3, [7, 0, 0]);

        ReadOnlySpan<float> got = embedding.AsReadOnlySpan<float>();
        Assert.Equal(MathF.Sin(7f), got[0], 5);
        Assert.Equal(MathF.Sin(0.07f), got[1], 5);
        Assert.Equal(MathF.Cos(7f), got[2], 5);
        Assert.Equal(MathF.Cos(0.07f), got[3], 5);
    }

    [Fact]
    public void ShapeGrid_ListsTheCellsWithXSlowestAndSkipsNothing()
    {
        using Tensor grid = SamPositionEmbedding.ShapeGrid(8, 2, 1);

        Assert.Equal(new TensorShape(8, 8), grid.Shape);
        ReadOnlySpan<float> got = grid.AsReadOnlySpan<float>();
        // Row 1 is cell (0, 0, 1): x = 0 -> [0, 1], y = 0 -> [0, 1], z = 1 -> [sin 1, cos 1].
        Assert.Equal(MathF.Sin(1f), got[8 + 4], 5);
        Assert.Equal(MathF.Cos(1f), got[8 + 5], 5);
        // Row 4 is cell (1, 0, 0): x = 1 comes first.
        Assert.Equal(MathF.Sin(1f), got[4 * 8 + 0], 5);
        // A patch size divides the resolution.
        using Tensor coarse = SamPositionEmbedding.ShapeGrid(8, 4, 2);
        Assert.Equal(8, coarse.Shape[0]);
        Assert.Throws<ArgumentException>(() => SamPositionEmbedding.ShapeGrid(8, 5, 2));
    }

    [Fact]
    public void Absolute_RejectsRaggedCoordinates()
    {
        Assert.Throws<ArgumentException>(() => SamPositionEmbedding.Absolute(8, 3, [1, 2]));
    }

    [Fact]
    public void Sinusoid_PutsCosinesFirstAndPadsAnOddSizeWithZero()
    {
        float[] even = new float[4];
        SamTimestepEmbedder.Sinusoid(0f, even);
        Assert.Equal([1f, 1f, 0f, 0f], even);

        float[] odd = new float[5];
        SamTimestepEmbedder.Sinusoid(2f, odd);
        Assert.Equal(MathF.Cos(2f), odd[0], 5);
        Assert.Equal(MathF.Cos(2f * 0.01f), odd[1], 5);
        Assert.Equal(MathF.Sin(2f), odd[2], 5);
        Assert.Equal(0f, odd[4]);
    }
}

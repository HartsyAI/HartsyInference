using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>Tests for <see cref="TensorShape"/>.</summary>
public sealed class TensorShapeTests
{
    [Fact]
    public void Indexer_ReturnsCorrectDimension()
    {
        TensorShape shape = new TensorShape(2, 3, 4);

        Assert.Equal(2, shape[0]);
        Assert.Equal(3, shape[1]);
        Assert.Equal(4, shape[2]);
    }

    [Fact]
    public void Stride_RowMajor_Correct()
    {
        TensorShape shape = new TensorShape(2, 3, 4);

        Assert.Equal(12, shape.Stride(0));
        Assert.Equal(4, shape.Stride(1));
        Assert.Equal(1, shape.Stride(2));
    }

    [Fact]
    public void Unsqueeze_InsertsUnitDim()
    {
        TensorShape shape = new TensorShape(3, 4);

        TensorShape unsqueezed = shape.Unsqueeze(0);

        Assert.Equal(3, unsqueezed.Rank);
        Assert.Equal(1, unsqueezed[0]);
        Assert.Equal(3, unsqueezed[1]);
        Assert.Equal(4, unsqueezed[2]);
    }

    [Fact]
    public void Squeeze_NonUnitDim_Throws()
    {
        TensorShape shape = new TensorShape(2, 3, 4);

        Assert.Throws<InvalidOperationException>(() => shape.Squeeze(0));
    }

    [Fact]
    public void Permute_TransposesShape()
    {
        TensorShape shape = new TensorShape(2, 3, 4);

        TensorShape permuted = shape.Permute(new ReadOnlySpan<int>([2, 0, 1]));

        Assert.Equal(3, permuted.Rank);
        Assert.Equal(4, permuted[0]);
        Assert.Equal(2, permuted[1]);
        Assert.Equal(3, permuted[2]);
    }

    [Fact]
    public void Equality_DifferentShape_NotEqual()
    {
        TensorShape a = new TensorShape(2, 3, 4);
        TensorShape b = new TensorShape(2, 4, 3);

        Assert.NotEqual(a, b);
        Assert.True(a != b);
    }
}

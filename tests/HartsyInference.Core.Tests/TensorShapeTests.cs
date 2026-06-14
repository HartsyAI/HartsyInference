using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>Tests for <see cref="TensorShape"/>.</summary>
public sealed class TensorShapeTests
{
    [Fact]
    public void Constructor_1D_CorrectRankAndElement()
    {
        TensorShape shape = new TensorShape(10);

        Assert.Equal(1, shape.Rank);
        Assert.Equal(10, shape.ElementCount);
    }

    [Fact]
    public void Constructor_4D_NCHW()
    {
        TensorShape shape = new TensorShape(2, 3, 64, 64);

        Assert.Equal(4, shape.Rank);
        Assert.Equal(24576, shape.ElementCount);
    }

    [Fact]
    public void Indexer_ReturnsCorrectDimension()
    {
        TensorShape shape = new TensorShape(2, 3, 4);

        Assert.Equal(2, shape[0]);
        Assert.Equal(3, shape[1]);
        Assert.Equal(4, shape[2]);
    }

    [Fact]
    public void Indexer_OutOfRange_Throws()
    {
        TensorShape shape = new TensorShape(2, 3, 4);

        Assert.Throws<IndexOutOfRangeException>(() => shape[6]);
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
    public void Unsqueeze_AtEnd()
    {
        TensorShape shape = new TensorShape(3, 4);

        TensorShape unsqueezed = shape.Unsqueeze(2);

        Assert.Equal(3, unsqueezed.Rank);
        Assert.Equal(3, unsqueezed[0]);
        Assert.Equal(4, unsqueezed[1]);
        Assert.Equal(1, unsqueezed[2]);
    }

    [Fact]
    public void Unsqueeze_AtMaxRank_Throws()
    {
        TensorShape shape = new TensorShape(new ReadOnlySpan<long>([1, 2, 3, 4, 5, 6]));

        Assert.Throws<InvalidOperationException>(() => shape.Unsqueeze(0));
    }

    [Fact]
    public void Squeeze_RemovesUnitDim()
    {
        TensorShape shape = new TensorShape(1, 3, 4);

        TensorShape squeezed = shape.Squeeze(0);

        Assert.Equal(2, squeezed.Rank);
        Assert.Equal(3, squeezed[0]);
        Assert.Equal(4, squeezed[1]);
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
    public void Equality_SameShape_AreEqual()
    {
        TensorShape a = new TensorShape(2, 3, 4);
        TensorShape b = new TensorShape(2, 3, 4);

        Assert.Equal(a, b);
        Assert.True(a == b);
    }

    [Fact]
    public void Equality_DifferentShape_NotEqual()
    {
        TensorShape a = new TensorShape(2, 3, 4);
        TensorShape b = new TensorShape(2, 4, 3);

        Assert.NotEqual(a, b);
        Assert.True(a != b);
    }

    [Fact]
    public void CopyDimsTo_CopiesCorrectly()
    {
        TensorShape shape = new TensorShape(2, 3, 4);
        long[] dims = new long[3];

        shape.CopyDimsTo(dims.AsSpan());

        Assert.Equal(2, dims[0]);
        Assert.Equal(3, dims[1]);
        Assert.Equal(4, dims[2]);
    }

    [Fact]
    public void ToString_FormatsCorrectly()
    {
        TensorShape shape = new TensorShape(2, 3, 4);

        string result = shape.ToString();

        Assert.Equal("[2, 3, 4]", result);
    }
}

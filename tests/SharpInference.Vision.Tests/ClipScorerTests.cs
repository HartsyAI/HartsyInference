using SharpInference.Core.Tensors;
using SharpInference.Vision.Clip;
using SharpInference.Vision.Embeddings;
using Xunit;

namespace SharpInference.Vision.Tests;

/// <summary>Pure-math tests for <see cref="ClipScorer"/> using hand-built embeddings (no model).</summary>
public sealed class ClipScorerTests
{
    [Fact]
    public void Score_OrthogonalVectors_ReturnsZero()
    {
        using ImageEmbedding img = BuildUnit(dim: 4, oneAtIndex: 0);
        using TextEmbedding txt = BuildText(dim: 4, oneAtIndex: 1, text: "foo");
        float score = ClipScorer.Score(img, txt);
        Assert.InRange(score, -1e-6f, 1e-6f);
    }

    [Fact]
    public void Score_IdenticalUnitVectors_ReturnsOne()
    {
        using ImageEmbedding a = BuildUnit(dim: 8, oneAtIndex: 3);
        using TextEmbedding b = BuildText(dim: 8, oneAtIndex: 3, text: "x");
        float score = ClipScorer.Score(a, b);
        Assert.InRange(score, 1f - 1e-6f, 1f + 1e-6f);
    }

    [Fact]
    public void Score_OppositeVectors_ReturnsMinusOne()
    {
        using ImageEmbedding a = BuildUnit(dim: 4, oneAtIndex: 2);
        Tensor v = new(new TensorShape(1, 4), DType.F32);
        v.AsSpan<float>()[2] = -1f;
        using TextEmbedding b = new(v, "neg");
        float score = ClipScorer.Score(a, b);
        Assert.InRange(score, -1f - 1e-6f, -1f + 1e-6f);
    }

    [Fact]
    public void ScoreMatrix_ProducesExpectedShape()
    {
        ImageEmbedding[] imgs = [BuildUnit(4, 0), BuildUnit(4, 1)];
        TextEmbedding[] txts = [BuildText(4, 0, "a"), BuildText(4, 1, "b"), BuildText(4, 2, "c")];
        try
        {
            float[,] matrix = ClipScorer.ScoreMatrix(imgs, txts);
            Assert.Equal(2, matrix.GetLength(0));
            Assert.Equal(3, matrix.GetLength(1));

            Assert.InRange(matrix[0, 0], 1f - 1e-6f, 1f + 1e-6f);
            Assert.InRange(matrix[0, 1], -1e-6f, 1e-6f);
            Assert.InRange(matrix[1, 1], 1f - 1e-6f, 1f + 1e-6f);
            Assert.InRange(matrix[1, 0], -1e-6f, 1e-6f);
        }
        finally
        {
            foreach (ImageEmbedding e in imgs) e.Dispose();
            foreach (TextEmbedding e in txts) e.Dispose();
        }
    }

    [Fact]
    public void TopK_ReturnsBestMatchesDescending()
    {
        using ImageEmbedding query = BuildUnit(4, 0);
        TextEmbedding[] candidates = [
            BuildText(4, 1, "wrong"),
            BuildText(4, 0, "perfect"),
            BuildText(4, 2, "other"),
        ];
        try
        {
            IReadOnlyList<(int Index, float Score)> top = ClipScorer.TopK(query, candidates, k: 2);
            Assert.Equal(2, top.Count);
            Assert.Equal(1, top[0].Index);
            Assert.InRange(top[0].Score, 1f - 1e-6f, 1f + 1e-6f);
            Assert.True(top[0].Score >= top[1].Score);
        }
        finally
        {
            foreach (TextEmbedding e in candidates) e.Dispose();
        }
    }

    [Fact]
    public void Score_RejectsMismatchedDimensions()
    {
        using ImageEmbedding img = BuildUnit(4, 0);
        using TextEmbedding txt = BuildText(8, 0, "x");
        Assert.Throws<ArgumentException>(() => ClipScorer.Score(img, txt));
    }

    private static ImageEmbedding BuildUnit(int dim, int oneAtIndex)
    {
        Tensor v = new(new TensorShape(1, dim), DType.F32);
        Span<float> span = v.AsSpan<float>();
        span.Clear();
        span[oneAtIndex] = 1f;
        return new ImageEmbedding(v);
    }

    private static TextEmbedding BuildText(int dim, int oneAtIndex, string text)
    {
        Tensor v = new(new TensorShape(1, dim), DType.F32);
        Span<float> span = v.AsSpan<float>();
        span.Clear();
        span[oneAtIndex] = 1f;
        return new TextEmbedding(v, text);
    }
}

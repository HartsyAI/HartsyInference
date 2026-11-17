using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Layers;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Layers;

public sealed class HeadRmsNormTests
{
    private const int Heads = 4, HeadDim = 8, Tokens = 5;
    private static readonly IBackend Backend = new CpuBackend();

    [Fact]
    public void Apply_IsNormalizeTimesSqrtDimTimesGamma_OnAFlatGamma()
    {
        using HeadRmsNorm norm = new(HeadDim);
        Random rng = new(1);
        float[] x = Fill(rng, Tokens * Heads * HeadDim, 2f), gamma = Fill(rng, Heads * HeadDim, 1f);
        using Tensor heads = new(new TensorShape(1, Tokens, Heads, HeadDim), DType.F32);
        x.CopyTo(heads.AsSpan<float>());
        using Tensor g = new(new TensorShape(Heads * HeadDim), DType.F32);
        gamma.CopyTo(g.AsSpan<float>());

        using Tensor result = norm.Apply(Backend, heads, g, new TensorShape(1, Tokens, Heads * HeadDim));

        Assert.Equal(new TensorShape(1, Tokens, Heads * HeadDim), result.Shape);
        ReadOnlySpan<float> got = result.AsReadOnlySpan<float>();
        for (int t = 0; t < Tokens; t++)
            for (int h = 0; h < Heads; h++)
            {
                int row = (t * Heads + h) * HeadDim;
                double sq = 0;
                for (int d = 0; d < HeadDim; d++) sq += x[row + d] * (double)x[row + d];
                double inv = Math.Sqrt(HeadDim) / Math.Sqrt(sq);   // F.normalize(x) * sqrt(D)
                for (int d = 0; d < HeadDim; d++)
                    Assert.Equal(x[row + d] * inv * gamma[h * HeadDim + d], got[row + d], 4);
            }
    }

    [Fact]
    public void Apply_KeepsAnAllZeroHeadZero()
    {
        using HeadRmsNorm norm = new(HeadDim);
        using Tensor heads = new(new TensorShape(1, 1, 1, HeadDim), DType.F32);
        using Tensor gamma = new(new TensorShape(HeadDim), DType.F32);
        gamma.AsSpan<float>().Fill(3f);

        using Tensor result = norm.Apply(Backend, heads, gamma);

        foreach (float v in result.AsReadOnlySpan<float>()) Assert.Equal(0f, v);
    }

    [Fact]
    public void Apply_ConsumesItsInput()
    {
        using HeadRmsNorm norm = new(HeadDim);
        Tensor heads = new(new TensorShape(1, 2, 1, HeadDim), DType.F32);
        heads.AsSpan<float>().Fill(1f);
        using Tensor gamma = new(new TensorShape(HeadDim), DType.F32);
        gamma.AsSpan<float>().Fill(1f);

        norm.Apply(Backend, heads, gamma).Dispose();

        Assert.Throws<ObjectDisposedException>(() => heads.AsSpan<float>());
    }

    [Fact]
    public void Apply_AfterDispose_Throws()
    {
        HeadRmsNorm norm = new(HeadDim);
        norm.Dispose();
        using Tensor heads = new(new TensorShape(1, 1, 1, HeadDim), DType.F32);
        using Tensor gamma = new(new TensorShape(HeadDim), DType.F32);

        Assert.Throws<ObjectDisposedException>(() => norm.Apply(Backend, heads, gamma));
    }

    private static float[] Fill(Random rng, int n, float scale)
    {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return v;
    }
}

using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer.AttnRes;
using Xunit;

namespace HartsyInference.LLM.Tests.AttnRes;

/// <summary>The backend-composed block residual mix against the scalar reference, on the CPU backend.</summary>
public sealed class AttentionResidualTests
{
    private const float Tol = 1e-4f;

    [Theory]
    [InlineData(1, 8)]
    [InlineData(3, 8)]
    [InlineData(9, 16)]
    public void Mix_MatchesScalarReference(int n, int d)
    {
        Random rng = new(n * 31 + d);
        float[][] sources = Enumerable.Range(0, n).Select(_ => Floats(rng, d, -2f, 2f)).ToArray();
        float[] normWeight = Floats(rng, d, 0.5f, 1.5f);
        float[] query = Floats(rng, d, -3f, 3f);
        float[] expected = new float[d];
        AttentionResidual.MixReference(sources, normWeight, query, expected);

        using Tensor src = Make(sources.SelectMany(s => s).ToArray(), n, d);
        using Tensor norm = Make(normWeight, d);
        using Tensor q = Make(query, d, 1);
        using Tensor output = new(new TensorShape(1, d), DType.F32);
        IBackend backend = new CpuBackend();
        AttentionResidual.Mix(backend, src, norm, q, output);

        float[] actual = output.AsReadOnlySpan<float>().ToArray();
        for (int i = 0; i < d; i++) Assert.Equal(expected[i], actual[i], Tol);
    }

    private static float[] Floats(Random rng, int n, float lo, float hi)
    {
        float[] values = new float[n];
        for (int i = 0; i < n; i++) values[i] = lo + (float)rng.NextDouble() * (hi - lo);
        return values;
    }

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}

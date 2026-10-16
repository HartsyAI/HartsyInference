using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>The host references behind <see cref="IBackend.SituGlu"/> and <see cref="IBackend.RmsNormModulate"/>, the
/// defaults every backend falls back to and the oracle the GPU parity tests compare against.</summary>
public sealed class FusedNormActReferenceTests
{
    [Theory]
    [InlineData(0f, 0f)]
    [InlineData(1.5f, -2f)]
    [InlineData(-30f, 7f)]
    [InlineData(500f, 500f)]
    [InlineData(-500f, -500f)]
    [InlineData(80f, -100f)]
    public void SituScalar_MatchesTheFormula_AndStaysFiniteAtTheTails(float value, float gate)
    {
        double sigmoid = 1.0 / (1.0 + Math.Exp(-gate));
        double expected = 4.0 * Math.Tanh(gate / 4.0) * sigmoid * (25.0 * Math.Tanh(value / 25.0));
        float actual = SituGluReference.Scalar(value, gate);
        Assert.True(float.IsFinite(actual));
        Assert.Equal(expected, actual, 1e-5);
    }

    [Fact]
    public void SituGlu_ReadsTheValueHalfFirst()
    {
        using Tensor input = Make([2f, -1f, 3f, 0.5f], new TensorShape(1, 4));
        using Tensor output = new(new TensorShape(1, 2), DType.F32);
        SituGluReference.Apply(output, input);
        ReadOnlySpan<float> got = output.AsReadOnlySpan<float>();
        Assert.Equal(SituGluReference.Scalar(2f, 3f), got[0]);
        Assert.Equal(SituGluReference.Scalar(-1f, 0.5f), got[1]);
    }

    [Fact]
    public void SituGlu_RefusesAnOddPairing()
    {
        using Tensor input = new(new TensorShape(2, 6), DType.F32);
        using Tensor output = new(new TensorShape(2, 4), DType.F32);
        Assert.Throws<ArgumentException>(() => SituGluReference.Apply(output, input));
    }

    [Fact]
    public void RmsNormModulate_AppliesTheWeightAndEachBatchsModulation()
    {
        const int batches = 2, seq = 3, dim = 5;
        float[] x = Sequence(batches * seq * dim, 0.37f);
        float[] w = Sequence(dim, 0.11f);
        float[] scale = Sequence(batches * dim, -0.07f);
        float[] shift = Sequence(batches * dim, 0.05f);
        using Tensor input = Make(x, new TensorShape(batches, seq, dim));
        using Tensor weight = Make(w, new TensorShape(dim));
        using Tensor scaleT = Make(scale, new TensorShape(batches, dim));
        using Tensor shiftT = Make(shift, new TensorShape(batches, dim));
        using Tensor output = new(input.Shape, DType.F32);
        RmsNormModulateReference.Apply(output, input, weight, scaleT, shiftT, 1e-6f);

        ReadOnlySpan<float> got = output.AsReadOnlySpan<float>();
        for (int row = 0; row < batches * seq; row++)
        {
            double sumSq = 0;
            for (int d = 0; d < dim; d++) sumSq += (double)x[row * dim + d] * x[row * dim + d];
            double inv = 1.0 / Math.Sqrt(sumSq / dim + 1e-6);
            int b = row / seq;
            for (int d = 0; d < dim; d++)
            {
                double want = x[row * dim + d] * inv * w[d] * (1 + scale[b * dim + d]) + shift[b * dim + d];
                Assert.Equal(want, got[row * dim + d], 1e-5);
            }
        }
    }

    [Fact]
    public void RmsNormModulate_WritesF16FromAnF32Input()
    {
        using Tensor input = Make(Sequence(8, 0.5f), new TensorShape(2, 4));
        using Tensor weight = Make([1f, 1f, 1f, 1f], new TensorShape(4));
        using Tensor scale = Make([0f, 0f, 0f, 0f], new TensorShape(4));
        using Tensor wide = new(input.Shape, DType.F32);
        using Tensor narrow = new(input.Shape, DType.F16);
        RmsNormModulateReference.Apply(wide, input, weight, scale, null, 1e-6f);
        RmsNormModulateReference.Apply(narrow, input, weight, scale, null, 1e-6f);
        using Tensor widened = narrow.CastTo(DType.F32);
        ReadOnlySpan<float> a = wide.AsReadOnlySpan<float>();
        ReadOnlySpan<float> b = widened.AsReadOnlySpan<float>();
        for (int i = 0; i < a.Length; i++) Assert.Equal(a[i], b[i], 2e-3);
    }

    [Fact]
    public void RmsNormModulate_RefusesAModulationThatDoesNotTileTheRows()
    {
        using Tensor input = new(new TensorShape(1, 3, 4), DType.F32);
        using Tensor weight = new(new TensorShape(4), DType.F32);
        using Tensor scale = new(new TensorShape(2, 4), DType.F32);
        using Tensor output = new(input.Shape, DType.F32);
        Assert.Throws<ArgumentException>(() => RmsNormModulateReference.Apply(output, input, weight, scale, null, 1e-6f));
    }

    private static Tensor Make(float[] values, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static float[] Sequence(int count, float step)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = MathF.Sin(i * step + 0.3f) * 2f;
        return values;
    }
}

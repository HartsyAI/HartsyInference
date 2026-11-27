using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Guidance;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Scaling each row to another tensor's row norm is the per-row guided step at guidance 0 and delta 1 over a zero output, so it needs no op of
/// its own (FastFrames' per-head renorm).</summary>
public sealed class RowNormMatchTests
{
    [Theory]
    [InlineData(1, 8)]
    [InlineData(12, 128)]
    public void TheRowNormStepOverZeros_IsTheRowNormMatch_BitForBit(int rows, int width)
    {
        const float Eps = 1e-6f;
        Random rng = new(4);
        using Tensor x = Filled(rng, rows, width), reference = Filled(rng, rows, width), output = new(x.Shape, DType.F32);
        x.AsSpan<float>()[..width].Clear(); // a zero row: the eps is all the denominator
        using CpuBackend backend = new();

        backend.Fill(output, 0f);
        ((IBackend)backend).GuidedEulerStep(output, reference, x, GuidedEulerSpec.RowNorm(guidance: 0f, delta: 1f, eps: Eps));

        ReadOnlySpan<float> xs = x.AsReadOnlySpan<float>(), rs = reference.AsReadOnlySpan<float>(), actual = output.AsReadOnlySpan<float>();
        for (int row = 0; row < rows; row++)
        {
            double xNorm = 0, rNorm = 0;
            for (int d = row * width; d < (row + 1) * width; d++)
            {
                xNorm += (double)xs[d] * xs[d];
                rNorm += (double)rs[d] * rs[d];
            }
            float scale = (float)(Math.Sqrt(rNorm) / (Math.Sqrt(xNorm) + Eps));
            for (int d = row * width; d < (row + 1) * width; d++) Assert.Equal(xs[d] * scale, actual[d]);
        }
    }

    private static Tensor Filled(Random rng, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = (float)(rng.NextDouble() * 4 - 2);
        return t;
    }
}

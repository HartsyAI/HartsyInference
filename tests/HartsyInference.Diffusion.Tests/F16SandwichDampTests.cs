using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The RMSNorm that consumes an F16-damped projection must reproduce the undamped norm. Rows are small
/// (RMS ~0.05, the scale of an Ideogram 4 sublayer output), where keeping the plain eps visibly shrinks the result.</summary>
public sealed unsafe class F16SandwichDampTests
{
    private const int Dim = 64;
    private const float Eps = 1e-5f;

    [Fact]
    public void DampedNorm_WithMatchedEps_EqualsUndampedNorm()
    {
        (float[] reference, float[] damped) = Norms(F16SandwichDamp.NormEps(Eps, damped: true));
        for (int i = 0; i < Dim; i++)
            Assert.Equal(reference[i], damped[i], 1e-5f * MathF.Max(1f, MathF.Abs(reference[i])));
    }

    [Fact]
    public void DampedNorm_WithPlainEps_ShrinksSmallRows()
    {
        (float[] reference, float[] damped) = Norms(Eps);
        double ratio = Rms(damped) / Rms(reference);
        Assert.True(ratio < 0.8, $"plain eps kept {ratio:P0} of the norm; the case no longer exercises the eps term");
    }

    private static (float[] Reference, float[] Damped) Norms(float dampedEps)
    {
        IBackend cpu = new CpuBackend();
        using Tensor x = new(new TensorShape(1, 1, Dim), DType.F32);
        using Tensor xDamped = new(new TensorShape(1, 1, Dim), DType.F32);
        using Tensor weight = new(new TensorShape(Dim), DType.F32);
        Random rng = new(7);
        for (int i = 0; i < Dim; i++)
        {
            float v = (float)(rng.NextDouble() * 2 - 1) * 0.08f;
            ((float*)x.DataPointer)[i] = v;
            ((float*)xDamped.DataPointer)[i] = v * F16SandwichDamp.Factor;
            ((float*)weight.DataPointer)[i] = 1f + 0.01f * i;
        }
        using Tensor reference = new(x.Shape, DType.F32);
        using Tensor damped = new(x.Shape, DType.F32);
        cpu.RmsNorm(reference, x, weight, Eps);
        cpu.RmsNorm(damped, xDamped, weight, dampedEps);
        return (new ReadOnlySpan<float>((float*)reference.DataPointer, Dim).ToArray(),
            new ReadOnlySpan<float>((float*)damped.DataPointer, Dim).ToArray());
    }

    private static double Rms(float[] values) => Math.Sqrt(values.Average(v => (double)v * v));
}

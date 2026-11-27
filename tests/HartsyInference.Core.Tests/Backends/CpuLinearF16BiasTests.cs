using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>The CPU Linear takes an F16 bias the way it takes an F16 weight: widened to F32 for the call. The Fibo 2.0 denoiser
/// stores its biases as F16 (the half-precision cast), so a CPU run reaches this path with an F16 bias on every block.</summary>
public sealed class CpuLinearF16BiasTests
{
    [Fact]
    public void Linear_WithF16Bias_MatchesTheSameValuesAsF32()
    {
        // Bias values are multiples of 1/4, exact in F16, so the widened and the F32 runs must agree bit for bit.
        float[] bias = [0.25f, -1.5f, 2.75f];
        using Tensor input = Fill(new TensorShape(2, 4), 1);
        using Tensor weight = Fill(new TensorShape(3, 4), 2);
        using Tensor biasF32 = new(new TensorShape(3), DType.F32);
        using Tensor biasF16 = new(new TensorShape(3), DType.F16);
        for (int i = 0; i < 3; i++)
        {
            biasF32.AsSpan<float>()[i] = bias[i];
            biasF16.AsSpan<ushort>()[i] = BitConverter.HalfToUInt16Bits((Half)bias[i]);
        }

        using IBackend backend = new CpuBackend();
        using Tensor outF32 = new(new TensorShape(2, 3), DType.F32);
        using Tensor outF16 = new(new TensorShape(2, 3), DType.F32);
        backend.Linear(outF32, input, weight, biasF32);
        backend.Linear(outF16, input, weight, biasF16); // threw "the CPU backend is F32-only" before the bias was widened

        Assert.Equal(outF32.AsReadOnlySpan<float>().ToArray(), outF16.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void Linear_WithF16Bias_WidensOnlyTheBiasAndLeavesTheCallerBiasUntouched()
    {
        float[] bias = [0.5f, -0.25f, 1.0f];
        using Tensor input = Fill(new TensorShape(1, 4), 3);
        using Tensor weight = Fill(new TensorShape(3, 4), 4);
        using Tensor biasF16 = new(new TensorShape(3), DType.F16);
        ushort[] before = new ushort[3];
        for (int i = 0; i < 3; i++)
        {
            before[i] = BitConverter.HalfToUInt16Bits((Half)bias[i]);
            biasF16.AsSpan<ushort>()[i] = before[i];
        }

        using IBackend backend = new CpuBackend();
        using Tensor output = new(new TensorShape(1, 3), DType.F32);
        backend.Linear(output, input, weight, biasF16);

        Assert.Equal(DType.F16, biasF16.DType);
        Assert.Equal(before, biasF16.AsReadOnlySpan<ushort>().ToArray());
    }

    private static Tensor Fill(TensorShape shape, int seed)
    {
        Tensor t = new(shape, DType.F32);
        Random random = new(seed);
        foreach (ref float value in t.AsSpan<float>())
            value = (float)(random.NextDouble() * 2 - 1);
        return t;
    }
}

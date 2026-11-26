using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>The CPU Linear takes a BF16 or F16 bias the way it takes a BF16 or F16 weight: widened to F32 for the call. A checkpoint's
/// bias is stored in its own dtype, so a CPU run of a BF16 checkpoint reaches this with a BF16 bias.</summary>
public sealed class CpuLinearBf16BiasTests
{
    [Fact]
    public void Linear_WithBf16Bias_MatchesTheSameValuesAsF32()
    {
        // Bias values are multiples of 1/4, exact in BF16, so the widened and the F32 runs must agree bit for bit.
        float[] bias = [0.25f, -1.5f, 2.75f];
        using Tensor input = Fill(new TensorShape(2, 4), 1);
        using Tensor weight = Fill(new TensorShape(3, 4), 2);
        using Tensor biasF32 = new(new TensorShape(3), DType.F32);
        using Tensor biasBf16 = new(new TensorShape(3), DType.BF16);
        for (int i = 0; i < 3; i++)
        {
            biasF32.AsSpan<float>()[i] = bias[i];
            biasBf16.AsSpan<ushort>()[i] = TensorCasts.F32ToBf16Bits(bias[i]);
        }

        using IBackend backend = new CpuBackend();
        using Tensor outF32 = new(new TensorShape(2, 3), DType.F32);
        using Tensor outBf16 = new(new TensorShape(2, 3), DType.F32);
        backend.Linear(outF32, input, weight, biasF32);
        backend.Linear(outBf16, input, weight, biasBf16); // threw "the CPU backend is F32-only" before the bias was widened

        Assert.Equal(outF32.AsReadOnlySpan<float>().ToArray(), outBf16.AsReadOnlySpan<float>().ToArray());
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

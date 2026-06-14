using HartsyInference.Cpu.Kernels;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Cpu.Tests;

public sealed unsafe class NormKernelTests
{
    private const float Tolerance = 1e-5f;

    [Fact]
    public void GroupNorm_UniformInput_ReturnsWeightedBias()
    {
        // input [1,4,2,2] all 5.0, weight=1.0, bias=0.0, groups=2
        using Tensor input = new Tensor(new TensorShape(1, 4, 2, 2), DType.F32);
        using Tensor weight = new Tensor(new TensorShape(4), DType.F32);
        using Tensor bias = new Tensor(new TensorShape(4), DType.F32);
        using Tensor output = new Tensor(new TensorShape(1, 4, 2, 2), DType.F32);

        Span<float> inSpan = input.AsSpan<float>();
        for (int i = 0; i < 16; i++)
        {
            inSpan[i] = 5.0f;
        }

        Span<float> wSpan = weight.AsSpan<float>();
        for (int i = 0; i < 4; i++)
        {
            wSpan[i] = 1.0f;
        }
        // bias is already zeroed

        NormKernels.GroupNorm(output, input, weight, bias, 2, 1e-5f);

        Span<float> outSpan = output.AsSpan<float>();
        // Uniform input -> variance=0, normalized = (5-5)/sqrt(0+eps) * 1 + 0 ~ 0
        for (int i = 0; i < 16; i++)
        {
            Assert.Equal(0.0f, outSpan[i], Tolerance);
        }
    }

    [Fact]
    public void GroupNorm_KnownValues()
    {
        // input [1,2,1,4] with values [1,2,3,4, 5,6,7,8]
        using Tensor input = new Tensor(new TensorShape(1, 2, 1, 4), DType.F32);
        using Tensor weight = new Tensor(new TensorShape(2), DType.F32);
        using Tensor bias = new Tensor(new TensorShape(2), DType.F32);
        using Tensor output = new Tensor(new TensorShape(1, 2, 1, 4), DType.F32);

        Span<float> inSpan = input.AsSpan<float>();
        inSpan[0] = 1.0f; inSpan[1] = 2.0f; inSpan[2] = 3.0f; inSpan[3] = 4.0f;
        inSpan[4] = 5.0f; inSpan[5] = 6.0f; inSpan[6] = 7.0f; inSpan[7] = 8.0f;

        Span<float> wSpan = weight.AsSpan<float>();
        wSpan[0] = 1.0f; wSpan[1] = 1.0f;
        // bias is already zeroed

        float eps = 1e-5f;
        NormKernels.GroupNorm(output, input, weight, bias, 2, eps);

        Span<float> outSpan = output.AsSpan<float>();

        // Group 0 (channel 0): values [1,2,3,4], mean=2.5, var=1.25
        float mean0 = 2.5f;
        float invStd0 = 1.0f / MathF.Sqrt(1.25f + eps);
        Assert.Equal((1.0f - mean0) * invStd0, outSpan[0], Tolerance);
        Assert.Equal((2.0f - mean0) * invStd0, outSpan[1], Tolerance);
        Assert.Equal((3.0f - mean0) * invStd0, outSpan[2], Tolerance);
        Assert.Equal((4.0f - mean0) * invStd0, outSpan[3], Tolerance);

        // Group 1 (channel 1): values [5,6,7,8], mean=6.5, var=1.25
        float mean1 = 6.5f;
        float invStd1 = 1.0f / MathF.Sqrt(1.25f + eps);
        Assert.Equal((5.0f - mean1) * invStd1, outSpan[4], Tolerance);
        Assert.Equal((6.0f - mean1) * invStd1, outSpan[5], Tolerance);
        Assert.Equal((7.0f - mean1) * invStd1, outSpan[6], Tolerance);
        Assert.Equal((8.0f - mean1) * invStd1, outSpan[7], Tolerance);
    }

    [Fact]
    public void LayerNorm_UniformInput_ReturnsZeroPlusBias()
    {
        // input [2,4] all 3.0, weight=1, bias=0.5
        using Tensor input = new Tensor(new TensorShape(2, 4), DType.F32);
        using Tensor weight = new Tensor(new TensorShape(4), DType.F32);
        using Tensor bias = new Tensor(new TensorShape(4), DType.F32);
        using Tensor output = new Tensor(new TensorShape(2, 4), DType.F32);

        Span<float> inSpan = input.AsSpan<float>();
        for (int i = 0; i < 8; i++)
        {
            inSpan[i] = 3.0f;
        }

        Span<float> wSpan = weight.AsSpan<float>();
        Span<float> bSpan = bias.AsSpan<float>();
        for (int i = 0; i < 4; i++)
        {
            wSpan[i] = 1.0f;
            bSpan[i] = 0.5f;
        }

        NormKernels.LayerNorm(output, input, weight, bias, 1e-5f);

        Span<float> outSpan = output.AsSpan<float>();
        // Uniform input -> normalized = 0, output = 0*1 + 0.5 = 0.5
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(0.5f, outSpan[i], Tolerance);
        }
    }

    [Fact]
    public void LayerNorm_KnownValues()
    {
        // input [1,4] = [1,2,3,4], weight=[1,1,1,1], bias=[0,0,0,0]
        using Tensor input = new Tensor(new TensorShape(1, 4), DType.F32);
        using Tensor weight = new Tensor(new TensorShape(4), DType.F32);
        using Tensor bias = new Tensor(new TensorShape(4), DType.F32);
        using Tensor output = new Tensor(new TensorShape(1, 4), DType.F32);

        Span<float> inSpan = input.AsSpan<float>();
        inSpan[0] = 1.0f; inSpan[1] = 2.0f; inSpan[2] = 3.0f; inSpan[3] = 4.0f;

        Span<float> wSpan = weight.AsSpan<float>();
        for (int i = 0; i < 4; i++)
        {
            wSpan[i] = 1.0f;
        }
        // bias is already zeroed

        float eps = 1e-5f;
        NormKernels.LayerNorm(output, input, weight, bias, eps);

        Span<float> outSpan = output.AsSpan<float>();

        // mean=2.5, var=1.25
        float mean = 2.5f;
        float invStd = 1.0f / MathF.Sqrt(1.25f + eps);
        Assert.Equal((1.0f - mean) * invStd, outSpan[0], Tolerance);
        Assert.Equal((2.0f - mean) * invStd, outSpan[1], Tolerance);
        Assert.Equal((3.0f - mean) * invStd, outSpan[2], Tolerance);
        Assert.Equal((4.0f - mean) * invStd, outSpan[3], Tolerance);
    }

    [Fact]
    public void RmsNorm_KnownValues()
    {
        // input [1,4] = [1,2,3,4], weight=[1,1,1,1]
        using Tensor input = new Tensor(new TensorShape(1, 4), DType.F32);
        using Tensor weight = new Tensor(new TensorShape(4), DType.F32);
        using Tensor output = new Tensor(new TensorShape(1, 4), DType.F32);

        Span<float> inSpan = input.AsSpan<float>();
        inSpan[0] = 1.0f; inSpan[1] = 2.0f; inSpan[2] = 3.0f; inSpan[3] = 4.0f;

        Span<float> wSpan = weight.AsSpan<float>();
        for (int i = 0; i < 4; i++)
        {
            wSpan[i] = 1.0f;
        }

        float eps = 1e-5f;
        NormKernels.RmsNorm(output, input, weight, eps);

        Span<float> outSpan = output.AsSpan<float>();

        // rms = sqrt(mean([1,4,9,16]) + eps) = sqrt(7.5 + eps)
        float rms = MathF.Sqrt(7.5f + eps);
        float invRms = 1.0f / rms;
        Assert.Equal(1.0f * invRms, outSpan[0], Tolerance);
        Assert.Equal(2.0f * invRms, outSpan[1], Tolerance);
        Assert.Equal(3.0f * invRms, outSpan[2], Tolerance);
        Assert.Equal(4.0f * invRms, outSpan[3], Tolerance);
    }
}

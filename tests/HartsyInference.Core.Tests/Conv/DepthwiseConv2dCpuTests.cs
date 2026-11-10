using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.Core.Tests.Conv;

/// <summary>Depthwise Conv2D argument validation, dtype admission and the host BF16 output path; runs without a GPU.</summary>
public sealed class DepthwiseConv2dCpuTests
{
    private const int Batch = 2, Channels = 3, Height = 7, Width = 6, Kernel = 3;

    [Theory]
    [InlineData(1, 1, true)]
    [InlineData(2, 1, false)]
    [InlineData(1, 0, true)]
    public void Conv2dDepthwise_Bf16Output_MatchesF32PathOnBf16Operands(int stride, int pad, bool bf16Operands)
    {
        int outH = (Height + 2 * pad - Kernel) / stride + 1;
        int outW = (Width + 2 * pad - Kernel) / stride + 1;
        TensorShape inShape = new(Batch, Channels, Height, Width);
        TensorShape weightShape = new(Channels, 1, Kernel, Kernel);
        TensorShape biasShape = new(Channels);
        TensorShape outShape = new(Batch, Channels, outH, outW);
        DType operandDtype = bf16Operands ? DType.BF16 : DType.F32;

        using Tensor input = new(inShape, operandDtype);
        using Tensor weight = new(weightShape, operandDtype);
        using Tensor bias = new(biasShape, operandDtype);
        using Tensor refInput = new(inShape, DType.F32);
        using Tensor refWeight = new(weightShape, DType.F32);
        using Tensor refBias = new(biasShape, DType.F32);
        using Tensor refOutput = new(outShape, DType.F32);
        using Tensor output = new(outShape, DType.BF16);

        Random random = new(11);
        Fill(input, refInput, random, 1f);
        Fill(weight, refWeight, random, 0.5f);
        Fill(bias, refBias, random, 0.5f);

        IBackend backend = new CpuBackend();
        backend.Conv2dDepthwise(refOutput, refInput, refWeight, refBias, stride, stride, pad, pad);
        backend.Conv2dDepthwise(output, input, weight, bias, stride, stride, pad, pad);

        // The BF16 path rounds operands and accumulates in the same F32 tap order, so only the final rounding remains.
        ReadOnlySpan<float> expected = refOutput.AsSpan<float>();
        ReadOnlySpan<ushort> actual = output.AsSpan<ushort>();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(BFloat16Bits.FromFloat(expected[i]), actual[i]);
    }

    [Fact]
    public void Conv2dDepthwise_F32Output_RejectsBf16Operand()
    {
        IBackend backend = new CpuBackend();
        using Tensor input = new(new TensorShape(Batch, Channels, Height, Width), DType.BF16);
        using Tensor weight = new(new TensorShape(Channels, 1, Kernel, Kernel), DType.F32);
        using Tensor output = new(new TensorShape(Batch, Channels, Height, Width), DType.F32);

        Assert.Throws<NotSupportedException>(() => backend.Conv2dDepthwise(output, input, weight, null, 1, 1, 1, 1));
    }

    [Fact]
    public void Conv2dDepthwise_F16Output_IsRejectedOnHost()
    {
        IBackend backend = new CpuBackend();
        using Tensor input = new(new TensorShape(Batch, Channels, Height, Width), DType.F32);
        using Tensor weight = new(new TensorShape(Channels, 1, Kernel, Kernel), DType.F32);
        using Tensor output = new(new TensorShape(Batch, Channels, Height, Width), DType.F16);

        Assert.Throws<NotSupportedException>(() => backend.Conv2dDepthwise(output, input, weight, null, 1, 1, 1, 1));
    }

    [Fact]
    public void Conv2dDepthwise_Bf16Output_RejectsF16Operand()
    {
        IBackend backend = new CpuBackend();
        using Tensor input = new(new TensorShape(Batch, Channels, Height, Width), DType.F16);
        using Tensor weight = new(new TensorShape(Channels, 1, Kernel, Kernel), DType.BF16);
        using Tensor output = new(new TensorShape(Batch, Channels, Height, Width), DType.BF16);

        Assert.Throws<NotSupportedException>(() => backend.Conv2dDepthwise(output, input, weight, null, 1, 1, 1, 1));
    }

    [Fact]
    public void Conv2dDepthwise_Bf16Output_RejectsRank3Input()
    {
        ExpectArgumentError(new TensorShape(Batch, Channels, Height), new TensorShape(Channels, 1, Kernel, Kernel), new TensorShape(Batch, Channels, Height, Width), -1);
    }

    [Fact]
    public void Conv2dDepthwise_Bf16Output_RejectsWeightWithoutUnitInputChannel()
    {
        ExpectArgumentError(new TensorShape(Batch, Channels, Height, Width), new TensorShape(Channels, 2, Kernel, Kernel), new TensorShape(Batch, Channels, Height, Width), -1);
    }

    [Fact]
    public void Conv2dDepthwise_Bf16Output_RejectsOutputChannelMismatch()
    {
        ExpectArgumentError(new TensorShape(Batch, Channels, Height, Width), new TensorShape(Channels, 1, Kernel, Kernel), new TensorShape(Batch, Channels + 1, Height, Width), -1);
    }

    [Fact]
    public void Conv2dDepthwise_Bf16Output_RejectsBiasOfWrongLength()
    {
        ExpectArgumentError(new TensorShape(Batch, Channels, Height, Width), new TensorShape(Channels, 1, Kernel, Kernel), new TensorShape(Batch, Channels, Height, Width), Channels + 1);
    }

    [Fact]
    public void DepthwiseConv2dContract_ReturnsGeometryForValidShapes()
    {
        using Tensor input = new(new TensorShape(Batch, Channels, Height, Width), DType.BF16);
        using Tensor weight = new(new TensorShape(Channels, 1, Kernel, Kernel), DType.BF16);
        using Tensor bias = new(new TensorShape(Channels), DType.BF16);
        using Tensor output = new(new TensorShape(Batch, Channels, 5, 4), DType.BF16);

        DepthwiseConv2dGeometry geometry = DepthwiseConv2dContract.Validate(input, weight, bias, output);

        Assert.Equal(new DepthwiseConv2dGeometry(Batch, Channels, Height, Width, 5, 4, Kernel, Kernel), geometry);
    }

    [Fact]
    public void DepthwiseConv2dPlan_AdmitsOnlyF32F16AndBf16Outputs()
    {
        Assert.True(DepthwiseConv2dPlan.IsSupportedOutput(DType.F32));
        Assert.True(DepthwiseConv2dPlan.IsSupportedOutput(DType.F16));
        Assert.True(DepthwiseConv2dPlan.IsSupportedOutput(DType.BF16));
        Assert.False(DepthwiseConv2dPlan.IsSupportedOutput(DType.F8E4M3));
        Assert.False(DepthwiseConv2dPlan.IsSupportedOutput(DType.Q8_0));
    }

    [Fact]
    public void DepthwiseConv2dPlan_RejectsUnsupportedOutputDtype()
    {
        using Tensor input = new(new TensorShape(Batch, Channels, Height, Width), DType.F32);
        using Tensor weight = new(new TensorShape(Channels, 1, Kernel, Kernel), DType.F32);
        using Tensor output = new(new TensorShape(Batch, Channels, Height, Width), DType.F8E4M3);

        Assert.Throws<NotSupportedException>(() => DepthwiseConv2dPlan.Validate(input, weight, null, output));
    }

    /// <summary>Runs a BF16-output call whose shapes must fail the geometry contract; a negative <paramref name="biasCount"/> means no bias.</summary>
    private static void ExpectArgumentError(TensorShape inputShape, TensorShape weightShape, TensorShape outputShape, int biasCount)
    {
        IBackend backend = new CpuBackend();
        using Tensor input = new(inputShape, DType.F32);
        using Tensor weight = new(weightShape, DType.F32);
        using Tensor output = new(outputShape, DType.BF16);
        using Tensor? bias = biasCount < 0 ? null : new Tensor(new TensorShape(biasCount), DType.F32);

        Assert.Throws<ArgumentException>(() => backend.Conv2dDepthwise(output, input, weight, bias, 1, 1, 1, 1));
    }

    /// <summary>Fills an F32 or BF16 operand and the F32 reference with the same values, the reference holding them as BF16 rounds them.</summary>
    private static void Fill(Tensor operand, Tensor reference, Random random, float scale)
    {
        Span<float> referenceValues = reference.AsSpan<float>();
        Span<ushort> bf16Values = operand.DType == DType.BF16 ? operand.AsSpan<ushort>() : Span<ushort>.Empty;
        Span<float> f32Values = operand.DType == DType.BF16 ? Span<float>.Empty : operand.AsSpan<float>();
        for (int i = 0; i < referenceValues.Length; i++)
        {
            float value = ((float)random.NextDouble() * 2f - 1f) * scale;
            ushort bits = BFloat16Bits.FromFloat(value);
            referenceValues[i] = BFloat16Bits.ToFloat(bits);
            if (operand.DType == DType.BF16) bf16Values[i] = bits;
            else f32Values[i] = value;
        }
    }
}

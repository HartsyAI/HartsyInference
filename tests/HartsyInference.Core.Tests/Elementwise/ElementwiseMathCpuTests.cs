using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Elementwise;

/// <summary>The host <see cref="IBackend.Exp"/> and <see cref="IBackend.Reciprocal"/>: F32 and BF16 on either side, IEEE edge values
/// and the contract's refusals; runs without a GPU.</summary>
public sealed class ElementwiseMathCpuTests
{
    private static readonly float[] Values = [-3.5f, -1f, -0.125f, 0f, 1e-3f, 0.5f, 1f, 2.75f, 9f, -88f];

    [Fact]
    public void Exp_F32_MatchesMathFExp()
    {
        IBackend backend = new CpuBackend();
        using Tensor input = F32(Values);
        using Tensor output = new(input.Shape, DType.F32);

        backend.Exp(output, input);

        ReadOnlySpan<float> actual = output.AsSpan<float>();
        for (int i = 0; i < Values.Length; i++)
            Assert.Equal(MathF.Exp(Values[i]), actual[i]);
    }

    [Fact]
    public void Reciprocal_F32_IsIeeeDivision_WithSignedInfinityAtZero()
    {
        IBackend backend = new CpuBackend();
        using Tensor input = F32([2f, -4f, 3f, 0f, -0f, 1e-30f]);
        using Tensor output = new(input.Shape, DType.F32);

        backend.Reciprocal(output, input);

        ReadOnlySpan<float> actual = output.AsSpan<float>();
        Assert.Equal(0.5f, actual[0]);
        Assert.Equal(-0.25f, actual[1]);
        Assert.Equal(1f / 3f, actual[2]);
        Assert.Equal(float.PositiveInfinity, actual[3]);
        Assert.Equal(float.NegativeInfinity, actual[4]);
        Assert.Equal(1f / 1e-30f, actual[5]);
    }

    [Theory]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void Exp_Bf16Sides_RoundOnceFromTheF32Result(bool bf16Input, bool bf16Output)
    {
        IBackend backend = new CpuBackend();
        float[] operands = bf16Input ? RoundedToBf16(Values) : Values;
        using Tensor input = bf16Input ? Bf16(Values) : F32(Values);
        using Tensor output = new(input.Shape, bf16Output ? DType.BF16 : DType.F32);

        backend.Exp(output, input);

        for (int i = 0; i < operands.Length; i++)
        {
            float expected = MathF.Exp(operands[i]);
            if (bf16Output)
                Assert.Equal(BFloat16Bits.FromFloat(expected), output.AsSpan<ushort>()[i]);
            else
                Assert.Equal(expected, output.AsSpan<float>()[i]);
        }
    }

    [Fact]
    public void Reciprocal_Bf16Output_RoundsToNearestEven()
    {
        IBackend backend = new CpuBackend();
        using Tensor input = F32([3f, 7f, -9f]);
        using Tensor output = new(input.Shape, DType.BF16);

        backend.Reciprocal(output, input);

        ReadOnlySpan<ushort> actual = output.AsSpan<ushort>();
        Assert.Equal(BFloat16Bits.FromFloat(1f / 3f), actual[0]);
        Assert.Equal(BFloat16Bits.FromFloat(1f / 7f), actual[1]);
        Assert.Equal(BFloat16Bits.FromFloat(-1f / 9f), actual[2]);
    }

    [Fact]
    public void Exp_ShapesMayDiffer_WhenElementCountsAgree()
    {
        IBackend backend = new CpuBackend();
        using Tensor input = new(new TensorShape(2, 3), DType.F32);
        input.AsSpan<float>().Fill(1f);
        using Tensor output = new(new TensorShape(6, 1), DType.F32);

        backend.Exp(output, input);

        foreach (float value in output.AsSpan<float>())
            Assert.Equal(MathF.E, value);
    }

    [Fact]
    public void Contract_RefusesOtherDtypesCountMismatchAndAliasing()
    {
        IBackend backend = new CpuBackend();
        using Tensor f32 = F32([1f, 2f]);
        using Tensor f16 = new(new TensorShape(2), DType.F16);
        using Tensor longer = new(new TensorShape(3), DType.F32);

        Assert.Throws<NotSupportedException>(() => backend.Exp(f16, f32));
        Assert.Throws<NotSupportedException>(() => backend.Reciprocal(f32, f16));
        Assert.Throws<ArgumentException>(() => backend.Exp(longer, f32));
        Assert.Throws<ArgumentException>(() => backend.Reciprocal(f32, f32));
    }

    private static Tensor F32(float[] values)
    {
        Tensor tensor = new(new TensorShape(values.Length), DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static Tensor Bf16(float[] values)
    {
        Tensor tensor = new(new TensorShape(values.Length), DType.BF16);
        Span<ushort> bits = tensor.AsSpan<ushort>();
        for (int i = 0; i < values.Length; i++)
            bits[i] = BFloat16Bits.FromFloat(values[i]);
        return tensor;
    }

    private static float[] RoundedToBf16(float[] values)
    {
        float[] rounded = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
            rounded[i] = BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(values[i]));
        return rounded;
    }
}

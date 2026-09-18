using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>Round-trips <see cref="Int8ConvRotCodec.QuantizeFromF32"/> against its dequant twin. The rotation is the
/// part that fails silently: H is symmetric and orthogonal, so applying it the wrong number of times still produces
/// plausible-looking weights, and the only signal is an image or clip that is subtly wrong.</summary>
public sealed class Int8ConvRotQuantizeTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(16)]
    [InlineData(64)]
    public void QuantizeThenDequantize_ReturnsTheOriginalWithinTheStepSize(int groupSize)
    {
        const int rows = 6, columns = 64;
        using Tensor original = Ramp(rows, columns);
        using Tensor source = original.CastTo(DType.F32);

        (Tensor packed, Tensor rowScale) = Int8ConvRotCodec.QuantizeFromF32(source, groupSize);
        try
        {
            using Tensor restored = Int8ConvRotCodec.DequantToBf16(packed, rowScale, groupSize);
            using Tensor restoredF32 = restored.CastTo(DType.F32);

            ReadOnlySpan<float> before = original.AsReadOnlySpan<float>();
            ReadOnlySpan<float> after = restoredF32.AsReadOnlySpan<float>();
            ReadOnlySpan<float> scales = rowScale.AsReadOnlySpan<float>();
            for (int row = 0; row < rows; row++)
            {
                // One int8 step per element after the rotation, which spreads a group's error across it, plus BF16's
                // 8-bit mantissa on the way back out.
                float budget = (scales[row] * groupSize switch { 0 => 1f, _ => MathF.Sqrt(groupSize) }) + 0.02f;
                for (int column = 0; column < columns; column++)
                {
                    int i = (row * columns) + column;
                    Assert.True(MathF.Abs(before[i] - after[i]) <= budget,
                        $"[{row},{column}]: {before[i]} became {after[i]} (budget {budget}).");
                }
            }
        }
        finally
        {
            rowScale.Dispose();
            packed.Dispose();
        }
    }

    [Fact]
    public void QuantizeFromF32_RecomputesThePerRowScaleFromTheValuesItIsGiven()
    {
        // The scale must come from THESE values, not from whatever the weight carried before: a merged LoRA moves
        // each row's absmax, and reusing the old scale clips everything it pushed past it.
        using Tensor source = new Tensor(new TensorShape(2, 32), DType.F32);
        Span<float> values = source.AsSpan<float>();
        values[..32].Fill(0.5f);
        values[32..].Fill(8.0f);

        (Tensor packed, Tensor rowScale) = Int8ConvRotCodec.QuantizeFromF32(source, 0);
        try
        {
            ReadOnlySpan<float> scales = rowScale.AsReadOnlySpan<float>();
            Assert.Equal(0.5f / 127f, scales[0], 6);
            Assert.Equal(8.0f / 127f, scales[1], 6);
        }
        finally
        {
            rowScale.Dispose();
            packed.Dispose();
        }
    }

    [Fact]
    public void QuantizeFromF32_RefusesAGroupSizeThatDoesNotDivideTheInputDimension()
    {
        // A partial trailing group would rotate a window that runs off the row, so it refuses rather than pack it.
        using Tensor source = new Tensor(new TensorShape(2, 40), DType.F32);
        Assert.Throws<ArgumentException>(() => Int8ConvRotCodec.QuantizeFromF32(source, 16));
    }

    [Fact]
    public void QuantizeFromF32_RefusesAGroupSizeTheRotationCannotExpress()
    {
        using Tensor source = new Tensor(new TensorShape(2, 32), DType.F32);
        Assert.Throws<ArgumentOutOfRangeException>(() => Int8ConvRotCodec.QuantizeFromF32(source, 32));
    }

    private static Tensor Ramp(int rows, int columns)
    {
        Tensor t = new Tensor(new TensorShape(rows, columns), DType.F32);
        Span<float> values = t.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Sin(i * 0.19f) * 2.0f;
        }
        return t;
    }
}

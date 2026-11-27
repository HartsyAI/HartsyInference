using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Tensors;

/// <summary>The LongCat INT8 dequant as the reference's bf16 QuantizedLinear.forward computes it (quantization.py @ 58eb336, lines 27-31):
/// <c>weight_int8.to(bf16) * weight_scale.to(bf16)</c>, the product rounded to nearest-even. The default truncating path is unchanged.</summary>
public sealed class Int8ReferenceRoundingTests
{
    private static (Tensor Weight, Tensor Scale) Layer(sbyte[,] levels, float[] scales)
    {
        int rows = levels.GetLength(0), cols = levels.GetLength(1);
        Tensor weight = new(new TensorShape(rows, cols), DType.I8);
        Span<sbyte> w = weight.AsSpan<sbyte>();
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                w[r * cols + c] = levels[r, c];
        Tensor scale = new(new TensorShape(rows), DType.F32);
        scales.AsSpan().CopyTo(scale.AsSpan<float>());
        return (weight, scale);
    }

    /// <summary>The reference's value: the level and the scale in BF16, their product rounded to nearest-even.</summary>
    private static ushort Reference(sbyte level, float scale)
    {
        float scaleBf16 = BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(scale));
        return BFloat16Bits.FromFloat(level * scaleBf16);
    }

    [Fact]
    public void TheReferenceRoundingMatchesTheBf16ProductOnEveryElement()
    {
        sbyte[,] levels = { { -128, -1, 3 }, { 127, 5, 7 } };
        float[] scales = { 0.0123456f, 1.7f };
        (Tensor weight, Tensor scale) = Layer(levels, scales);
        using (weight)
        using (scale)
        {
            using Tensor dequant = Int8ConvRotCodec.DequantToBf16(weight, scale, convRotGroupSize: 0, referenceRounding: true);
            ReadOnlySpan<ushort> bits = dequant.AsReadOnlySpan<ushort>();
            for (int r = 0; r < 2; r++)
                for (int c = 0; c < 3; c++)
                    Assert.Equal(Reference(levels[r, c], scales[r]), bits[r * 3 + c]);
        }
    }

    [Fact]
    public void TruncationAndTheReferenceRoundingDifferWhereTheScaleRoundsUp()
    {
        // 0.1f is 0x3DCCCCCD: truncation keeps 0x3DCC, the reference rounds its BF16 form up to 0x3DCD.
        (Tensor weight, Tensor scale) = Layer(new sbyte[,] { { 1 } }, new[] { 0.1f });
        using (weight)
        using (scale)
        {
            using Tensor truncated = Int8ConvRotCodec.DequantToBf16(weight, scale, convRotGroupSize: 0);
            using Tensor rounded = Int8ConvRotCodec.DequantToBf16(weight, scale, convRotGroupSize: 0, referenceRounding: true);
            Assert.Equal(0x3DCC, truncated.AsReadOnlySpan<ushort>()[0]);
            Assert.Equal(0x3DCD, rounded.AsReadOnlySpan<ushort>()[0]);
            Assert.Equal(Reference(1, 0.1f), rounded.AsReadOnlySpan<ushort>()[0]);
        }
    }

    [Fact]
    public void TheReferenceRoundingRefusesARotatedWeight()
    {
        (Tensor weight, Tensor scale) = Layer(new sbyte[,] { { 1, 2, 3, 4 } }, new[] { 1f });
        using (weight)
        using (scale)
        {
            Assert.Throws<ArgumentException>(() => Int8ConvRotCodec.DequantToBf16(weight, scale, convRotGroupSize: 2, referenceRounding: true));
        }
    }
}

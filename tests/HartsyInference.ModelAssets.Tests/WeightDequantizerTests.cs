using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.ModelAssets.BlockScale;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed class WeightDequantizerTests
{
    private static readonly float[] Fp4Magnitudes = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f];

    private static double ScalarFp4(int nibble) => nibble == 8 ? 0.0 : ((nibble & 8) != 0 ? -1.0 : 1.0) * Fp4Magnitudes[nibble & 7];

    [Fact]
    public void Mxfp4_Weight_Decodes_Through_The_Dispatcher_Like_The_Codec()
    {
        const int rows = 3, cols = 64;
        Random rng = new(5);
        using Tensor scale = new(new TensorShape(rows, cols / 32), DType.F8E8M0);
        Span<byte> scaleBytes = scale.AsSpan<byte>();
        for (int i = 0; i < scaleBytes.Length; i++) scaleBytes[i] = (byte)rng.Next(110, 140);
        using Tensor weight = new(new TensorShape(rows, cols / 2), DType.I8);
        rng.NextBytes(weight.AsSpan<byte>());
        QuantRecipe recipe = new()
        {
            Encoding = QuantEncoding.Mxfp4E8M0, Geometry = new BlockGeometry(1, 32), ScaleDType = scale.DType,
            LogicalRows = rows, LogicalCols = cols, Scale = scale,
        };

        float[] decoded = WeightDequantizer.ToF32(weight, new QuantWeightInfo { Format = recipe.FormatName, Recipe = recipe });

        Assert.Equal(rows * cols, decoded.Length);
        ReadOnlySpan<byte> packed = weight.AsReadOnlySpan<byte>();
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
            {
                byte b = packed[r * cols / 2 + c / 2];
                int nibble = c % 2 == 0 ? b & 0xF : b >> 4;
                double expected = ScalarFp4(nibble) * Math.Pow(2, scaleBytes[r * (cols / 32) + c / 32] - 127);
                Assert.Equal((float)expected, decoded[r * cols + c]);
            }
    }

    [Fact]
    public void Unquantized_Bf16_Weight_Widens_To_F32()
    {
        using Tensor weight = new(new TensorShape(1, 2), DType.BF16);
        Span<ushort> bits = weight.AsSpan<ushort>();
        bits[0] = 0x3F80; // 1.0
        bits[1] = 0xC000; // -2.0
        Assert.Equal([1f, -2f], WeightDequantizer.ToF32(weight, null));
    }

    [Fact]
    public void Unwired_Encoding_Is_Refused()
    {
        using Tensor weight = new(new TensorShape(1, 4), DType.I8);
        QuantRecipe recipe = new()
        {
            Encoding = QuantEncoding.Gguf, Geometry = new BlockGeometry(1, 32), ScaleDType = DType.F32, LogicalRows = 1, LogicalCols = 4,
        };
        Assert.Throws<NotSupportedException>(() =>
            WeightDequantizer.ToF32(weight, new QuantWeightInfo { Format = recipe.FormatName, Recipe = recipe }));
    }
}

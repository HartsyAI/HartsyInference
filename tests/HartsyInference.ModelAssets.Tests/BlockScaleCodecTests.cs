using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.ModelAssets.BlockScale;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>The host codecs against decoders written here from the format definitions, not from the codec's own tables.</summary>
public sealed class BlockScaleCodecTests
{
    private static readonly float[] Fp4Magnitudes = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f];

    private static double ScalarE4M3(byte b)
    {
        int exp = (b >> 3) & 0xF, man = b & 7;
        if (exp == 15 && man == 7) return double.NaN;
        double magnitude = exp == 0 ? man / 8.0 * Math.Pow(2, -6) : (1 + man / 8.0) * Math.Pow(2, exp - 7);
        return (b & 0x80) != 0 ? -magnitude : magnitude;
    }

    private static double ScalarE8M0(byte e) => e == 255 ? double.NaN : Math.Pow(2, e - 127);

    // The reference FP4_TABLE holds +0.0, not -0.0, at nibble 8.
    private static double ScalarFp4(int nibble) => nibble == 8 ? 0.0 : ((nibble & 8) != 0 ? -1.0 : 1.0) * Fp4Magnitudes[nibble & 7];

    private static Tensor ScaleTensor(int rows, int cols, Random rng, DType? dtype = null)
    {
        Tensor scale = new Tensor(new TensorShape(rows, cols), dtype ?? DType.F8E8M0);
        Span<byte> bytes = scale.AsSpan<byte>();
        for (int i = 0; i < bytes.Length; i++)
            bytes[i] = (byte)rng.Next(96, 160);
        return scale;
    }

    private static QuantRecipe Recipe(QuantEncoding encoding, BlockGeometry geometry, long rows, long cols, Tensor scale) => new()
    {
        Encoding = encoding, Geometry = geometry, ScaleDType = scale.DType, LogicalRows = rows, LogicalCols = cols, Scale = scale,
    };

    private static byte[] RandomBytes(int count, Random rng)
    {
        byte[] bytes = new byte[count];
        rng.NextBytes(bytes);
        return bytes;
    }

    private static void AssertBitEqual(double expected, float actual, string where)
    {
        float expectedF = (float)expected;
        if (float.IsNaN(expectedF)) Assert.True(float.IsNaN(actual), where);
        else Assert.True(BitConverter.SingleToInt32Bits(expectedF) == BitConverter.SingleToInt32Bits(actual), $"{where}: {expectedF} vs {actual}");
    }

    [Theory]
    [InlineData(32, 32, 64, 96)]
    [InlineData(1, 32, 8, 96)]
    public void Fp8_MatchesScalarDecoderForEveryElement(int blockRows, int blockCols, int rows, int cols)
    {
        Random rng = new Random(11);
        BlockGeometry geometry = new BlockGeometry(blockRows, blockCols);
        (long scaleRows, long scaleCols) = geometry.ScaleShape(rows, cols);
        using Tensor scale = ScaleTensor((int)scaleRows, (int)scaleCols, rng);
        byte[] packed = RandomBytes(rows * cols, rng);
        QuantRecipe recipe = Recipe(QuantEncoding.Fp8E4M3BlockE8M0, geometry, rows, cols, scale);
        float[] dest = new float[rows * cols];

        Fp8BlockE8M0Codec.DequantRows(packed, recipe, 0, rows, dest);

        ReadOnlySpan<byte> scaleBytes = scale.AsReadOnlySpan<byte>();
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                double expected = ScalarE4M3(packed[r * cols + c]) * ScalarE8M0(scaleBytes[(r / blockRows) * (int)scaleCols + c / blockCols]);
                AssertBitEqual(expected, dest[r * cols + c], $"[{r},{c}]");
            }
        }
    }

    [Fact]
    public void Fp8_RowWindowEqualsSameRowsOfFullDecode()
    {
        Random rng = new Random(12);
        using Tensor scale = ScaleTensor(4, 3, rng);
        byte[] packed = RandomBytes(128 * 96, rng);
        QuantRecipe recipe = Recipe(QuantEncoding.Fp8E4M3BlockE8M0, new BlockGeometry(32, 32), 128, 96, scale);
        float[] full = new float[128 * 96];
        float[] window = new float[64 * 96];

        Fp8BlockE8M0Codec.DequantRows(packed, recipe, 0, 128, full);
        Fp8BlockE8M0Codec.DequantRows(packed, recipe, 40, 64, window);

        Assert.Equal(full.AsSpan(40 * 96, 64 * 96).ToArray(), window);
    }

    [Fact]
    public void Mxfp4_MatchesScalarDecoder_LowNibbleIsEvenElement()
    {
        Random rng = new Random(13);
        const int rows = 6, cols = 96;
        using Tensor scale = ScaleTensor(rows, cols / 32, rng);
        byte[] packed = RandomBytes(rows * cols / 2, rng);
        QuantRecipe recipe = Recipe(QuantEncoding.Mxfp4E8M0, new BlockGeometry(1, 32), rows, cols, scale);
        float[] dest = new float[rows * cols];

        Mxfp4E8M0Codec.DequantRows(packed, recipe, 0, rows, dest);

        ReadOnlySpan<byte> scaleBytes = scale.AsReadOnlySpan<byte>();
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                byte b = packed[r * cols / 2 + c / 2];
                int nibble = (c % 2 == 0) ? b & 0xF : b >> 4;
                AssertBitEqual(ScalarFp4(nibble) * ScalarE8M0(scaleBytes[r * (cols / 32) + c / 32]), dest[r * cols + c], $"[{r},{c}]");
            }
        }
    }

    [Fact]
    public void Mxfp4_PinsNibbleOrderAndPositiveZeroForNibble8()
    {
        using Tensor scale = new Tensor(new TensorShape(1, 1), DType.F8E8M0);
        scale.AsSpan<byte>()[0] = 128; // 2^1
        byte[] packed = new byte[16];
        packed[0] = 0x21;              // low nibble 1 (0.5) is element 0, high nibble 2 (1.0) is element 1
        packed[1] = 0x88;              // nibble 8 is +0.0 in the reference table
        QuantRecipe recipe = Recipe(QuantEncoding.Mxfp4E8M0, new BlockGeometry(1, 32), 1, 32, scale);
        float[] dest = new float[32];

        Mxfp4E8M0Codec.DequantRows(packed, recipe, 0, 1, dest);

        Assert.Equal(1.0f, dest[0]);
        Assert.Equal(2.0f, dest[1]);
        Assert.Equal(0, BitConverter.SingleToInt32Bits(dest[2]));
        Assert.Equal(0, BitConverter.SingleToInt32Bits(dest[3]));
    }

    [Fact]
    public void E8M0_ExtremesDecodeAsDefined()
    {
        using Tensor scale = new Tensor(new TensorShape(1, 2), DType.F8E8M0);
        scale.AsSpan<byte>()[0] = 0;   // 2^-127, a subnormal float, not zero
        scale.AsSpan<byte>()[1] = 255; // NaN
        byte[] packed = [0x38, 0x38];  // e4m3 1.0 twice
        QuantRecipe recipe = Recipe(QuantEncoding.Fp8E4M3BlockE8M0, new BlockGeometry(1, 1), 1, 2, scale);
        float[] dest = new float[2];

        Fp8BlockE8M0Codec.DequantRows(packed, recipe, 0, 1, dest);

        Assert.Equal((float)Math.Pow(2, -127), dest[0]);
        Assert.True(float.IsNaN(dest[1]));
    }

    [Fact]
    public void Mxfp4_SliceCols_DecodesAColumnContiguousCopyWithTheBorrowedScale()
    {
        Random rng = new Random(17);
        const int rows = 8, cols = 128, sliceStart = 64, sliceCols = 64;
        using Tensor scale = ScaleTensor(rows, cols / 32, rng);
        byte[] packed = RandomBytes(rows * cols / 2, rng);
        QuantRecipe recipe = Recipe(QuantEncoding.Mxfp4E8M0, new BlockGeometry(1, 32), rows, cols, scale);
        float[] full = new float[rows * cols];
        Mxfp4E8M0Codec.DequantRows(packed, recipe, 0, rows, full);

        QuantRecipe sliced = recipe.SliceCols(sliceStart, sliceCols, "w");
        byte[] slicedPacked = new byte[rows * sliceCols / 2];
        for (int r = 0; r < rows; r++)
            Array.Copy(packed, r * cols / 2 + sliceStart / 2, slicedPacked, r * sliceCols / 2, sliceCols / 2);
        float[] window = new float[rows * sliceCols];
        Mxfp4E8M0Codec.DequantRows(slicedPacked, sliced, 0, rows, window);

        for (int r = 0; r < rows; r++)
            Assert.Equal(full.AsSpan(r * cols + sliceStart, sliceCols).ToArray(), window.AsSpan(r * sliceCols, sliceCols).ToArray());
    }

    [Fact]
    public void SliceCols_OffBlockOrOddForFourBit_Refuses()
    {
        using Tensor scale = ScaleTensor(4, 4, new Random(18));
        QuantRecipe mxfp4 = Recipe(QuantEncoding.Mxfp4E8M0, new BlockGeometry(1, 32), 4, 128, scale);

        Assert.Throws<NotSupportedException>(() => mxfp4.SliceCols(16, 32, "w"));
        Assert.Throws<NotSupportedException>(() => mxfp4.SliceCols(32, 33, "w"));
    }

    [Fact]
    public void DequantRows_RejectsRecipeThatDoesNotDescribeTheBytes()
    {
        using Tensor scale = ScaleTensor(2, 2, new Random(19));
        QuantRecipe recipe = Recipe(QuantEncoding.Fp8E4M3BlockE8M0, new BlockGeometry(32, 32), 64, 64, scale);

        Assert.Throws<ArgumentException>(() => Fp8BlockE8M0Codec.DequantRows(new byte[100], recipe, 0, 1, new float[64]));
        Assert.Throws<ArgumentException>(() => Fp8BlockE8M0Codec.DequantRows(new byte[64 * 64], recipe, 0, 1, new float[63]));
        Assert.Throws<Core.Exceptions.HartsyInferenceException>(() => Mxfp4E8M0Codec.DequantRows(new byte[64 * 32], recipe, 0, 1, new float[64]));
    }
}

using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>ModelOpt NVFP4 and MLX affine host codecs against the python reference fixture and against scalar decoders written here.</summary>
public sealed class DerivativeQuantCodecTests
{
    private static readonly float[] Fp4 = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f, 0f, -0.5f, -1f, -1.5f, -2f, -3f, -4f, -6f];

    private static JsonElement Fixture()
    {
        string path = Path.Combine(RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures", "derivative_quant_codecs.json");
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static Tensor BytesTensor(byte[] bytes, DType dtype, params long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), dtype);
        bytes.CopyTo(tensor.AsSpan<byte>());
        return tensor;
    }

    private static Tensor FloatTensor(uint[] bits, params long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        Span<float> span = tensor.AsSpan<float>();
        for (int i = 0; i < bits.Length; i++) span[i] = BitConverter.UInt32BitsToSingle(bits[i]);
        return tensor;
    }

    private static uint[] RandomFloatBits(Random rng, int count) =>
        Enumerable.Range(0, count).Select(_ => BitConverter.SingleToUInt32Bits((float)rng.NextDouble())).ToArray();

    private static Tensor Scalar(float value)
    {
        Tensor tensor = new(new TensorShape(1), DType.F32);
        tensor.AsSpan<float>()[0] = value;
        return tensor;
    }

    private static uint[] Bits(JsonElement array) => array.EnumerateArray().Select(static e => e.GetUInt32()).ToArray();

    private static void AssertBitEqual(uint[] expected, float[] actual, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            uint got = BitConverter.SingleToUInt32Bits(actual[i]);
            if (float.IsNaN(BitConverter.UInt32BitsToSingle(expected[i]))) Assert.True(float.IsNaN(actual[i]), $"{name}[{i}] expected NaN");
            else Assert.True(expected[i] == got, $"{name}[{i}]: expected 0x{expected[i]:X8}, got 0x{got:X8}");
        }
    }

    private static QuantRecipe Nvfp4Recipe(long rows, long cols, Tensor scale, Tensor global, int blockRows = 1) => new()
    {
        Encoding = QuantEncoding.Nvfp4, Geometry = new BlockGeometry(blockRows, 16), ScaleDType = DType.F8E4M3,
        LogicalRows = rows, LogicalCols = cols, Scale = scale, GlobalScale = global,
    };

    private static QuantRecipe AffineRecipe(int bits, long rows, long cols, Tensor scale, Tensor bias) => new()
    {
        Encoding = bits == 4 ? QuantEncoding.AffineInt4 : QuantEncoding.AffineInt8, Geometry = new BlockGeometry(1, 64), ScaleDType = DType.F32,
        LogicalRows = rows, LogicalCols = cols, Scale = scale, Bias = bias,
    };

    private static double ScalarE4M3(byte b)
    {
        int exp = (b >> 3) & 0xF, man = b & 7;
        if (exp == 15 && man == 7) return double.NaN;
        double magnitude = exp == 0 ? man / 8.0 * Math.Pow(2, -6) : (1 + man / 8.0) * Math.Pow(2, exp - 7);
        return (b & 0x80) != 0 ? -magnitude : magnitude;
    }

    [Fact]
    public void ModelOptNvfp4_MatchesThePythonModeloptDecoder()
    {
        foreach (JsonElement c in Fixture().GetProperty("nvfp4").EnumerateArray())
        {
            int rows = c.GetProperty("rows").GetInt32(), cols = c.GetProperty("cols").GetInt32();
            using Tensor scale = BytesTensor(Convert.FromHexString(c.GetProperty("scaleHex").GetString()!), DType.F8E4M3, rows, cols / 16);
            using Tensor global = Scalar(BitConverter.UInt32BitsToSingle(c.GetProperty("globalScaleBits").GetUInt32()));
            QuantRecipe recipe = Nvfp4Recipe(rows, cols, scale, global);
            float[] actual = new float[rows * cols];

            ModelOptNvfp4Codec.DequantRows(Convert.FromHexString(c.GetProperty("packedHex").GetString()!), recipe, 0, rows, actual);

            AssertBitEqual(Bits(c.GetProperty("expectedBits")), actual, $"nvfp4 {rows}x{cols}");
        }
    }

    [Fact]
    public void QuarkU8ScaleMxfp4_MatchesThePythonE8m0Decoder()
    {
        foreach (JsonElement c in Fixture().GetProperty("quark").EnumerateArray())
        {
            int rows = c.GetProperty("rows").GetInt32(), cols = c.GetProperty("cols").GetInt32();
            using Tensor scale = BytesTensor(Convert.FromHexString(c.GetProperty("scaleHex").GetString()!), DType.U8, rows, cols / 32);
            QuantRecipe recipe = new()
            {
                Encoding = QuantEncoding.Mxfp4E8M0, Geometry = new BlockGeometry(1, 32), ScaleDType = DType.U8,
                LogicalRows = rows, LogicalCols = cols, Scale = scale,
            };
            float[] actual = new float[rows * cols];

            Mxfp4E8M0Codec.DequantRows(Convert.FromHexString(c.GetProperty("packedHex").GetString()!), recipe, 0, rows, actual);

            AssertBitEqual(Bits(c.GetProperty("expectedBits")), actual, $"quark {rows}x{cols}");
        }
    }

    [Fact]
    public void MlxAffine_MatchesMxDequantizeForFourAndEightBit()
    {
        int seen = 0;
        foreach (JsonElement c in Fixture().GetProperty("mlxAffine").EnumerateArray())
        {
            int rows = c.GetProperty("rows").GetInt32(), cols = c.GetProperty("cols").GetInt32(), bits = c.GetProperty("bits").GetInt32();
            using Tensor scale = FloatTensor(Bits(c.GetProperty("scaleBits")), rows, cols / 64);
            using Tensor bias = FloatTensor(Bits(c.GetProperty("biasBits")), rows, cols / 64);
            QuantRecipe recipe = AffineRecipe(bits, rows, cols, scale, bias);
            float[] actual = new float[rows * cols];

            AffineIntCodec.DequantRows(Convert.FromHexString(c.GetProperty("packedHex").GetString()!), recipe, 0, rows, actual);

            AssertBitEqual(Bits(c.GetProperty("expectedBits")), actual, $"mlx int{bits} {rows}x{cols}");
            seen++;
        }
        Assert.Equal(4, seen);
    }

    [Theory]
    [InlineData(4, 96, 1)]
    [InlineData(6, 64, 4)]
    public void ModelOptNvfp4_MatchesAScalarDecoderIncludingScaleExtremes(int rows, int cols, int blockRows)
    {
        Random rng = new(7);
        byte[] packed = new byte[rows * cols / 2];
        rng.NextBytes(packed);
        byte[] scaleBytes = new byte[(rows + blockRows - 1) / blockRows * (cols / 16)];
        rng.NextBytes(scaleBytes);
        (scaleBytes[0], scaleBytes[1], scaleBytes[2], scaleBytes[3]) = (0x7F, 0xFF, 0x80, 0x7E);
        using Tensor scale = BytesTensor(scaleBytes, DType.F8E4M3, scaleBytes.Length / (cols / 16), cols / 16);
        using Tensor global = Scalar(0.37f);
        float[] actual = new float[rows * cols];

        ModelOptNvfp4Codec.DequantRows(packed, Nvfp4Recipe(rows, cols, scale, global, blockRows), 0, rows, actual);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                byte b = packed[r * cols / 2 + c / 2];
                int nibble = (c & 1) == 0 ? b & 15 : b >> 4;
                float s = (float)ScalarE4M3(scaleBytes[(r / blockRows) * (cols / 16) + c / 16]) * 0.37f;
                float expected = Fp4[nibble] * s;
                float got = actual[r * cols + c];
                if (float.IsNaN(expected)) Assert.True(float.IsNaN(got), $"[{r},{c}]");
                else Assert.Equal(BitConverter.SingleToUInt32Bits(expected), BitConverter.SingleToUInt32Bits(got));
            }
        }
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    public void MlxAffine_MatchesAScalarDecoderAndReadsLowestFieldFirst(int bits)
    {
        Random rng = new(bits);
        const int rows = 3, cols = 192;
        byte[] packed = new byte[rows * cols * bits / 8];
        rng.NextBytes(packed);
        float[] scaleValues = new float[rows * 3], biasValues = new float[rows * 3];
        for (int i = 0; i < scaleValues.Length; i++)
        {
            scaleValues[i] = (float)(rng.NextDouble() * 0.05);
            biasValues[i] = (float)(rng.NextDouble() - 0.5);
        }
        using Tensor scale = FloatTensor(scaleValues.Select(BitConverter.SingleToUInt32Bits).ToArray(), rows, 3);
        using Tensor bias = FloatTensor(biasValues.Select(BitConverter.SingleToUInt32Bits).ToArray(), rows, 3);
        float[] actual = new float[rows * cols];

        AffineIntCodec.DequantRows(packed, AffineRecipe(bits, rows, cols, scale, bias), 0, rows, actual);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                // A U32 word is little-endian, so its lowest field is the first element of the group of 32/bits.
                int perWord = 32 / bits;
                uint word = BitConverter.ToUInt32(packed, (r * (cols / perWord) + c / perWord) * 4);
                int q = (int)((word >> (bits * (c % perWord))) & ((1u << bits) - 1));
                float product = q * scaleValues[r * 3 + c / 64];
                float expected = product + biasValues[r * 3 + c / 64];
                Assert.Equal(BitConverter.SingleToUInt32Bits(expected), BitConverter.SingleToUInt32Bits(actual[r * cols + c]));
            }
        }
    }

    [Fact]
    public void ModelOptNvfp4_DoesNotDecodeTheComfyLayout()
    {
        // Comfy packs the high nibble first; a byte 0x21 must give element 0 = 0.5 (low nibble 1) here.
        using Tensor scale = BytesTensor([0x38], DType.F8E4M3, 1, 1);
        using Tensor global = Scalar(1f);
        float[] dest = new float[16];
        byte[] packed = new byte[8];
        packed[0] = 0x21;

        ModelOptNvfp4Codec.DequantRows(packed, Nvfp4Recipe(1, 16, scale, global), 0, 1, dest);

        Assert.Equal([0.5f, 1f], dest.AsSpan(0, 2).ToArray());
    }

    [Fact]
    public void Codecs_RefuseRecipesThatDoNotDescribeTheirInputs()
    {
        using Tensor scale = BytesTensor(new byte[4], DType.F8E4M3, 1, 4);
        byte[] packed = new byte[32];
        float[] dest = new float[64];
        QuantRecipe noGlobal = Nvfp4Recipe(1, 64, scale, Scalar(1f)) with { GlobalScale = null };
        HartsyInferenceException missingGlobal =
            Assert.Throws<HartsyInferenceException>(() => ModelOptNvfp4Codec.DequantRows(packed, noGlobal, 0, 1, dest));
        Assert.Contains("global scale", missingGlobal.Message);
        using Tensor vector = new(new TensorShape(2), DType.F32);
        QuantRecipe vectorGlobal = Nvfp4Recipe(1, 64, scale, vector);
        Assert.Throws<HartsyInferenceException>(() => ModelOptNvfp4Codec.DequantRows(packed, vectorGlobal, 0, 1, dest));
        using Tensor e8m0 = BytesTensor(new byte[4], DType.F8E8M0, 1, 4);
        QuantRecipe wrongScale = Nvfp4Recipe(1, 64, e8m0, Scalar(1f)) with { ScaleDType = DType.F8E8M0 };
        Assert.Throws<NotSupportedException>(() => ModelOptNvfp4Codec.DequantRows(packed, wrongScale, 0, 1, dest));
        Assert.Throws<ArgumentException>(() => ModelOptNvfp4Codec.DequantRows(new byte[31], Nvfp4Recipe(1, 64, scale, Scalar(1f)), 0, 1, dest));

        using Tensor f32Scale = FloatTensor(new uint[2], 1, 2);
        using Tensor f32Bias = FloatTensor(new uint[2], 1, 2);
        QuantRecipe noBias = AffineRecipe(4, 1, 128, f32Scale, f32Bias) with { Bias = null };
        HartsyInferenceException missingBias =
            Assert.Throws<HartsyInferenceException>(() => AffineIntCodec.DequantRows(new byte[64], noBias, 0, 1, new float[128]));
        Assert.Contains("bias", missingBias.Message);
        using Tensor shortBias = FloatTensor(new uint[1], 1, 1);
        QuantRecipe mismatched = AffineRecipe(4, 1, 128, f32Scale, shortBias);
        Assert.Throws<HartsyInferenceException>(() => AffineIntCodec.DequantRows(new byte[64], mismatched, 0, 1, new float[128]));
        using Tensor bf16Scale = BytesTensor(new byte[4], DType.BF16, 1, 2);
        QuantRecipe bf16 = AffineRecipe(4, 1, 128, bf16Scale, bf16Scale) with { ScaleDType = DType.BF16 };
        Assert.Throws<NotSupportedException>(() => AffineIntCodec.DequantRows(new byte[64], bf16, 0, 1, new float[128]));
        QuantRecipe notAffine = AffineRecipe(4, 1, 128, f32Scale, f32Bias) with { Encoding = QuantEncoding.Nvfp4 };
        Assert.Throws<HartsyInferenceException>(() => AffineIntCodec.DequantRows(new byte[64], notAffine, 0, 1, new float[128]));
    }

    [Fact]
    public void FormatNames_AreDistinctFromTheComfyNamesBlackwellRoutesOn()
    {
        using Tensor scale = new(new TensorShape(1, 1), DType.F32);
        QuantRecipe nvfp4 = Nvfp4Recipe(1, 16, scale, scale);
        Assert.Equal("recipe-nvfp4", nvfp4.FormatName);
        Assert.Equal("recipe-affine-int4", AffineRecipe(4, 1, 64, scale, scale).FormatName);
        Assert.Equal("recipe-affine-int8", AffineRecipe(8, 1, 64, scale, scale).FormatName);
        Assert.Equal(2, AffineRecipe(4, 1, 64, scale, scale).ElementsPerByte);
        Assert.Equal(1, AffineRecipe(8, 1, 64, scale, scale).ElementsPerByte);
        Assert.Equal(2, nvfp4.ElementsPerByte);
    }

    [Fact]
    public void Nvfp4_SliceRows_KeepsTheGlobalScaleAndDecodesTheWindowLikeTheFullMatrix()
    {
        Random rng = new(31);
        const int rows = 8, cols = 64;
        byte[] packed = new byte[rows * cols / 2];
        rng.NextBytes(packed);
        byte[] scaleBytes = new byte[rows * cols / 16];
        for (int i = 0; i < scaleBytes.Length; i++) scaleBytes[i] = (byte)rng.Next(0x20, 0x70);
        using Tensor scale = BytesTensor(scaleBytes, DType.F8E4M3, rows, cols / 16);
        using Tensor global = Scalar(0.02f);
        QuantRecipe recipe = Nvfp4Recipe(rows, cols, scale, global);
        float[] full = new float[rows * cols];
        ModelOptNvfp4Codec.DequantRows(packed, recipe, 0, rows, full);

        QuantRecipe sliced = recipe.SliceRows(3, 4, "w");
        float[] window = new float[4 * cols];
        ModelOptNvfp4Codec.DequantRows(packed.AsSpan(3 * cols / 2, 4 * cols / 2), sliced, 0, 4, window);

        Assert.Same(global, sliced.GlobalScale);
        Assert.Equal(full.AsSpan(3 * cols, 4 * cols).ToArray(), window);
    }

    [Fact]
    public void Nvfp4_SliceCols_BorrowsTheScaleWithAColumnOffset()
    {
        Random rng = new(32);
        const int rows = 4, cols = 128, start = 48, count = 64;
        byte[] packed = new byte[rows * cols / 2];
        rng.NextBytes(packed);
        byte[] scaleBytes = new byte[rows * cols / 16];
        for (int i = 0; i < scaleBytes.Length; i++) scaleBytes[i] = (byte)rng.Next(0x20, 0x70);
        using Tensor scale = BytesTensor(scaleBytes, DType.F8E4M3, rows, cols / 16);
        using Tensor global = Scalar(1.5f);
        QuantRecipe recipe = Nvfp4Recipe(rows, cols, scale, global);
        float[] full = new float[rows * cols];
        ModelOptNvfp4Codec.DequantRows(packed, recipe, 0, rows, full);

        QuantRecipe sliced = recipe.SliceCols(start, count, "w");
        byte[] slicedPacked = new byte[rows * count / 2];
        for (int r = 0; r < rows; r++) Array.Copy(packed, r * cols / 2 + start / 2, slicedPacked, r * count / 2, count / 2);
        float[] window = new float[rows * count];
        ModelOptNvfp4Codec.DequantRows(slicedPacked, sliced, 0, rows, window);

        Assert.Equal(start / 16, sliced.ScaleColOffset);
        for (int r = 0; r < rows; r++) Assert.Equal(full.AsSpan(r * cols + start, count).ToArray(), window.AsSpan(r * count, count).ToArray());
        Assert.Throws<NotSupportedException>(() => recipe.SliceCols(8, 32, "w"));
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    public void MlxAffine_SliceRowsAndCols_SliceTheBiasWithTheScale(int bits)
    {
        Random rng = new(40 + bits);
        const int rows = 6, cols = 256, start = 128, count = 128;
        byte[] packed = new byte[rows * cols * bits / 8];
        rng.NextBytes(packed);
        uint[] scaleBits = Enumerable.Range(0, rows * 4).Select(_ => BitConverter.SingleToUInt32Bits((float)(rng.NextDouble() * 0.1))).ToArray();
        uint[] biasBits = Enumerable.Range(0, rows * 4).Select(_ => BitConverter.SingleToUInt32Bits((float)(rng.NextDouble() - 0.5))).ToArray();
        using Tensor scale = FloatTensor(scaleBits, rows, 4);
        using Tensor bias = FloatTensor(biasBits, rows, 4);
        QuantRecipe recipe = AffineRecipe(bits, rows, cols, scale, bias);
        float[] full = new float[rows * cols];
        AffineIntCodec.DequantRows(packed, recipe, 0, rows, full);

        QuantRecipe rowSlice = recipe.SliceRows(2, 3, "w");
        float[] rowWindow = new float[3 * cols];
        int rowBytes = cols * bits / 8;
        AffineIntCodec.DequantRows(packed.AsSpan(2 * rowBytes, 3 * rowBytes), rowSlice, 0, 3, rowWindow);
        Assert.Equal(full.AsSpan(2 * cols, 3 * cols).ToArray(), rowWindow);
        Assert.Equal(new TensorShape(3, 4), rowSlice.Bias!.Shape);

        QuantRecipe colSlice = recipe.SliceCols(start, count, "w");
        int sliceBytes = count * bits / 8;
        byte[] slicedPacked = new byte[rows * sliceBytes];
        for (int r = 0; r < rows; r++) Array.Copy(packed, r * rowBytes + start * bits / 8, slicedPacked, r * sliceBytes, sliceBytes);
        float[] colWindow = new float[rows * count];
        AffineIntCodec.DequantRows(slicedPacked, colSlice, 0, rows, colWindow);
        Assert.Equal(start / 64, colSlice.ScaleColOffset);
        for (int r = 0; r < rows; r++) Assert.Equal(full.AsSpan(r * cols + start, count).ToArray(), colWindow.AsSpan(r * count, count).ToArray());

        Assert.Throws<NotSupportedException>(() => recipe.SliceCols(32, 64, "w"));
        Assert.Throws<NotSupportedException>(() => recipe.SliceCols(64, 33, "w"));
    }

    [Fact]
    public void RowWindowedDecode_MatchesTheFullDecodeForBothFormats()
    {
        Random rng = new(50);
        const int rows = 5, cols = 128;
        byte[] packed = new byte[rows * cols / 2];
        rng.NextBytes(packed);
        using Tensor scale = FloatTensor(RandomFloatBits(rng, rows * 2), rows, 2);
        using Tensor bias = FloatTensor(RandomFloatBits(rng, rows * 2), rows, 2);
        QuantRecipe recipe = AffineRecipe(4, rows, cols, scale, bias);
        float[] full = new float[rows * cols];
        AffineIntCodec.DequantRows(packed, recipe, 0, rows, full);

        float[] window = new float[2 * cols];
        AffineIntCodec.DequantRows(packed, recipe, 2, 2, window);

        Assert.Equal(full.AsSpan(2 * cols, 2 * cols).ToArray(), window);
    }
}

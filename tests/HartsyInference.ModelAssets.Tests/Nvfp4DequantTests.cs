using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Covers NVFP4 (ComfyUI comfy_quant "nvfp4") dequantization: U8 nibble-packed e2m1 values with
/// per-16-element F8-E4M3 block scales and a global F32 scale. Nibble order is comfy_kitchen's
/// <c>hi_first=True</c>: element 2j in the HIGH nibble, 2j+1 in the LOW nibble.</summary>
public sealed unsafe class Nvfp4DequantTests
{
    private readonly ITestOutputHelper _output;

    public Nvfp4DequantTests(ITestOutputHelper output) => _output = output;

    /// <summary>Packs a pair of e2m1 nibbles hi-first, as comfy_kitchen does.</summary>
    private static byte Pack(byte even, byte odd) => (byte)((even << 4) | odd);

    [Fact]
    public void DequantNvfp4_KnownNibbles_ExactValues()
    {
        // One row, one 16-element block (8 packed bytes). Values chosen to hit every e2m1 magnitude
        // plus both signs: nibble bits are [s e1 e0 m] → 0..7 = +{0, .5, 1, 1.5, 2, 3, 4, 6}, 8..15 = negated.
        using Tensor packed = new(new TensorShape(1, 8), DType.U8);
        byte* p = (byte*)packed.DataPointer;
        p[0] = Pack(0, 1);   // elements 0,1 = 0, 0.5
        p[1] = Pack(2, 3);   // 1, 1.5
        p[2] = Pack(4, 5);   // 2, 3
        p[3] = Pack(6, 7);   // 4, 6
        p[4] = Pack(9, 10);  // -0.5, -1
        p[5] = Pack(11, 12); // -1.5, -2
        p[6] = Pack(13, 14); // -3, -4
        p[7] = Pack(15, 8);  // -6, -0

        // Block scale = 2.0 encoded as e4m3 (0x40: exp=8 → 2^(8-7) * 1.0), global scale 0.5 → net 1.0.
        using Tensor blockScales = new(new TensorShape(1, 1), DType.F8E4M3);
        ((byte*)blockScales.DataPointer)[0] = 0x40;

        using Tensor result = CheckpointConvertUtils.DequantNvfp4ToF16(packed, blockScales, globalScale: 0.5f);

        Assert.Equal(DType.F16, result.DType);
        Assert.Equal(new long[] { 1, 16 }, new[] { result.Shape[0], result.Shape[1] });
        float[] expected = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f, -0.5f, -1f, -1.5f, -2f, -3f, -4f, -6f, -0f];
        Half* r = (Half*)result.DataPointer;
        for (int i = 0; i < 16; i++)
            Assert.Equal(expected[i], (float)r[i]);
    }

    [Fact]
    public void DequantNvfp4ToFp8_MatchesF16Path_WithinFp8Precision()
    {
        // Same 16-element block as the exact-values test, but dequantized to fp8 (block scale ÷8 folded into the value,
        // 8 · global scale on Fp8ScaleFactor). fp8_value * Fp8ScaleFactor must reproduce the F16 dequant.
        using Tensor packed = new(new TensorShape(1, 8), DType.U8);
        byte* p = (byte*)packed.DataPointer;
        p[0] = Pack(0, 1); p[1] = Pack(2, 3); p[2] = Pack(4, 5); p[3] = Pack(6, 7);
        p[4] = Pack(9, 10); p[5] = Pack(11, 12); p[6] = Pack(13, 14); p[7] = Pack(15, 8);
        using Tensor blockScales = new(new TensorShape(1, 1), DType.F8E4M3);
        ((byte*)blockScales.DataPointer)[0] = 0x40; // 2.0
        const float g = 0.5f;

        using Tensor f16 = CheckpointConvertUtils.DequantNvfp4ToF16(packed, blockScales, g);
        using Tensor fp8 = CheckpointConvertUtils.DequantNvfp4ToFp8(packed, blockScales, g);

        Assert.Equal(DType.F8E4M3, fp8.DType);
        Assert.Equal(8f * g, fp8.Fp8ScaleFactor);
        // fp8 stores e2m1*block_scale/8 (= the F16 value / 8g); real value = decoded_fp8 * Fp8ScaleFactor.
        using Tensor fp8AsF32 = fp8.CastTo(DType.F32);          // CastTo folds Fp8ScaleFactor into the value
        Half* a = (Half*)f16.DataPointer;
        float* b = (float*)fp8AsF32.DataPointer;
        for (int i = 0; i < 16; i++)
            Assert.True(MathF.Abs((float)a[i] - b[i]) < 1e-3f, $"idx {i}: F16={(float)a[i]} fp8*scale={b[i]}");
    }

    /// <summary>The fp8 fold over every (e2m1 nibble, finite block-scale byte) pair, against bounds taken from e4m3's
    /// format alone and fixed before the fold runs: half an ulp (2⁻⁴ relative) for a normal value, half a subnormal step
    /// at the folded scale (2⁻⁷ · g) below it, and no clipping — <c>e2m1 · block_scale</c> reaches 6 · 448 = 2688, past
    /// e4m3's 448. The negative control folds without the ÷8 and requires the same bounds to reject it.</summary>
    [Fact]
    public void DequantNvfp4ToFp8_EveryNibbleAndScaleByte_StaysWithinE4M3Rounding()
    {
        // One 128-row x 4-block-column swizzle tile; its 512 blocks cycle through the 127 finite non-negative scale
        // bytes (0x7F is NaN) and each block holds all 16 nibbles in element order.
        const int Rows = 128, BlockCols = 4, Cols = BlockCols * 16, PackedCols = Cols / 2;
        const float G = 0.37f;
        static byte ScaleByte(int row, int bcol) => (byte)((row * BlockCols + bcol) % 127);
        static double E4M3(byte b)
        {
            int e = (b >> 3) & 0x0F, m = b & 0x07;
            return e == 0 ? m / 8.0 * Math.Pow(2, -6) : (1 + m / 8.0) * Math.Pow(2, e - 7);
        }
        double[] e2m1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
        double Expected(int row, int col)
        {
            int nibble = col % 16;
            double magnitude = e2m1[nibble & 7] * E4M3(ScaleByte(row, col / 16)) * G;
            return (nibble & 8) != 0 ? -magnitude : magnitude;
        }
        // The 1e-6 slack is for the F32 multiply by Fp8ScaleFactor in CastTo; exact half-ulp ties sit on the bound.
        static double Bound(double expected) => Math.Max(Math.Pow(2, -4) * Math.Abs(expected), Math.Pow(2, -7) * G) * (1 + 1e-6);

        using Tensor packed = new(new TensorShape(Rows, PackedCols), DType.U8);
        byte* p = (byte*)packed.DataPointer;
        for (int i = 0; i < Rows * PackedCols; i++)
            p[i] = Pack((byte)(2 * (i % 8)), (byte)(2 * (i % 8) + 1));
        using Tensor blockScales = new(new TensorShape(Rows, BlockCols), DType.F8E4M3);
        byte* s = (byte*)blockScales.DataPointer;
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < BlockCols; c++)
                s[BlockScaleSwizzle.SwizzledIndex(r, c, BlockCols)] = ScaleByte(r, c);
        }

        using Tensor fp8 = CheckpointConvertUtils.DequantNvfp4ToFp8(packed, blockScales, G);
        using Tensor real = fp8.CastTo(DType.F32);          // folds Fp8ScaleFactor into the value
        float* got = (float*)real.DataPointer;
        int pastE4M3Max = 0;
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < Cols; c++)
            {
                double expected = Expected(r, c);
                double error = Math.Abs(got[r * Cols + c] - expected);
                Assert.True(error <= Bound(expected), $"row {r} col {c}: expected {expected}, got {got[r * Cols + c]}");
                if (Math.Abs(expected) / G > 448) pastE4M3Max++;
            }
        }
        Assert.True(pastE4M3Max > 0, "no folded value exceeds e4m3's 448, so the clipping case is not covered");

        // Negative control: the same fold without the ÷8 (block scale only, g on Fp8ScaleFactor) clips past 448.
        using Tensor unscaled = new(new TensorShape(Rows, Cols), DType.F32);
        float* u = (float*)unscaled.DataPointer;
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < Cols; c++)
                u[r * Cols + c] = (float)(Expected(r, c) / G);
        }
        using Tensor unscaledFp8 = unscaled.CastTo(DType.F8E4M3);
        unscaledFp8.Fp8ScaleFactor = G;
        using Tensor unscaledReal = unscaledFp8.CastTo(DType.F32);
        float* bad = (float*)unscaledReal.DataPointer;
        int rejected = 0;
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < Cols; c++)
            {
                double expected = Expected(r, c);
                if (Math.Abs(bad[r * Cols + c] - expected) > Bound(expected)) rejected++;
            }
        }
        _output.WriteLine($"{pastE4M3Max} of {Rows * Cols} values past 448 before the ÷8; the unscaled fold breaks the bound on {rejected}.");
        Assert.True(rejected > 0, "the bounds accept the unscaled fold, so they cannot see clipping");
    }

    /// <summary>The block-scale matrix ComfyUI stores is in NVIDIA's blocked layout, not row-major, so the
    /// dequant has to invert the permutation to find the scale for a logical <c>(row, blockColumn)</c>.
    ///
    /// <para>Reading it row-major instead is silent: the stored and padded shapes are identical whenever rows are a
    /// multiple of 128 and block columns a multiple of 4, so no shape check fires and the weights simply come out
    /// wrong. Measured against a BF16 copy of the same real tensor (Qwen3-8B <c>layers.0.self_attn.k_proj</c>, the
    /// Klein 9B encoder), row-major correlates 0.9186 where the swizzled read correlates 0.9954 — degraded output,
    /// never an exception.</para>
    ///
    /// <para>Every scale here is distinct, so any index that lands on the wrong one changes the result. The final
    /// assertion is the negative control: it pins that a row-major read would actually disagree, without which this
    /// test would pass against the bug it exists to catch.</para></summary>
    [Fact]
    public void DequantNvfp4_ReadsBlockScalesThroughTheSwizzle_NotRowMajor()
    {
        // 128 rows x 8 block columns is exactly one swizzle tile: paddedRows == 128, paddedCols == 8, so the
        // permutation is onto and every stored byte is addressed. 8 block columns = 128 values = 64 packed bytes.
        const int Rows = 128, BlockCols = 8, Cols = BlockCols * 16, PackedCols = Cols / 2;

        // A distinct E4M3 exponent per (row, blockColumn): byte 0x30..0x3F is 2^(e-7) for e = 6..7 and friends,
        // all positive powers of two, so the expected product stays exactly representable in F16.
        byte ScaleByte(int row, int bcol) => (byte)(0x30 + ((row + bcol * 5) % 8));
        static float ScaleValue(byte b)
        {
            int e = (b >> 3) & 0x0F, m = b & 0x07;
            return e == 0 ? m / 8.0f * MathF.Pow(2, -6) : (1 + m / 8.0f) * MathF.Pow(2, e - 7);
        }

        using Tensor packed = new(new TensorShape(Rows, PackedCols), DType.U8);
        byte* p = (byte*)packed.DataPointer;
        for (long i = 0; i < packed.ElementCount; i++) p[i] = Pack(2, 4);   // e2m1 1.0 and 2.0

        using Tensor blockScales = new(new TensorShape(Rows, BlockCols), DType.F8E4M3);
        byte* s = (byte*)blockScales.DataPointer;
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < BlockCols; c++)
            {
                s[BlockScaleSwizzle.SwizzledIndex(r, c, BlockCols)] = ScaleByte(r, c);
            }
        }

        using Tensor result = CheckpointConvertUtils.DequantNvfp4ToF16(packed, blockScales, globalScale: 1f);
        Half* got = (Half*)result.DataPointer;

        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < Cols; c++)
            {
                float expected = (c % 2 == 0 ? 1.0f : 2.0f) * ScaleValue(ScaleByte(r, c / 16));
                Assert.Equal(expected, (float)got[(long)r * Cols + c]);
            }
        }

        // Negative control: the pre-fix reading. If this agreed, the test above could not tell the two apart.
        int disagreements = 0;
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < BlockCols; c++)
            {
                if (s[r * BlockCols + c] != ScaleByte(r, c)) disagreements++;
            }
        }
        Assert.True(disagreements > 0,
            "row-major and swizzled indexing agree on this fixture, so it cannot detect the layout bug");
        _output.WriteLine($"row-major would read a different scale at {disagreements}/{Rows * BlockCols} positions");
    }

    [Fact]
    public void Quantize_RoundTrip_WithinBoundPerBlock()
    {
        const int rows = 4, cols = 64;
        float[] src = new float[rows * cols];
        Random rng = new(11);
        for (int i = 0; i < src.Length; i++) src[i] = (float)(rng.NextDouble() * 2 - 1) * (1 + i % 7);

        (byte[] packed, byte[] scales, float global) = ModelOptNvfp4Codec.Quantize(src, rows, cols);
        float[] dec = Decode(packed, scales, global, rows, cols);

        AssertWithinBlockBound(src, dec, rows, cols, global);
    }

    [Fact]
    public void Quantize_ZeroMatrix_DecodesToZero()
    {
        float[] src = new float[32];
        (byte[] packed, byte[] scales, float global) = ModelOptNvfp4Codec.Quantize(src, 2, 16);
        Assert.Equal(1f, global);
        foreach (float v in Decode(packed, scales, global, 2, 16)) Assert.Equal(0f, v);
    }

    [Fact]
    public void Quantize_RejectsColumnsNotMultipleOf16()
    {
        Assert.Throws<ArgumentException>(() => ModelOptNvfp4Codec.Quantize(new float[2 * 12], 2, 12));
    }

    [Theory]
    [InlineData(-1L, -16L)]
    [InlineData(0L, 16L)]
    [InlineData(2L, -16L)]
    public void Quantize_RejectsNonPositiveDimensions(long rows, long cols)
    {
        Assert.Throws<ArgumentException>(() => ModelOptNvfp4Codec.Quantize(new float[16], rows, cols));
    }

    [Fact]
    public void Quantize_RejectsOverflowingDimensions()
    {
        Assert.Throws<ArgumentException>(() => ModelOptNvfp4Codec.Quantize(new float[16], long.MaxValue / 8, 16));
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void Quantize_RejectsNonFiniteValues(float bad)
    {
        float[] src = new float[16];
        src[7] = bad;
        Assert.Throws<ArgumentException>(() => ModelOptNvfp4Codec.Quantize(src, 1, 16));
    }

    [Fact]
    public void Quantize_RejectsMagnitudeWhoseGlobalScaleUnderflows()
    {
        float[] src = new float[16];
        src[0] = 1e-40f;
        Assert.Throws<ArgumentException>(() => ModelOptNvfp4Codec.Quantize(src, 1, 16));
    }

    [Fact]
    public void Quantize_BlockBelowNormalScaleRange_StaysWithinSubnormalBound()
    {
        // The first block is 1e-6 of the tensor maximum: its E4M3 scale falls below the normal range and
        // can underflow to zero, so it is held to the absolute subnormal bound instead of the relative one.
        const int cols = 32;
        float[] src = new float[cols];
        for (int j = 0; j < 16; j++) src[j] = (j % 2 == 0 ? 1f : -1f) * 1e-6f * (1 + j % 5);
        for (int j = 16; j < 32; j++) src[j] = (j % 3 - 1) * 1f;

        (byte[] packed, byte[] scales, float global) = ModelOptNvfp4Codec.Quantize(src, 1, cols);
        float[] dec = Decode(packed, scales, global, 1, cols);

        AssertWithinBlockBound(src, dec, 1, cols, global);
    }

    /// <summary>Asserts each element is within the bound that <see cref="ModelOptNvfp4Codec.Quantize"/> guarantees for its block.
    /// Blocks whose maximum is at least <see cref="ModelOptNvfp4Codec.NormalScaleBlockFraction"/> of the tensor maximum get
    /// <see cref="ModelOptNvfp4Codec.MaxRoundTripError"/> of their block maximum; smaller blocks get
    /// <see cref="ModelOptNvfp4Codec.SubnormalBlockError"/> of the global scale.</summary>
    private static void AssertWithinBlockBound(float[] src, float[] dec, int rows, int cols, float global, string label = "element")
    {
        float amax = 0f;
        foreach (float v in src) amax = MathF.Max(amax, MathF.Abs(v));
        float normalThreshold = amax * ModelOptNvfp4Codec.NormalScaleBlockFraction;

        for (int r = 0; r < rows; r++)
        for (int blk = 0; blk < cols / 16; blk++)
        {
            float blockMax = 0f;
            for (int j = 0; j < 16; j++) blockMax = MathF.Max(blockMax, MathF.Abs(src[r * cols + blk * 16 + j]));
            float bound = blockMax >= normalThreshold
                ? blockMax * ModelOptNvfp4Codec.MaxRoundTripError
                : ModelOptNvfp4Codec.SubnormalBlockError * global;
            for (int j = 0; j < 16; j++)
            {
                int i = r * cols + blk * 16 + j;
                Assert.True(MathF.Abs(dec[i] - src[i]) <= bound * (1f + 1e-5f), $"{label}[{i}]: {dec[i]} vs {src[i]}, bound {bound}");
            }
        }
    }

    [Fact]
    public void QuantizeToCompanions_RoundTripsThroughProductionLoader()
    {
        const int rows = 4, cols = 48;
        float[] src = new float[rows * cols];
        Random rng = new(5);
        for (int i = 0; i < src.Length; i++) src[i] = (float)(rng.NextDouble() * 2 - 1) * (1 + i % 5);
        string dir = Directory.CreateTempSubdirectory("nvfp4-companions-").FullName;
        string path = Path.Combine(dir, "nvfp4.safetensors");
        try
        {
            List<KeyValuePair<string, Tensor>> companions = ModelOptNvfp4Codec.QuantizeToCompanions("layer.weight", src, rows, cols);
            SafeTensorsMerger.Write(path, companions);
            foreach (KeyValuePair<string, Tensor> c in companions) c.Value.Dispose();

            (List<KeyValuePair<string, Tensor>> tensors, List<SafeTensorsLoader> loaders) = SafeTensorsMerger.Open(path);
            try
            {
                Dictionary<string, Tensor> byKey = tensors.ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal);
                Assert.Equal(3, byKey.Count);
                Tensor weight = byKey["layer.weight"], scale = byKey["layer.weight_scale"], scale2 = byKey["layer.weight_scale_2"];
                Assert.Equal(DType.U8, weight.DType);
                Assert.Equal(DType.F8E4M3, scale.DType);
                Assert.Equal(DType.F32, scale2.DType);

                byte[] packed = weight.AsReadOnlySpan<byte>().ToArray();
                byte[] scales = scale.AsReadOnlySpan<byte>().ToArray();
                float global = scale2.AsReadOnlySpan<float>()[0];
                float[] dec = Decode(packed, scales, global, rows, cols);

                AssertWithinBlockBound(src, dec, rows, cols, global);
            }
            finally
            {
                foreach (SafeTensorsLoader loader in loaders) loader.Dispose();
            }
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>Decodes a row-major NVFP4 matrix through the production dequantizer.</summary>
    private static float[] Decode(byte[] packed, byte[] scales, float global, int rows, int cols)
    {
        using Tensor pk = new(new TensorShape(packed.Length), DType.U8);
        packed.AsSpan().CopyTo(pk.AsSpan<byte>());
        using Tensor sc = new(new TensorShape(rows, cols / 16), DType.F8E4M3);
        scales.AsSpan().CopyTo(sc.AsSpan<byte>());
        using Tensor g = new(new TensorShape(1), DType.F32);
        g.AsSpan<float>()[0] = global;
        QuantRecipe recipe = new()
        {
            Encoding = QuantEncoding.Nvfp4,
            Geometry = new BlockGeometry(1, 16),
            ScaleDType = DType.F8E4M3,
            LogicalRows = rows,
            LogicalCols = cols,
            Scale = sc,
            GlobalScale = g,
        };
        float[] dest = new float[rows * cols];
        ModelOptNvfp4Codec.DequantRows(packed, recipe, 0, rows, dest);
        return dest;
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Quantize_RealTensor_RoundTripsWithinBound()
    {
        string path = Environment.GetEnvironmentVariable("HARTSY_QUANT_TEST_SAFETENSORS") ?? "";
        if (!RealWeightGate.Require(_output.WriteLine, path)) return;

        using SafeTensorsLoader loader = new();
        loader.Load(path);
        SafeTensorDescriptor d = loader.Descriptors.Values.First(x => x.DType == DType.F32 && x.Shape.Rank == 2 && x.Shape[1] % 16 == 0 && x.Shape.ElementCount >= 4096);
        using Tensor w = loader.GetTensor(d.Name);
        int rows = (int)w.Shape[0], cols = (int)w.Shape[1];
        ReadOnlySpan<float> src = w.AsReadOnlySpan<float>();

        (byte[] packed, byte[] scales, float global) = ModelOptNvfp4Codec.Quantize(src, rows, cols);
        float[] dec = Decode(packed, scales, global, rows, cols);
        AssertWithinBlockBound(src.ToArray(), dec, rows, cols, global, d.Name);
    }

    /// <summary>Padding to the 128-row, 4-column tile is accepted; a scale tensor larger than that is a mismatch, and so is
    /// one padded in a single dimension, whose blocked indexing would read past its end (row 1 of [2, 4] is index 16).</summary>
    [Theory]
    [InlineData(128, 4, true)]
    [InlineData(128, 1, true)]
    [InlineData(2, 1, false)]
    [InlineData(2, 4, false)]
    [InlineData(256, 4, false)]
    [InlineData(128, 8, false)]
    public void DequantNvfp4_AcceptsPaddingOnly(int scaleRows, int scaleCols, bool accepted)
    {
        using Tensor packed = new(new TensorShape(2, 8), DType.U8);
        using Tensor blockScales = new(new TensorShape(scaleRows, scaleCols), DType.F8E4M3);
        if (accepted)
        {
            using Tensor f16 = CheckpointConvertUtils.DequantNvfp4ToF16(packed, blockScales, 1f);
            using Tensor fp8 = CheckpointConvertUtils.DequantNvfp4ToFp8(packed, blockScales, 1f);
        }
        else
        {
            Assert.Throws<ArgumentException>(() => CheckpointConvertUtils.DequantNvfp4ToF16(packed, blockScales, 1f));
            Assert.Throws<ArgumentException>(() => CheckpointConvertUtils.DequantNvfp4ToFp8(packed, blockScales, 1f));
        }
    }
}

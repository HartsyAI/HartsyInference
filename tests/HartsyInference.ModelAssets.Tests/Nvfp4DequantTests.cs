using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.SafeTensors;
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
        Tensor packed = new(new TensorShape(1, 8), DType.U8);
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
        Tensor blockScales = new(new TensorShape(1, 1), DType.F8E4M3);
        ((byte*)blockScales.DataPointer)[0] = 0x40;

        Tensor result = CheckpointConvertUtils.DequantNvfp4ToF16(packed, blockScales, globalScale: 0.5f);

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
        // Same 16-element block as the exact-values test, but dequantized to fp8 (block scale folded into value,
        // global scale on Fp8ScaleFactor). fp8_value * Fp8ScaleFactor must reproduce the F16 dequant.
        Tensor packed = new(new TensorShape(1, 8), DType.U8);
        byte* p = (byte*)packed.DataPointer;
        p[0] = Pack(0, 1); p[1] = Pack(2, 3); p[2] = Pack(4, 5); p[3] = Pack(6, 7);
        p[4] = Pack(9, 10); p[5] = Pack(11, 12); p[6] = Pack(13, 14); p[7] = Pack(15, 8);
        Tensor blockScales = new(new TensorShape(1, 1), DType.F8E4M3);
        ((byte*)blockScales.DataPointer)[0] = 0x40; // 2.0
        const float g = 0.5f;

        Tensor f16 = CheckpointConvertUtils.DequantNvfp4ToF16(packed, blockScales, g);
        Tensor fp8 = CheckpointConvertUtils.DequantNvfp4ToFp8(packed, blockScales, g);

        Assert.Equal(DType.F8E4M3, fp8.DType);
        Assert.Equal(g, fp8.Fp8ScaleFactor);
        // fp8 stores e2m1*block_scale (= the F16 value / g); real value = decoded_fp8 * Fp8ScaleFactor.
        Tensor fp8AsF32 = fp8.CastTo(DType.F32);          // CastTo folds Fp8ScaleFactor into the value
        Half* a = (Half*)f16.DataPointer;
        float* b = (float*)fp8AsF32.DataPointer;
        for (int i = 0; i < 16; i++)
            Assert.True(MathF.Abs((float)a[i] - b[i]) < 1e-3f, $"idx {i}: F16={(float)a[i]} fp8*scale={b[i]}");
    }

    [Fact]
    public void DequantNvfp4_BlockScaleSelection_UsesPerBlockScale()
    {
        // Two 16-element blocks in one row: same nibble pattern, different block scales → second block doubled.
        Tensor packed = new(new TensorShape(1, 16), DType.U8);
        byte* p = (byte*)packed.DataPointer;
        for (int j = 0; j < 16; j++) p[j] = Pack(2, 2); // every element = 1.0 pre-scale

        Tensor blockScales = new(new TensorShape(1, 2), DType.F8E4M3);
        byte* s = (byte*)blockScales.DataPointer;
        s[0] = 0x38; // 1.0 (exp=7)
        s[1] = 0x40; // 2.0 (exp=8)

        Tensor result = CheckpointConvertUtils.DequantNvfp4ToF16(packed, blockScales, globalScale: 1f);
        Half* r = (Half*)result.DataPointer;
        for (int i = 0; i < 16; i++) Assert.Equal(1f, (float)r[i]);
        for (int i = 16; i < 32; i++) Assert.Equal(2f, (float)r[i]);
    }

    [Fact]
    public void ApplyFp8ScaledDequant_Nvfp4Group_ReplacedWithF16AndCompanionsDropped()
    {
        Tensor packed = new(new TensorShape(2, 8), DType.U8);
        Tensor blockScales = new(new TensorShape(2, 1), DType.F8E4M3);
        Tensor scale2 = new(new TensorShape(1), DType.F32);
        ((float*)scale2.DataPointer)[0] = 1f;

        Dictionary<string, Tensor> source = new()
        {
            ["blk.weight"] = packed,
            ["blk.weight_scale"] = blockScales,
            ["blk.weight_scale_2"] = scale2,
        };
        Dictionary<string, Tensor> result = CheckpointConvertUtils.ApplyFp8ScaledDequant(source);

        Assert.True(result.ContainsKey("blk.weight"));
        Assert.Equal(DType.F16, result["blk.weight"].DType);
        Assert.Equal(16, result["blk.weight"].Shape[1]);
        Assert.False(result.ContainsKey("blk.weight_scale"));
        Assert.False(result.ContainsKey("blk.weight_scale_2"));
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

        Tensor packed = new(new TensorShape(Rows, PackedCols), DType.U8);
        byte* p = (byte*)packed.DataPointer;
        for (long i = 0; i < packed.ElementCount; i++) p[i] = Pack(2, 4);   // e2m1 1.0 and 2.0

        Tensor blockScales = new(new TensorShape(Rows, BlockCols), DType.F8E4M3);
        byte* s = (byte*)blockScales.DataPointer;
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < BlockCols; c++)
            {
                s[BlockScaleSwizzle.SwizzledIndex(r, c, BlockCols)] = ScaleByte(r, c);
            }
        }

        Tensor result = CheckpointConvertUtils.DequantNvfp4ToF16(packed, blockScales, globalScale: 1f);
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
}

using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Gguf.Codecs;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Tests the codec registry + each newly-added quant codec via hand-built canonical block bytes from ggml-quants.c. Each test produces 32 or 256 elements with a known scale + known quant pattern, then verifies dequant matches the canonical formula.</summary>
public sealed class GgufCodecRegistryTests
{
    [Fact]
    public void Registry_AllExpectedCodecsRegistered()
    {
        Assert.True(GgufCodecRegistry.Supports(DType.Q8_0));
        Assert.True(GgufCodecRegistry.Supports(DType.Q4_K));
        Assert.True(GgufCodecRegistry.Supports(DType.Q5_K));
        Assert.True(GgufCodecRegistry.Supports(DType.Q4_0));
        Assert.True(GgufCodecRegistry.Supports(DType.Q4_1));
        Assert.True(GgufCodecRegistry.Supports(DType.Q5_0));
        Assert.True(GgufCodecRegistry.Supports(DType.Q5_1));
        Assert.True(GgufCodecRegistry.Supports(DType.Q8_1));
        Assert.True(GgufCodecRegistry.Supports(DType.Q2_K));
        Assert.True(GgufCodecRegistry.Supports(DType.Q3_K));
        Assert.True(GgufCodecRegistry.Supports(DType.Q6_K));
        Assert.True(GgufCodecRegistry.Supports(DType.IQ4_NL));
        Assert.True(GgufCodecRegistry.Supports(DType.IQ4_XS));
        Assert.True(GgufCodecRegistry.Supports(DType.MXFP4));
    }

    /// <summary>One hand-built IQ4_XS super-block: sub-block 0 at scale 33 (dl = 1), sub-block 1 at 34 (dl = 2), so the
    /// nibble → codepoint → scale chain shows through at every position checked. A wrong bit split of the 6-bit
    /// scale between scales_l and scales_h moves these values, which is the transcription error this pins.</summary>
    [Fact]
    public unsafe void IQ4_XS_KnownBlock_DequantizesCorrectly()
    {
        using Tensor t = new Tensor(new TensorShape(256), DType.IQ4_XS);
        byte* b = (byte*)t.DataPointer;
        new Span<byte>(b, 136).Clear();
        b[0] = 0x00; b[1] = 0x3C;            // d = 1.0
        b[2] = 0x0A; b[3] = 0x00;            // scales_h: ib0 high bits = 2, ib1 high bits = 2
        b[4] = 0x21;                         // scales_l: ib0 low nibble = 1 (ls 33), ib1 low nibble = 2 (ls 34)
        b[8] = 0x08;                         // sub-block 0, byte 0: low nibble 8 → +1, high nibble 0 → −127
        b[8 + 16] = 0xF0;                    // sub-block 1, byte 0: low nibble 0 → −127, high nibble 15 → +113
        float* dst = stackalloc float[256];
        GgufCodecRegistry.Get(DType.IQ4_XS).DequantizeToF32(b, dst, 256);
        Assert.Equal(1f, dst[0]);
        Assert.Equal(-127f, dst[16]);
        Assert.Equal(-127f, dst[1]);
        Assert.Equal(-254f, dst[32]);
        Assert.Equal(226f, dst[48]);
    }

    [Fact]
    public void Registry_ThrowsOnUnregisteredDtype()
    {
        Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(() => GgufCodecRegistry.Get(DType.TQ1_0));   // ternary, no codec by decision
    }

    [Fact]
    public unsafe void Q4_0_KnownBlock_DequantizesCorrectly()
    {
        Tensor src = new Tensor(new TensorShape(32), DType.Q4_0);
        try
        {
            byte* block = (byte*)src.DataPointer;
            *(Half*)block = (Half)0.5f;
            byte* q = block + 2;
            for (int i = 0; i < 16; i++) q[i] = 0x21;

            using Tensor dst = GgufDequantizer.Dequantize(src, DType.F32);
            float* d = (float*)dst.DataPointer;
            for (int i = 0; i < 16; i++)
            {
                Assert.True(MathF.Abs(d[i] - 0.5f * (1 - 8)) < 1e-3f, $"low nibble at i={i}: expected {0.5f * (1 - 8)}, got {d[i]}");
            }
            for (int i = 16; i < 32; i++)
            {
                Assert.True(MathF.Abs(d[i] - 0.5f * (2 - 8)) < 1e-3f, $"high nibble at i={i}: expected {0.5f * (2 - 8)}, got {d[i]}");
            }
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void Q4_1_KnownBlock_DequantizesCorrectly()
    {
        Tensor src = new Tensor(new TensorShape(32), DType.Q4_1);
        try
        {
            byte* block = (byte*)src.DataPointer;
            *(Half*)block = (Half)0.25f;
            *(Half*)(block + 2) = (Half)1.0f;
            byte* q = block + 4;
            for (int i = 0; i < 16; i++) q[i] = 0x32;

            using Tensor dst = GgufDequantizer.Dequantize(src, DType.F32);
            float* d = (float*)dst.DataPointer;
            for (int i = 0; i < 16; i++)
            {
                Assert.True(MathF.Abs(d[i] - (0.25f * 2 + 1.0f)) < 1e-3f, $"low nibble at i={i}: got {d[i]}");
            }
            for (int i = 16; i < 32; i++)
            {
                Assert.True(MathF.Abs(d[i] - (0.25f * 3 + 1.0f)) < 1e-3f, $"high nibble at i={i}: got {d[i]}");
            }
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void Q5_0_KnownBlock_LowAndHighBitsCombineCorrectly()
    {
        Tensor src = new Tensor(new TensorShape(32), DType.Q5_0);
        try
        {
            byte* block = (byte*)src.DataPointer;
            *(Half*)block = (Half)1.0f;
            *(uint*)(block + 2) = 0xFFFFFFFFu;
            byte* q = block + 6;
            for (int i = 0; i < 16; i++) q[i] = 0x55;

            using Tensor dst = GgufDequantizer.Dequantize(src, DType.F32);
            float* d = (float*)dst.DataPointer;
            for (int i = 0; i < 32; i++)
            {
                Assert.True(MathF.Abs(d[i] - (5 + 16 - 16)) < 1e-3f, $"i={i}: expected 5.0, got {d[i]}");
            }
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void Q5_1_KnownBlock_AddsMin()
    {
        Tensor src = new Tensor(new TensorShape(32), DType.Q5_1);
        try
        {
            byte* block = (byte*)src.DataPointer;
            *(Half*)block = (Half)1.0f;
            *(Half*)(block + 2) = (Half)2.0f;
            *(uint*)(block + 4) = 0x00000000u;
            byte* q = block + 8;
            for (int i = 0; i < 16; i++) q[i] = 0x77;

            using Tensor dst = GgufDequantizer.Dequantize(src, DType.F32);
            float* d = (float*)dst.DataPointer;
            for (int i = 0; i < 32; i++)
            {
                Assert.True(MathF.Abs(d[i] - (1.0f * 7 + 2.0f)) < 1e-3f, $"i={i}: got {d[i]}");
            }
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void Q8_1_KnownBlock_DequantizesIgnoringSumField()
    {
        Tensor src = new Tensor(new TensorShape(32), DType.Q8_1);
        try
        {
            byte* block = (byte*)src.DataPointer;
            *(Half*)block = (Half)0.5f;
            *(Half*)(block + 2) = (Half)999.0f;
            sbyte* q = (sbyte*)(block + 4);
            for (int i = 0; i < 32; i++) q[i] = (sbyte)(i - 16);

            using Tensor dst = GgufDequantizer.Dequantize(src, DType.F32);
            float* d = (float*)dst.DataPointer;
            for (int i = 0; i < 32; i++)
            {
                Assert.True(MathF.Abs(d[i] - 0.5f * (i - 16)) < 1e-3f, $"i={i}: got {d[i]}");
            }
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void Q6_K_KnownBlock_DequantizesUsing6BitQuants()
    {
        Tensor src = new Tensor(new TensorShape(256), DType.Q6_K);
        try
        {
            byte* block = (byte*)src.DataPointer;
            byte* ql = block;
            byte* qh = block + 128;
            sbyte* scales = (sbyte*)(block + 192);
            for (int i = 0; i < 128; i++) ql[i] = 0;
            for (int i = 0; i < 64; i++) qh[i] = 0;
            for (int i = 0; i < 16; i++) scales[i] = 1;
            *(Half*)(block + 208) = (Half)1.0f;

            using Tensor dst = GgufDequantizer.Dequantize(src, DType.F32);
            float* d = (float*)dst.DataPointer;
            for (int i = 0; i < 256; i++)
            {
                Assert.True(MathF.Abs(d[i] - (-32f)) < 1e-3f, $"i={i}: expected -32.0, got {d[i]}");
            }
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void Q2_K_KnownBlock_AllZeroQuantsProduceMin()
    {
        Tensor src = new Tensor(new TensorShape(256), DType.Q2_K);
        try
        {
            byte* block = (byte*)src.DataPointer;
            byte* scales = block;
            byte* qs = block + 16;
            for (int i = 0; i < 16; i++) scales[i] = 0x21;
            for (int i = 0; i < 64; i++) qs[i] = 0;
            *(Half*)(block + 80) = (Half)1.0f;
            *(Half*)(block + 82) = (Half)1.0f;

            using Tensor dst = GgufDequantizer.Dequantize(src, DType.F32);
            float* d = (float*)dst.DataPointer;
            for (int i = 0; i < 16; i++)
            {
                Assert.True(MathF.Abs(d[i] - (-2.0f)) < 1e-3f, $"i={i}: expected -2.0 (= 1*1*0 - 1*2), got {d[i]}");
            }
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void IQ4_NL_KnownBlock_UsesLookupTable()
    {
        Tensor src = new Tensor(new TensorShape(32), DType.IQ4_NL);
        try
        {
            byte* block = (byte*)src.DataPointer;
            *(Half*)block = (Half)1.0f;
            byte* q = block + 2;
            for (int i = 0; i < 16; i++) q[i] = 0x80;

            using Tensor dst = GgufDequantizer.Dequantize(src, DType.F32);
            float* d = (float*)dst.DataPointer;
            for (int i = 0; i < 16; i++)
            {
                Assert.True(MathF.Abs(d[i] - (-127f)) < 1e-3f, $"low nibble at i={i}: expected -127 (KValues[0]), got {d[i]}");
            }
            for (int i = 16; i < 32; i++)
            {
                Assert.True(MathF.Abs(d[i] - 1f) < 1e-3f, $"high nibble at i={i}: expected 1 (KValues[8]), got {d[i]}");
            }
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void MXFP4_KnownBlock_DequantizesUsingE8M0ScaleAndCodewordTable()
    {
        // e=130 -> scale = 2^(130-128) = 4 (ggml_e8m0_to_fp32_half). Nibble pattern 0x21: low nibble=1 ->
        // kvalues_mxfp4[1]=1 -> 1*4=4; high nibble=2 -> kvalues_mxfp4[2]=2 -> 2*4=8.
        Tensor src = new Tensor(new TensorShape(32), DType.MXFP4);
        try
        {
            byte* block = (byte*)src.DataPointer;
            block[0] = 130;
            byte* q = block + 1;
            for (int i = 0; i < 16; i++) q[i] = 0x21;

            using Tensor dst = GgufDequantizer.Dequantize(src, DType.F32);
            float* d = (float*)dst.DataPointer;
            for (int i = 0; i < 16; i++)
            {
                Assert.True(MathF.Abs(d[i] - 4f) < 1e-3f, $"low nibble at i={i}: expected 4, got {d[i]}");
            }
            for (int i = 16; i < 32; i++)
            {
                Assert.True(MathF.Abs(d[i] - 8f) < 1e-3f, $"high nibble at i={i}: expected 8, got {d[i]}");
            }
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void MXFP4_ZeroScale_ProducesZeroRegardlessOfCodeword()
    {
        // e=0 -> scale = 2^-128 (denormal path), not a NaN/zero special case per ggml (NaN e=255 is unhandled
        // upstream too) -- just confirms the denormal branch of E8M0ToFp32Half doesn't blow up or go negative.
        Tensor src = new Tensor(new TensorShape(32), DType.MXFP4);
        try
        {
            byte* block = (byte*)src.DataPointer;
            block[0] = 0;
            byte* q = block + 1;
            for (int i = 0; i < 16; i++) q[i] = 0x71;

            using Tensor dst = GgufDequantizer.Dequantize(src, DType.F32);
            float* d = (float*)dst.DataPointer;
            for (int i = 0; i < 16; i++)
                Assert.True(d[i] >= 0f && d[i] < 1e-30f, $"i={i}: expected ~0, got {d[i]}");
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void Q8_0_QuantizeRoundtrip_PreservesMagnitudes()
    {
        Tensor src = new Tensor(new TensorShape(32), DType.F32);
        Tensor quantized = new Tensor(new TensorShape(32), DType.Q8_0);
        try
        {
            float* sp = (float*)src.DataPointer;
            for (int i = 0; i < 32; i++) sp[i] = i / 16.0f - 1.0f;

            IGgufCodec codec = GgufCodecRegistry.Get(DType.Q8_0);
            Assert.True(codec.SupportsQuantize);
            codec.QuantizeFromF32(sp, (byte*)quantized.DataPointer, 32);

            using Tensor dequantized = GgufDequantizer.Dequantize(quantized, DType.F32);
            float* dp = (float*)dequantized.DataPointer;
            float maxErr = 0f;
            for (int i = 0; i < 32; i++) maxErr = Math.Max(maxErr, MathF.Abs(dp[i] - sp[i]));
            Assert.True(maxErr < 0.01f, $"Q8_0 round-trip max error {maxErr} too large.");
        }
        finally
        {
            src.Dispose();
            quantized.Dispose();
        }
    }

    [Theory]
    [InlineData("Q4_K", 0.05f)]
    [InlineData("Q5_K", 0.025f)]
    [InlineData("Q6_K", 0.01f)]
    public unsafe void K_Quants_QuantizeRoundtrip_WithinExpectedTolerance(string dtypeName, float tolerance)
    {
        DType dtype = dtypeName switch
        {
            "Q4_K" => DType.Q4_K,
            "Q5_K" => DType.Q5_K,
            "Q6_K" => DType.Q6_K,
            _ => throw new ArgumentException(dtypeName),
        };

        Tensor src = new Tensor(new TensorShape(256), DType.F32);
        Tensor quantized = new Tensor(new TensorShape(256), dtype);
        try
        {
            float* sp = (float*)src.DataPointer;
            Random rng = new Random(42);
            for (int i = 0; i < 256; i++) sp[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            IGgufCodec codec = GgufCodecRegistry.Get(dtype);
            Assert.True(codec.SupportsQuantize, $"{dtype} should support quantize.");
            codec.QuantizeFromF32(sp, (byte*)quantized.DataPointer, 256);

            using Tensor dequantized = GgufDequantizer.Dequantize(quantized, DType.F32);
            float* dp = (float*)dequantized.DataPointer;
            float sumSqErr = 0f;
            for (int i = 0; i < 256; i++)
            {
                float err = dp[i] - sp[i];
                sumSqErr += err * err;
            }
            float rmse = MathF.Sqrt(sumSqErr / 256);
            Assert.True(rmse < tolerance, $"{dtype} RMSE {rmse:F4} exceeds tolerance {tolerance:F4}.");
        }
        finally
        {
            src.Dispose();
            quantized.Dispose();
        }
    }

    /// <summary>Q3_K's sixteen 6-bit scales are packed the way ggml's <c>dequantize_row_q3_K</c> reads them: low nibbles
    /// of bytes 0..7 are entries 0..7, high nibbles are entries 8..15, and byte 8 + s % 4 carries entry s's two high bits at
    /// bit 2·(s / 4). Every entry gets a distinct value so a permuted read shows up as the wrong run scaled.</summary>
    [Fact]
    public unsafe void Q3_K_KnownBlock_ScalesUnpackInGgmlOrder()
    {
        byte[] block = new byte[110];
        for (int i = 0; i < 32; i++) block[i] = 0xFF;             // hmask: every high bit set → no −4 offset
        block[0] = 0xFC;                                           // except bits 0 and 1 of byte 0: element 0 of runs 0 (h0 j0) and 2 (h0 j1) gets −4
        for (int i = 32; i < 96; i++) block[i] = 0x55;             // qs: every 2-bit field is 1
        for (int s = 0; s < 16; s++)                               // scale entry s = s − 8 (raw 6-bit s + 24)
        {
            int raw = s + 24;
            if (s < 8) block[96 + s] |= (byte)(raw & 0x0F); else block[96 + s - 8] |= (byte)((raw & 0x0F) << 4);
            block[96 + 8 + (s & 3)] |= (byte)((raw >> 4) << (2 * (s >> 2)));
        }
        block[108] = 0x00; block[109] = 0x3C;                      // d = 1.0
        float[] dst = new float[256];
        fixed (byte* src = block) fixed (float* d = dst)
            GgufCodecRegistry.Get(DType.Q3_K).DequantizeToF32(src, d, 256);
        for (int s = 0; s < 16; s++) Assert.Equal(s - 8, dst[16 * s + 1]);   // run s, element 1: scale × 1
        Assert.Equal(24f, dst[0]);                                            // run 0, element 0: (−8) × (1 − 4)
        Assert.Equal(18f, dst[32]);                                           // run 2 (h0 j1 half0): hmask byte 0, bit 1
        Assert.Equal(-7f, dst[16]);                                           // run 1 (h0 j0 half1): hmask byte 16, untouched
    }

    /// <summary>Each i-quant format from a hand-built block: codebook index 0 everywhere (a known grid entry), zero scale
    /// fields (a known multiplier), and exactly one sign or shift field set, so a misread of any field lands on a
    /// distinct element. Values follow ggml's <c>dequantize_row_iq*</c> with d = 1.</summary>
    [Theory]
    [InlineData("IQ2_XXS", 66, -1.0f, 1, 1.0f)]    // grid[0] = 8s, db = 0.125; ksigns[1] flips elements 0 and 7
    [InlineData("IQ2_XS", 74, -1.0f, 1, 1.0f)]
    [InlineData("IQ2_S", 82, 1.0f, 5, -1.0f)]      // explicit sign byte 0x20 flips element 5
    [InlineData("IQ3_XXS", 98, -1.0f, 1, 1.0f)]    // grid[0] = 4s, db = 0.25
    [InlineData("IQ3_S", 110, 1.0f, 5, -1.0f)]     // grid[0] = 1s, db = 1
    [InlineData("IQ1_S", 50, -0.875f, 32, -1.125f)] // grid[0] = −1s; group 1 carries the −1/8 shift
    [InlineData("IQ1_M", 56, -1.125f, 8, -0.875f)]  // half 0 of group 0 carries the −1/8 shift, half 1 does not
    public unsafe void IQ_KnownBlock_DecodesEachField(string name, int blockBytes, float first, int probe, float probed)
    {
        DType dtype = name switch
        {
            "IQ2_XXS" => DType.IQ2_XXS, "IQ2_XS" => DType.IQ2_XS, "IQ2_S" => DType.IQ2_S, "IQ3_XXS" => DType.IQ3_XXS,
            "IQ3_S" => DType.IQ3_S, "IQ1_S" => DType.IQ1_S, "IQ1_M" => DType.IQ1_M, _ => throw new ArgumentOutOfRangeException(nameof(name)),
        };
        Assert.Equal(blockBytes, dtype.BlockByteSize);
        byte[] block = new byte[blockBytes];
        switch (name)
        {
            case "IQ2_XXS": block[0] = 0x00; block[1] = 0x3C; block[6] = 1; break;          // aux1 of group 0 = 1: sign pattern 1, scale 0
            case "IQ2_XS": block[0] = 0x00; block[1] = 0x3C; block[3] = 0x02; break;        // word 0 = 512: grid 0, sign pattern 1
            case "IQ2_S": block[0] = 0x00; block[1] = 0x3C; block[34] = 0x20; break;        // signs[0] bit 5
            case "IQ3_XXS": block[0] = 0x00; block[1] = 0x3C; block[66] = 1; break;         // scales-and-signs word 0 = 1
            case "IQ3_S": block[0] = 0x00; block[1] = 0x3C; block[74] = 0x20; break;        // signs[0] bit 5
            case "IQ1_S": block[0] = 0x00; block[1] = 0x3C; block[37] = 0x80; break;        // qh[1] bit 15: group 1 shifts by −1/8
            case "IQ1_M": block[53] = 0xC0; block[55] = 0x30; block[32] = 0x08; break;      // d = 1.0 (0x3C00 spread over the nibbles); qh[0] bit 3
        }
        float[] dst = new float[256];
        fixed (byte* src = block) fixed (float* d = dst)
            GgufCodecRegistry.Get(dtype).DequantizeToF32(src, d, 256);
        Assert.Equal(first, dst[0]);
        Assert.Equal(probed, dst[probe]);
    }
}

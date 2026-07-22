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
        Assert.True(GgufCodecRegistry.Supports(DType.MXFP4));
    }

    [Fact]
    public void Registry_ThrowsOnUnregisteredDtype()
    {
        Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(() => GgufCodecRegistry.Get(DType.IQ2_XXS));
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
}

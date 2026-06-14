using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.Gguf;
using Xunit;

namespace HartsyInference.ModelHandler.Tests;

/// <summary>Unit tests for K-quant dequantization. Builds canonical block bytes by hand using the ggml `block_q4_K` / `block_q5_K` layouts and verifies <see cref="GgufDequantizer.Dequantize"/> round-trips correctly. Reference: <c>ggml-quants.c</c> in llama.cpp.</summary>
public sealed class GgufDequantizerTests
{
    [Fact]
    public unsafe void Q8_0_RoundTrip_Identity()
    {
        const int blockSize = 32;
        TensorShape shape = new TensorShape(blockSize);
        Tensor src = new Tensor(shape, DType.Q8_0);
        try
        {
            byte* p = (byte*)src.DataPointer;
            Half scale = (Half)0.5f;
            *(Half*)p = scale;
            for (int i = 0; i < blockSize; i++) p[2 + i] = (byte)(sbyte)(i - 16);
            using Tensor dst = GgufDequantizer.Dequantize(src, DType.F32);
            float* d = (float*)dst.DataPointer;
            for (int i = 0; i < blockSize; i++)
            {
                float expected = 0.5f * (i - 16);
                Assert.True(MathF.Abs(d[i] - expected) < 1e-3f, $"i={i}: expected {expected}, got {d[i]}");
            }
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void Q4K_GetScaleMin_AllSubBlocks_Within6BitRange()
    {
        // Build a 12-byte scales array with known patterns and verify each sub-block extracts (sc, m) ∈ [0..63].
        Tensor src = new Tensor(new TensorShape(256), DType.Q4_K);
        try
        {
            byte* block = (byte*)src.DataPointer;
            // d = 1, dmin = 0 — so result for each element = sc * q.
            *(Half*)block = (Half)1.0f;
            *(Half*)(block + 2) = (Half)0.0f;
            byte* scales = block + 4;
            // Make every sub-block scale = sc_j (0..7) and min = mm_j (8..15) — packed via canonical layout.
            // Sub-blocks 0..3 use q[j] (low 6 bits) and q[j+4] (low 6 bits).
            for (int j = 0; j < 4; j++)
            {
                scales[j] = (byte)(j & 63);
                scales[j + 4] = (byte)((j + 8) & 63);
            }
            // Sub-blocks 4..7 use q[j+4] (split) and q[j-4] high bits.
            for (int j = 4; j < 8; j++)
            {
                int sc = j;       // target scale 4..7
                int mm = j + 8;   // target min   12..15
                int low4Sc = sc & 0xF;
                int low4Mm = mm & 0xF;
                int hi2Sc = (sc >> 4) & 0x3;
                int hi2Mm = (mm >> 4) & 0x3;
                // q[j+4] holds low4Sc | (low4Mm << 4)
                scales[j + 4] = (byte)(low4Sc | (low4Mm << 4));
                // hi2Sc goes into top 2 bits of q[j-4]; hi2Mm into top 2 bits of q[j-0]
                scales[j - 4] |= (byte)(hi2Sc << 6);
                scales[j] |= (byte)(hi2Mm << 6);
            }

            byte* quantData = block + 16;
            for (int i = 0; i < 128; i++) quantData[i] = 0x11;

            using Tensor dst = GgufDequantizer.Dequantize(src, DType.F32);
            float* d = (float*)dst.DataPointer;

            for (int j = 0; j < 8; j++)
            {
                float observed = d[j * 32];
                Assert.True(observed > -1e-6f, $"sub-block {j}: dequantized value {observed} should be ≥ 0 (d=1 dmin=0).");
                Assert.True(observed < 64f, $"sub-block {j}: dequantized value {observed} suspiciously large.");
            }
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void Q4K_KnownBlock_DequantizesToExpectedValues()
    {
        // Construct a block where d=2.0, dmin=0.5, scale_0 = 3, min_0 = 1, all quants in sub-block 0 = 5.
        // Layout: 32 bytes for sub-block pair (0, 1) — low nibble = sub-block 0, high nibble = sub-block 1.
        // We set every byte to 0x05 so sub-block 0 reads 5 and sub-block 1 reads 0.
        // Expected: x = d * sc_0 * q - dmin * mm_0 = 2 * 3 * 5 - 0.5 * 1 = 30 - 0.5 = 29.5
        Tensor src = new Tensor(new TensorShape(256), DType.Q4_K);
        try
        {
            byte* block = (byte*)src.DataPointer;
            *(Half*)block = (Half)2.0f;
            *(Half*)(block + 2) = (Half)0.5f;
            byte* scales = block + 4;
            for (int i = 0; i < 12; i++) scales[i] = 0;
            scales[0] = 3;
            scales[4] = 1;
            byte* quantData = block + 16;
            for (int i = 0; i < 32; i++) quantData[i] = 0x05;

            using Tensor dst = GgufDequantizer.Dequantize(src, DType.F32);
            float* d = (float*)dst.DataPointer;
            for (int i = 0; i < 32; i++)
            {
                Assert.True(MathF.Abs(d[i] - 29.5f) < 1e-3f, $"i={i}: expected 29.5, got {d[i]}");
            }
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void Q5K_KnownBlock_DequantizesUsingBothLowAndHighBits()
    {
        // d=1.0, dmin=0, scale_0=1, min_0=0; low nibbles all 5 (0b0101), high bits all 1.
        // Expected: q = low(5) | (high(1) << 4) = 5 | 16 = 21
        // x = 1 * 1 * 21 - 0 = 21
        Tensor src = new Tensor(new TensorShape(256), DType.Q5_K);
        try
        {
            byte* block = (byte*)src.DataPointer;
            *(Half*)block = (Half)1.0f;
            *(Half*)(block + 2) = (Half)0.0f;
            byte* scales = block + 4;
            for (int i = 0; i < 12; i++) scales[i] = 0;
            scales[0] = 1;          // sub-block 0 scale
            scales[4] = 0;          // sub-block 0 min
            byte* highBits = block + 16;
            for (int i = 0; i < 32; i++) highBits[i] = 0x01;
            byte* lowBits = block + 48;
            for (int i = 0; i < 32; i++) lowBits[i] = 0x05;

            using Tensor dst = GgufDequantizer.Dequantize(src, DType.F32);
            float* d = (float*)dst.DataPointer;
            for (int i = 0; i < 32; i++)
            {
                Assert.True(MathF.Abs(d[i] - 21f) < 1e-3f, $"i={i}: expected 21.0, got {d[i]}");
            }
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void Q4K_F16_Output_Within_F16_Tolerance()
    {
        // Same block as Q4K_KnownBlock; F16 result tolerance.
        Tensor src = new Tensor(new TensorShape(256), DType.Q4_K);
        try
        {
            byte* block = (byte*)src.DataPointer;
            *(Half*)block = (Half)2.0f;
            *(Half*)(block + 2) = (Half)0.5f;
            byte* scales = block + 4;
            for (int i = 0; i < 12; i++) scales[i] = 0;
            scales[0] = 3;
            scales[4] = 1;
            byte* quantData = block + 16;
            for (int i = 0; i < 32; i++) quantData[i] = 0x05;

            using Tensor dst = GgufDequantizer.Dequantize(src, DType.F16);
            Half* d = (Half*)dst.DataPointer;
            for (int i = 0; i < 32; i++)
            {
                float observed = (float)d[i];
                Assert.True(MathF.Abs(observed - 29.5f) < 1e-1f, $"i={i}: expected 29.5, got {observed}");
            }
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public void Dequantize_NonQuantizedSource_Throws()
    {
        Tensor src = new Tensor(new TensorShape(8), DType.F32);
        try { Assert.Throws<ArgumentException>(() => GgufDequantizer.Dequantize(src, DType.F32)); }
        finally { src.Dispose(); }
    }

    [Fact]
    public void Dequantize_InvalidTargetDtype_Throws()
    {
        Tensor src = new Tensor(new TensorShape(32), DType.Q8_0);
        try { Assert.Throws<ArgumentException>(() => GgufDequantizer.Dequantize(src, DType.BF16)); }
        finally { src.Dispose(); }
    }
}

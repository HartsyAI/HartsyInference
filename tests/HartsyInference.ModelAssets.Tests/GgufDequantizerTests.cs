using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Unit tests for K-quant dequantization. Builds canonical block bytes by hand using the ggml `block_q4_K` / `block_q5_K` layouts and verifies <see cref="GgufDequantizer.Dequantize"/> round-trips correctly. Reference: <c>ggml-quants.c</c> in llama.cpp.</summary>
public sealed class GgufDequantizerTests
{
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
    public void Dequantize_NonQuantizedSource_Throws()
    {
        Tensor src = new Tensor(new TensorShape(8), DType.F32);
        try { Assert.Throws<ArgumentException>(() => GgufDequantizer.Dequantize(src, DType.F32)); }
        finally { src.Dispose(); }
    }
}

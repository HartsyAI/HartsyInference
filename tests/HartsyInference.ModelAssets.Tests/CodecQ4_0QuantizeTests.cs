using HartsyInference.ModelAssets.Gguf.Codecs;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Q4_0 quantization follows ggml's reference rule: the block's signed largest-magnitude value lands exactly on −8, so the
/// codes span <c>[−8d, 7d]</c>, and every value is within half a step of its clamp into that span.</summary>
public sealed unsafe class CodecQ4_0QuantizeTests
{
    [Fact]
    public void Quantize_KeepsEveryValueWithinHalfAStep_AndTheExtremeExact()
    {
        const int blocks = 64;
        float[] values = new float[blocks * 32];
        Random random = new(12);
        for (int i = 0; i < values.Length; i++) values[i] = (random.NextSingle() - 0.5f) * (1 + i / 32);
        byte[] packed = new byte[blocks * 18];
        float[] decoded = new float[values.Length];
        Codec_Q4_0 codec = new();
        fixed (float* src = values)
        fixed (byte* q = packed)
        fixed (float* dst = decoded)
        {
            codec.QuantizeFromF32(src, q, values.Length);
            codec.DequantizeToF32(q, dst, values.Length);
        }
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> block = values.AsSpan(b * 32, 32);
            int extreme = 0;
            for (int i = 1; i < 32; i++) if (MathF.Abs(block[i]) > MathF.Abs(block[extreme])) extreme = i;
            float d = (float)(Half)(block[extreme] / -8f), step = MathF.Abs(d);
            float low = MathF.Min(-8 * d, 7 * d), high = MathF.Max(-8 * d, 7 * d);
            for (int i = 0; i < 32; i++)
            {
                float target = Math.Clamp(block[i], low, high);
                Assert.True(MathF.Abs(decoded[b * 32 + i] - target) <= step * 0.5f + 1e-3f * step + 1e-6f,
                    $"block {b} value {i}: {block[i]} decoded {decoded[b * 32 + i]}, step {step}");
            }
            Assert.Equal(block[extreme], decoded[b * 32 + extreme], 2e-3f * MathF.Abs(block[extreme]));
        }
    }
}

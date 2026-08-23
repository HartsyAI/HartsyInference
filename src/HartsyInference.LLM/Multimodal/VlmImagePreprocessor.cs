using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Multimodal;

/// <summary>Turns a raw decoded RGB image into the normalized pixel tensor a VLM vision tower expects: bilinear-resize to a square <c>size×size</c>, scale to [0,1], then per-channel <c>(x - mean) / std</c>; decoding (PNG/JPG → RGB bytes) is the caller's responsibility, keeping the LLM package dependency-free.</summary>
public static class VlmImagePreprocessor
{
    /// <summary>Resizes interleaved <paramref name="rgb"/> (<c>H*W*3</c> bytes, row-major HWC) to <c>[1, 3, size, size]</c> and normalizes with <paramref name="mean"/>/<paramref name="std"/> (length 3).</summary>
    public static unsafe Tensor Preprocess(ReadOnlySpan<byte> rgb, int width, int height, int size, float[] mean, float[] std)
    {
        if (rgb.Length < (long)width * height * 3)
            throw new ArgumentException($"rgb has {rgb.Length} bytes, expected {(long)width * height * 3} for {width}x{height}.", nameof(rgb));

        Tensor outp = new(new TensorShape(1, 3, size, size), DType.F32);
        float* dst = (float*)outp.DataPointer;
        float sx = (float)width / size, sy = (float)height / size;
        fixed (byte* src = rgb)
        {
            for (int c = 0; c < 3; c++)
            {
                float m = mean[c], s = std[c];
                float* chan = dst + (long)c * size * size;
                for (int oy = 0; oy < size; oy++)
                {
                    // Bilinear sample centre of each output pixel.
                    float fy = (oy + 0.5f) * sy - 0.5f;
                    int y0 = (int)MathF.Floor(fy); float wy = fy - y0;
                    int y0c = Math.Clamp(y0, 0, height - 1), y1c = Math.Clamp(y0 + 1, 0, height - 1);
                    for (int ox = 0; ox < size; ox++)
                    {
                        float fx = (ox + 0.5f) * sx - 0.5f;
                        int x0 = (int)MathF.Floor(fx); float wx = fx - x0;
                        int x0c = Math.Clamp(x0, 0, width - 1), x1c = Math.Clamp(x0 + 1, 0, width - 1);
                        float p00 = src[((long)y0c * width + x0c) * 3 + c];
                        float p01 = src[((long)y0c * width + x1c) * 3 + c];
                        float p10 = src[((long)y1c * width + x0c) * 3 + c];
                        float p11 = src[((long)y1c * width + x1c) * 3 + c];
                        float top = p00 + (p01 - p00) * wx;
                        float bot = p10 + (p11 - p10) * wx;
                        float v = top + (bot - top) * wy;
                        chan[(long)oy * size + ox] = (v / 255f - m) / s;
                    }
                }
            }
        }
        return outp;
    }
}

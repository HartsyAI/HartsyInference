using HartsyInference.Core.Imaging;
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

    /// <summary>The checkpoint processor's smart resize: <see cref="SmartResize.Qwen2Vl"/> (the Core port of transformers' <c>smart_resize</c>), with this
    /// layer's refusal of a non-positive factor or a bound that cannot hold one block.</summary>
    public static (int Height, int Width) SmartResize(int height, int width, int factor, int minPixels, int maxPixels)
    {
        if (factor <= 0 || height <= 0 || width <= 0)
            throw new ArgumentException($"Image {width}x{height} and factor {factor} must be positive.");
        if (maxPixels < minPixels || maxPixels < (long)factor * factor)
            throw new ArgumentException($"Pixel bounds [{minPixels}, {maxPixels}] must hold at least one {factor}x{factor} block, with the maximum at or above the minimum.");
        return HartsyInference.Core.Imaging.SmartResize.Qwen2Vl(height, width, factor, minPixels, maxPixels);
    }

    /// <summary>The checkpoint's pixel path: <see cref="SmartResize"/> to the aspect-preserving size, the Core PIL-matching bicubic resample
    /// (<see cref="Resample.BicubicHwc8"/>, antialiased, rounded to 8 bits after each pass), then <c>(x/255 - mean) / std</c>. Returns <c>[1, 3, H', W']</c>.
    /// Unlike <see cref="Preprocess"/> the output is not square, so a caller reads the grid from the returned shape.</summary>
    public static unsafe Tensor PreprocessAspect(ReadOnlySpan<byte> rgb, int width, int height, int factor, int minPixels, int maxPixels, float[] mean, float[] std)
    {
        int need = checked(width * height * 3);
        if (rgb.Length < need)
            throw new ArgumentException($"rgb has {rgb.Length} bytes, expected {need} for {width}x{height}.", nameof(rgb));
        (int outH, int outW) = SmartResize(height, width, factor, minPixels, maxPixels);
        float[] hwc = new float[(long)outH * outW * 3];
        Resample.BicubicHwc8(rgb[..need], width, height, 3, hwc, outW, outH, -0.5f, antialias: true, quantizeU8: true);

        Tensor outp = new(new TensorShape(1, 3, outH, outW), DType.F32);
        float* dst = (float*)outp.DataPointer;
        for (int c = 0; c < 3; c++)
        {
            float m = mean[c], s = std[c];
            float* chan = dst + (long)c * outH * outW;
            for (long i = 0; i < (long)outH * outW; i++)
                chan[i] = (hwc[i * 3 + c] / 255f - m) / s;
        }
        return outp;
    }
}

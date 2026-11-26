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

    /// <summary>The checkpoint processor's smart resize (transformers <c>smart_resize</c>, Qwen2-VL family): both sides rounded to whole multiples of
    /// <paramref name="factor"/>, then scaled to keep the pixel count inside <c>[minPixels, maxPixels]</c> while keeping the aspect ratio as near as the
    /// rounding allows. Python's <c>round</c> rounds half to even, which <see cref="Math.Round(double)"/> does too.</summary>
    public static (int Height, int Width) SmartResize(int height, int width, int factor, int minPixels, int maxPixels)
    {
        if (factor <= 0 || height <= 0 || width <= 0)
            throw new ArgumentException($"Image {width}x{height} and factor {factor} must be positive.");
        if (Math.Max(height, width) / (double)Math.Min(height, width) > 200)
            throw new ArgumentException($"Image {width}x{height} has an aspect ratio above 200:1; the processor refuses it.");
        int hBar = (int)Math.Round(height / (double)factor) * factor;
        int wBar = (int)Math.Round(width / (double)factor) * factor;
        if ((long)hBar * wBar > maxPixels)
        {
            double beta = Math.Sqrt((double)height * width / maxPixels);
            hBar = Math.Max(factor, (int)Math.Floor(height / beta / factor) * factor);
            wBar = Math.Max(factor, (int)Math.Floor(width / beta / factor) * factor);
        }
        else if ((long)hBar * wBar < minPixels)
        {
            double beta = Math.Sqrt((double)minPixels / ((double)height * width));
            hBar = (int)Math.Ceiling(height * beta / factor) * factor;
            wBar = (int)Math.Ceiling(width * beta / factor) * factor;
        }
        return (hBar, wBar);
    }

    /// <summary>The checkpoint's pixel path: <see cref="SmartResize"/> to the aspect-preserving size, a bicubic resample (PIL's <c>BICUBIC</c>
    /// kernel, a = -0.5, with the antialias support PIL uses when shrinking), then <c>(x/255 - mean) / std</c>. Returns <c>[1, 3, H', W']</c>.
    /// Unlike <see cref="Preprocess"/> the output is not square, so a caller reads the grid from the returned shape.</summary>
    public static unsafe Tensor PreprocessAspect(ReadOnlySpan<byte> rgb, int width, int height, int factor, int minPixels, int maxPixels, float[] mean, float[] std)
    {
        if (rgb.Length < (long)width * height * 3)
            throw new ArgumentException($"rgb has {rgb.Length} bytes, expected {(long)width * height * 3} for {width}x{height}.", nameof(rgb));
        (int outH, int outW) = SmartResize(height, width, factor, minPixels, maxPixels);
        Taps[] hx = BicubicTaps(width, outW), vy = BicubicTaps(height, outH);

        Tensor outp = new(new TensorShape(1, 3, outH, outW), DType.F32);
        float* dst = (float*)outp.DataPointer;
        float[] rows = new float[(long)height * outW];   // horizontally resampled rows, one channel at a time
        fixed (byte* src = rgb)
        {
            for (int c = 0; c < 3; c++)
            {
                for (int y = 0; y < height; y++)
                {
                    byte* row = src + (long)y * width * 3 + c;
                    for (int ox = 0; ox < outW; ox++)
                    {
                        Taps t = hx[ox];
                        float acc = 0f;
                        for (int k = 0; k < t.Weights.Length; k++) acc += t.Weights[k] * row[(long)(t.Start + k) * 3];
                        rows[(long)y * outW + ox] = Clip8(acc);   // PIL keeps the intermediate pass in 8 bits
                    }
                }
                float m = mean[c], s = std[c];
                float* chan = dst + (long)c * outH * outW;
                for (int oy = 0; oy < outH; oy++)
                {
                    Taps t = vy[oy];
                    for (int ox = 0; ox < outW; ox++)
                    {
                        float acc = 0f;
                        for (int k = 0; k < t.Weights.Length; k++) acc += t.Weights[k] * rows[(long)(t.Start + k) * outW + ox];
                        chan[(long)oy * outW + ox] = (Clip8(acc) / 255f - m) / s;
                    }
                }
            }
        }
        return outp;
    }

    /// <summary>PIL's <c>clip8</c>: round half up, then clamp to a byte.</summary>
    private static float Clip8(float v) => MathF.Min(255f, MathF.Max(0f, MathF.Floor(v + 0.5f)));

    /// <summary>One output pixel's source window: the first input index and the normalized filter weights over it.</summary>
    private readonly record struct Taps(int Start, float[] Weights);

    /// <summary>PIL's precomputed coefficients for a bicubic resample of <paramref name="inSize"/> to <paramref name="outSize"/> (its
    /// <c>precompute_coeffs</c>): the filter widens by the scale when shrinking, the window is clipped to the image, weights sum to one.</summary>
    private static Taps[] BicubicTaps(int inSize, int outSize)
    {
        const double Support = 2.0;
        double scale = (double)inSize / outSize;
        double filterScale = Math.Max(scale, 1.0);
        double support = Support * filterScale;
        double ss = 1.0 / filterScale;
        Taps[] taps = new Taps[outSize];
        for (int xx = 0; xx < outSize; xx++)
        {
            double center = (xx + 0.5) * scale;
            int xmin = Math.Max((int)(center - support + 0.5), 0);
            int xmax = Math.Min((int)(center + support + 0.5), inSize) - xmin;
            double[] w = new double[xmax];
            double sum = 0;
            for (int k = 0; k < xmax; k++)
            {
                w[k] = BicubicFilter((k + xmin - center + 0.5) * ss);
                sum += w[k];
            }
            float[] weights = new float[xmax];
            for (int k = 0; k < xmax; k++) weights[k] = sum != 0 ? (float)(w[k] / sum) : 0f;
            taps[xx] = new Taps(xmin, weights);
        }
        return taps;
    }

    private static double BicubicFilter(double x)
    {
        const double a = -0.5;
        if (x < 0) x = -x;
        if (x < 1) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
        if (x < 2) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
        return 0.0;
    }
}

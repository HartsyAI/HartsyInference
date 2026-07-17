namespace HartsyInference.Vision.Codec;

/// <summary>Separable Keys-bicubic resampling with optional antialiasing (kernel support scaled by the
/// downscale factor — a proper low-pass prefilter; plain 4-tap bicubic aliases on downscales). Half-pixel
/// centers (<c>align_corners=False</c>); border taps are clipped to the source range and the remaining
/// weights renormalized (the PIL convention — NOT edge-replication). With <c>a = -0.5</c> and
/// <c>antialias = true</c> this bit-matches PIL <c>BICUBIC</c> / HF image processors / torch
/// <c>F.interpolate(mode="bicubic", antialias=True)</c> (torch's antialias path uses the PIL A = −0.5
/// kernel for up- AND downscales, unlike its non-AA A = −0.75 kernel). Host code: runs once per image in
/// pre/post-processing, never on the per-step hot path.</summary>
public static class Resample
{
    /// <summary>Resizes one row-major single-channel F32 plane.</summary>
    public static void BicubicPlane(ReadOnlySpan<float> src, int srcW, int srcH,
        Span<float> dst, int dstW, int dstH, float a, bool antialias)
    {
        ValidateArgs(src.Length, srcW, srcH, dst.Length, dstW, dstH, channels: 1);
        (int[] xStart, int[] xCount, float[] xWeights, int xTaps) = BuildWeights(srcW, dstW, a, antialias);
        (int[] yStart, int[] yCount, float[] yWeights, int yTaps) = BuildWeights(srcH, dstH, a, antialias);

        float[] tmp = new float[(long)srcH * dstW];
        for (int y = 0; y < srcH; y++)
        {
            ReadOnlySpan<float> row = src.Slice(y * srcW, srcW);
            for (int x = 0; x < dstW; x++)
            {
                float acc = 0f;
                int lo = xStart[x];
                for (int t = 0; t < xCount[x]; t++)
                    acc += xWeights[x * xTaps + t] * row[lo + t];
                tmp[(long)y * dstW + x] = acc;
            }
        }
        for (int y = 0; y < dstH; y++)
        {
            int lo = yStart[y];
            for (int x = 0; x < dstW; x++)
            {
                float acc = 0f;
                for (int t = 0; t < yCount[y]; t++)
                    acc += yWeights[y * yTaps + t] * tmp[(long)(lo + t) * dstW + x];
                dst[y * dstW + x] = acc;
            }
        }
    }

    /// <summary>Resizes interleaved (HWC) 8-bit pixels into interleaved F32 (still 0–255 scale; the caller
    /// applies its own normalization). <paramref name="quantizeU8"/> rounds + clamps each separable pass to
    /// the 0–255 grid the way PIL / torchvision resize 8-bit images (their horizontal pass materializes a
    /// uint8 intermediate); enable it when parity with an HF image processor matters — SigLIP patch tokens
    /// are sensitive enough that skipping it costs measurable conditioning correlation.</summary>
    public static void BicubicHwc8(ReadOnlySpan<byte> src, int srcW, int srcH, int channels,
        Span<float> dst, int dstW, int dstH, float a, bool antialias, bool quantizeU8 = false)
    {
        ValidateArgs(src.Length, srcW, srcH, dst.Length, dstW, dstH, channels);
        (int[] xStart, int[] xCount, float[] xWeights, int xTaps) = BuildWeights(srcW, dstW, a, antialias);
        (int[] yStart, int[] yCount, float[] yWeights, int yTaps) = BuildWeights(srcH, dstH, a, antialias);

        float[] tmp = new float[(long)srcH * dstW * channels];
        for (int y = 0; y < srcH; y++)
        {
            ReadOnlySpan<byte> row = src.Slice(y * srcW * channels, srcW * channels);
            for (int x = 0; x < dstW; x++)
            {
                int lo = xStart[x];
                for (int c = 0; c < channels; c++)
                {
                    float acc = 0f;
                    for (int t = 0; t < xCount[x]; t++)
                        acc += xWeights[x * xTaps + t] * row[(lo + t) * channels + c];
                    tmp[((long)y * dstW + x) * channels + c] = quantizeU8 ? QuantU8(acc) : acc;
                }
            }
        }
        for (int y = 0; y < dstH; y++)
        {
            int lo = yStart[y];
            for (int x = 0; x < dstW; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    float acc = 0f;
                    for (int t = 0; t < yCount[y]; t++)
                        acc += yWeights[y * yTaps + t] * tmp[((long)(lo + t) * dstW + x) * channels + c];
                    dst[(y * dstW + x) * channels + c] = quantizeU8 ? QuantU8(acc) : acc;
                }
            }
        }
    }

    /// <summary>Round-to-nearest onto the 0–255 grid (PIL's <c>clip8</c>).</summary>
    private static float QuantU8(float v) => Math.Clamp(MathF.Round(v), 0f, 255f);

    /// <summary>Per-destination-index filter taps: clipped start index, in-range tap count, and normalized
    /// kernel weights. With antialiasing the kernel is stretched by the downscale factor
    /// (<c>filterScale = max(src/dst, 1)</c>) so it acts as a proper low-pass prefilter. Border windows are
    /// clipped to <c>[0, srcLen)</c> and renormalized over the surviving taps (PIL semantics).</summary>
    private static (int[] Starts, int[] Counts, float[] Weights, int Taps) BuildWeights(int srcLen, int dstLen, float a, bool antialias)
    {
        double scale = (double)srcLen / dstLen;
        double filterScale = antialias ? Math.Max(scale, 1.0) : 1.0;
        double support = 2.0 * filterScale;
        int taps = (int)Math.Ceiling(support) * 2 + 1;
        int[] starts = new int[dstLen];
        int[] counts = new int[dstLen];
        float[] weights = new float[(long)dstLen * taps];

        for (int i = 0; i < dstLen; i++)
        {
            double center = (i + 0.5) * scale;
            int lo = Math.Max((int)Math.Floor(center - support + 0.5), 0);
            int hi = Math.Min((int)Math.Floor(center + support + 0.5), srcLen);
            starts[i] = lo;
            counts[i] = hi - lo;
            double sum = 0.0;
            for (int t = 0; t < hi - lo; t++)
            {
                double w = CubicKernel((lo + t + 0.5 - center) / filterScale, a);
                weights[i * taps + t] = (float)w;
                sum += w;
            }
            if (sum != 0.0)
            {
                float inv = (float)(1.0 / sum);
                for (int t = 0; t < hi - lo; t++) weights[i * taps + t] *= inv;
            }
        }
        return (starts, counts, weights, taps);
    }

    /// <summary>Keys cubic kernel with sharpness parameter <paramref name="a"/> (zero outside |x| ≥ 2).</summary>
    private static double CubicKernel(double x, double a)
    {
        x = Math.Abs(x);
        if (x < 1.0) return (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0;
        if (x < 2.0) return a * (x * x * x - 5.0 * x * x + 8.0 * x - 4.0);
        return 0.0;
    }

    private static void ValidateArgs(int srcLen, int srcW, int srcH, int dstLen, int dstW, int dstH, int channels)
    {
        if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0 || channels <= 0)
            throw new ArgumentException($"Resample dimensions must be positive; got src {srcW}x{srcH}, dst {dstW}x{dstH}, channels {channels}.");
        if (srcLen != (long)srcW * srcH * channels)
            throw new ArgumentException($"Source length {srcLen} != {srcW}x{srcH}x{channels}.");
        if (dstLen != (long)dstW * dstH * channels)
            throw new ArgumentException($"Destination length {dstLen} != {dstW}x{dstH}x{channels}.");
    }
}

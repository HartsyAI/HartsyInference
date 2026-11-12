namespace HartsyInference.Core.Imaging;

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
    // PIL Resample.c: 32 − 8 − 2 fractional bits, so a sum of 8-bit pixels times unit-sum weights fits an int.
    private const int PilPrecisionBits = 32 - 8 - 2;
    private const int PilRounding = 1 << (PilPrecisionBits - 1);

    /// <summary>Resizes one row-major single-channel F32 plane.</summary>
    public static void BicubicPlane(ReadOnlySpan<float> src, int srcW, int srcH,
        Span<float> dst, int dstW, int dstH, float a, bool antialias)
    {
        ValidateArgs(src.Length, srcW, srcH, dst.Length, dstW, dstH, channels: 1);
        SeparablePlane(src, srcW, srcH, dst, dstW, dstH, BuildWeights(srcW, dstW, a, antialias), BuildWeights(srcH, dstH, a, antialias));
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
        SeparableHwc8(src, srcW, srcH, channels, dst, dstW, dstH,
            BuildWeights(srcW, dstW, a, antialias), BuildWeights(srcH, dstH, a, antialias), quantizeU8);
    }

    /// <summary>Separable triangle (bilinear) resize of interleaved 8-bit pixels with PIL <c>BILINEAR</c> semantics,
    /// which is what torchvision <c>Resize</c> applies to PIL images: antialiased (kernel stretched by the downscale
    /// factor) when <paramref name="antialias"/> is set, half-pixel centers, border taps clipped and renormalized.
    /// <paramref name="quantizeU8"/> rounds each pass to the 0–255 grid as PIL does.</summary>
    public static void BilinearHwc8(ReadOnlySpan<byte> src, int srcW, int srcH, int channels,
        Span<float> dst, int dstW, int dstH, bool antialias, bool quantizeU8 = false)
    {
        ValidateArgs(src.Length, srcW, srcH, dst.Length, dstW, dstH, channels);
        SeparableHwc8(src, srcW, srcH, channels, dst, dstW, dstH,
            BuildWeights(srcW, dstW, 1.0, TriangleKernel, antialias), BuildWeights(srcH, dstH, 1.0, TriangleKernel, antialias), quantizeU8);
    }

    /// <summary>Bilinear resize of one row-major F32 plane with torch's <c>align_corners=False</c> sampling and no
    /// antialiasing, as <c>F.interpolate(mode="bilinear")</c> computes it on CPU.</summary>
    public static void BilinearPlaneF32(ReadOnlySpan<float> src, int srcW, int srcH, Span<float> dst, int dstW, int dstH)
    {
        ValidateArgs(src.Length, srcW, srcH, dst.Length, dstW, dstH, channels: 1);
        (int[] x0, int[] x1, float[] xl) = BuildTorchBilinearIndices(srcW, dstW);
        (int[] y0, int[] y1, float[] yl) = BuildTorchBilinearIndices(srcH, dstH);
        for (int y = 0; y < dstH; y++)
        {
            int r0 = y0[y] * srcW, r1 = y1[y] * srcW;
            float ly = yl[y], hy = 1f - ly;
            for (int x = 0; x < dstW; x++)
            {
                float lx = xl[x], hx = 1f - lx;
                float top = hx * src[r0 + x0[x]] + lx * src[r0 + x1[x]];
                float bottom = hx * src[r1 + x0[x]] + lx * src[r1 + x1[x]];
                dst[y * dstW + x] = hy * top + ly * bottom;
            }
        }
    }

    /// <summary>Bilinear resize of one row-major F32 plane with the triangle filter widened by the downscale factor
    /// (<c>F.interpolate(mode="bilinear", antialias=True)</c>); enlarging is plain half-pixel bilinear.</summary>
    public static void BilinearPlaneAntialias(ReadOnlySpan<float> src, int srcW, int srcH, Span<float> dst, int dstW, int dstH)
    {
        ValidateArgs(src.Length, srcW, srcH, dst.Length, dstW, dstH, channels: 1);
        SeparablePlane(src, srcW, srcH, dst, dstW, dstH,
            BuildWeights(srcW, dstW, 1.0, TriangleKernel, antialias: true), BuildWeights(srcH, dstH, 1.0, TriangleKernel, antialias: true));
    }

    /// <summary>The two separable passes (horizontal, then vertical) of a single-channel F32 plane over precomputed filter taps.</summary>
    private static void SeparablePlane(ReadOnlySpan<float> src, int srcW, int srcH, Span<float> dst, int dstW, int dstH,
        (int[] Starts, int[] Counts, float[] Weights, int Taps) horizontal, (int[] Starts, int[] Counts, float[] Weights, int Taps) vertical)
    {
        float[] tmp = new float[(long)srcH * dstW];
        for (int y = 0; y < srcH; y++)
        {
            ReadOnlySpan<float> row = src.Slice(y * srcW, srcW);
            for (int x = 0; x < dstW; x++)
            {
                float acc = 0f;
                int lo = horizontal.Starts[x];
                for (int t = 0; t < horizontal.Counts[x]; t++)
                    acc += horizontal.Weights[x * horizontal.Taps + t] * row[lo + t];
                tmp[(long)y * dstW + x] = acc;
            }
        }

        for (int y = 0; y < dstH; y++)
        {
            int lo = vertical.Starts[y];
            for (int x = 0; x < dstW; x++)
            {
                float acc = 0f;
                for (int t = 0; t < vertical.Counts[y]; t++)
                    acc += vertical.Weights[y * vertical.Taps + t] * tmp[(long)(lo + t) * dstW + x];
                dst[y * dstW + x] = acc;
            }
        }
    }

    private static (int[] Lo, int[] Hi, float[] Lambda) BuildTorchBilinearIndices(int srcLen, int dstLen)
    {
        double scale = (double)srcLen / dstLen;
        int[] lo = new int[dstLen];
        int[] hi = new int[dstLen];
        float[] lambda = new float[dstLen];
        for (int i = 0; i < dstLen; i++)
        {
            double src = scale * (i + 0.5) - 0.5;
            if (src < 0.0) src = 0.0;
            int l = (int)Math.Floor(src);
            if (l > srcLen - 1) l = srcLen - 1;
            lo[i] = l;
            hi[i] = l < srcLen - 1 ? l + 1 : l;
            lambda[i] = (float)(src - l);
        }
        return (lo, hi, lambda);
    }

    /// <summary>Separable radius-3 Lanczos resize for interleaved 8-bit pixels. The tap geometry follows PIL's
    /// half-pixel-centred resize; border taps are clipped and renormalized.</summary>
    public static void LanczosHwc8(ReadOnlySpan<byte> src, int srcW, int srcH, int channels,
        Span<float> dst, int dstW, int dstH)
    {
        ValidateArgs(src.Length, srcW, srcH, dst.Length, dstW, dstH, channels);
        (int[] xStart, int[] xCount, float[] xWeights, int xTaps) = BuildLanczosWeights(srcW, dstW);
        (int[] yStart, int[] yCount, float[] yWeights, int yTaps) = BuildLanczosWeights(srcH, dstH);
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
                    tmp[((long)y * dstW + x) * channels + c] = acc;
                }
            }
        }
        for (int y = 0; y < dstH; y++)
        {
            int lo = yStart[y];
            for (int x = 0; x < dstW; x++)
                for (int c = 0; c < channels; c++)
                {
                    float acc = 0f;
                    for (int t = 0; t < yCount[y]; t++)
                        acc += yWeights[y * yTaps + t] * tmp[((long)(lo + t) * dstW + x) * channels + c];
                    dst[(y * dstW + x) * channels + c] = acc;
                }
        }
    }

    /// <summary>PIL's <c>Image.resize(..., LANCZOS)</c> for interleaved 8-bit pixels, reproduced in its integer arithmetic:
    /// double-precision taps normalized per output pixel, converted to 22-bit fixed point, a horizontal pass rounded and clipped
    /// to 8 bits, then the vertical pass. A pass whose size does not change is skipped, as PIL skips it.</summary>
    public static void LanczosPil8(ReadOnlySpan<byte> src, int srcW, int srcH, int channels, Span<byte> dst, int dstW, int dstH)
    {
        ValidateArgs(src.Length, srcW, srcH, dst.Length, dstW, dstH, channels);
        byte[]? horizontal = null;
        ReadOnlySpan<byte> rows = src;
        if (dstW != srcW)
        {
            (int[] xStart, int[] xCount, int[] xWeights, int xTaps) = BuildPilLanczosWeights(srcW, dstW);
            horizontal = new byte[srcH * dstW * channels];
            for (int y = 0; y < srcH; y++)
            {
                ReadOnlySpan<byte> row = src.Slice(y * srcW * channels, srcW * channels);
                for (int x = 0; x < dstW; x++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        int acc = PilRounding;
                        for (int t = 0; t < xCount[x]; t++)
                            acc += row[(xStart[x] + t) * channels + c] * xWeights[x * xTaps + t];
                        horizontal[(y * dstW + x) * channels + c] = PilClip8(acc);
                    }
                }
            }
            rows = horizontal;
        }
        if (dstH == srcH)
        {
            rows.CopyTo(dst);
            return;
        }
        (int[] yStart, int[] yCount, int[] yWeights, int yTaps) = BuildPilLanczosWeights(srcH, dstH);
        for (int y = 0; y < dstH; y++)
        {
            for (int x = 0; x < dstW; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    int acc = PilRounding;
                    for (int t = 0; t < yCount[y]; t++)
                        acc += rows[((yStart[y] + t) * dstW + x) * channels + c] * yWeights[y * yTaps + t];
                    dst[(y * dstW + x) * channels + c] = PilClip8(acc);
                }
            }
        }
    }

    /// <summary>Two-pass separable filter over interleaved 8-bit pixels: horizontal into an F32 intermediate, then vertical.</summary>
    private static void SeparableHwc8(ReadOnlySpan<byte> src, int srcW, int srcH, int channels, Span<float> dst, int dstW, int dstH,
        (int[] Starts, int[] Counts, float[] Weights, int Taps) xw, (int[] Starts, int[] Counts, float[] Weights, int Taps) yw, bool quantizeU8)
    {
        (int[] xStart, int[] xCount, float[] xWeights, int xTaps) = xw;
        (int[] yStart, int[] yCount, float[] yWeights, int yTaps) = yw;
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

    /// <summary>Per-destination-index filter taps for the bicubic kernel (support radius 2).</summary>
    private static (int[] Starts, int[] Counts, float[] Weights, int Taps) BuildWeights(int srcLen, int dstLen, float a, bool antialias) =>
        BuildWeights(srcLen, dstLen, 2.0, x => CubicKernel(x, a), antialias);

    /// <summary>Per-destination-index filter taps: clipped start index, in-range tap count, and normalized
    /// kernel weights. With antialiasing the kernel is stretched by the downscale factor
    /// (<c>filterScale = max(src/dst, 1)</c>) so it acts as a proper low-pass prefilter. Border windows are
    /// clipped to <c>[0, srcLen)</c> and renormalized over the surviving taps (PIL semantics).</summary>
    private static (int[] Starts, int[] Counts, float[] Weights, int Taps) BuildWeights(
        int srcLen, int dstLen, double supportRadius, Func<double, double> kernel, bool antialias)
    {
        double scale = (double)srcLen / dstLen;
        double filterScale = antialias ? Math.Max(scale, 1.0) : 1.0;
        double support = supportRadius * filterScale;
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
                double w = kernel((lo + t + 0.5 - center) / filterScale);
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

    private static (int[] Starts, int[] Counts, float[] Weights, int Taps) BuildLanczosWeights(int srcLen, int dstLen)
    {
        double scale = (double)srcLen / dstLen;
        double filterScale = Math.Max(scale, 1.0);
        double support = 3.0 * filterScale;
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
                double w = LanczosKernel((lo + t + 0.5 - center) / filterScale);
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

    /// <summary>PIL <c>precompute_coeffs</c> plus <c>normalize_coeffs_8bpc</c> for its Lanczos filter: taps in double, normalized by
    /// their sum, then scaled to <see cref="PilPrecisionBits"/>-bit fixed point rounding half away from zero.</summary>
    private static (int[] Starts, int[] Counts, int[] Weights, int Taps) BuildPilLanczosWeights(int srcLen, int dstLen)
    {
        double scale = (double)srcLen / dstLen;
        double filterScale = Math.Max(scale, 1.0);
        double support = 3.0 * filterScale;
        int taps = (int)Math.Ceiling(support) * 2 + 1;
        int[] starts = new int[dstLen];
        int[] counts = new int[dstLen];
        int[] weights = new int[dstLen * taps];
        double[] k = new double[taps];
        double inverseScale = 1.0 / filterScale;
        for (int i = 0; i < dstLen; i++)
        {
            double center = (i + 0.5) * scale;
            int lo = Math.Max((int)(center - support + 0.5), 0);
            int count = Math.Min((int)(center + support + 0.5), srcLen) - lo;
            double sum = 0.0;
            for (int t = 0; t < count; t++)
            {
                k[t] = PilLanczos((t + lo - center + 0.5) * inverseScale);
                sum += k[t];
            }
            for (int t = 0; t < count; t++)
            {
                double w = sum != 0.0 ? k[t] / sum : k[t];
                weights[i * taps + t] = (int)(w < 0 ? -0.5 + w * (1 << PilPrecisionBits) : 0.5 + w * (1 << PilPrecisionBits));
            }
            starts[i] = lo;
            counts[i] = count;
        }
        return (starts, counts, weights, taps);
    }

    private static double PilLanczos(double x) => x >= -3.0 && x < 3.0 ? PilSinc(x) * PilSinc(x / 3.0) : 0.0;

    private static double PilSinc(double x)
    {
        if (x == 0.0) return 1.0;
        x *= Math.PI;
        return Math.Sin(x) / x;
    }

    /// <summary>PIL <c>clip8</c>: the fixed-point sum shifted down (an arithmetic shift, so negative sums floor) and clamped.</summary>
    private static byte PilClip8(int acc) => (byte)Math.Clamp(acc >> PilPrecisionBits, 0, 255);

    private static double LanczosKernel(double x)
    {
        double ax = Math.Abs(x);
        if (ax < 1e-12) return 1.0;
        if (ax >= 3.0) return 0.0;
        double pix = Math.PI * x;
        double pixOver3 = pix / 3.0;
        return (Math.Sin(pix) / pix) * (Math.Sin(pixOver3) / pixOver3);
    }

    /// <summary>Keys cubic kernel with sharpness parameter <paramref name="a"/> (zero outside |x| ≥ 2).</summary>
    private static double CubicKernel(double x, double a)
    {
        x = Math.Abs(x);
        if (x < 1.0) return (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0;
        if (x < 2.0) return a * (x * x * x - 5.0 * x * x + 8.0 * x - 4.0);
        return 0.0;
    }

    /// <summary>Triangle (bilinear) kernel (zero outside |x| ≥ 1).</summary>
    private static double TriangleKernel(double x) => Math.Max(0.0, 1.0 - Math.Abs(x));

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

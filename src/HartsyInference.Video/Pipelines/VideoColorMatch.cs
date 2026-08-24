namespace HartsyInference.Video.Pipelines;

/// <summary>Per-channel mean/std colour matching in Lab (D65) — the drift correction chunked character-animation
/// pipelines (kijai's Wan wrapper, Wan2GP) apply to every continuation chunk: each decoded frame is matched to the
/// STATIC reference image's statistics, so colour error cannot compound through the carried-frame chain. Operates on
/// interleaved RGB24 because that is what the recipe chunk loops traffic in; <c>strength</c> lerps between the
/// original and the fully-matched frame, and <c>&lt;= 0</c> is a byte-level no-op.</summary>
/// <remarks>Compute reference stats over content pixels only — never a letterboxed canvas. Black pad bars drag the
/// mean toward zero, and darkening is precisely this correction's community-documented failure mode.</remarks>
public static class VideoColorMatch
{
    /// <summary>Per-channel Lab mean/std of an image, the match target for <see cref="MatchToReference"/>.</summary>
    public readonly record struct LabStats(double MeanL, double MeanA, double MeanB, double StdL, double StdA, double StdB);

    /// <summary>A flatter channel than this matches by mean shift alone — scaling noise-floor variance up to the reference's would amplify banding into visible noise.</summary>
    private const double MinStd = 1e-4;

    private static readonly double[] SrgbToLinear = BuildSrgbToLinearLut();

    /// <summary>Per-channel Lab mean and standard deviation of an interleaved RGB24 image.</summary>
    public static LabStats ComputeStats(ReadOnlySpan<byte> rgb24, int width, int height)
    {
        ValidateBuffer(rgb24.Length, width, height);
        long count = (long)width * height;
        double sumL = 0, sumA = 0, sumB = 0, sqL = 0, sqA = 0, sqB = 0;
        for (long i = 0; i < count; i++)
        {
            (double l, double a, double b) = ToLab(rgb24[(int)(i * 3)], rgb24[(int)(i * 3 + 1)], rgb24[(int)(i * 3 + 2)]);
            sumL += l; sumA += a; sumB += b;
            sqL += l * l; sqA += a * a; sqB += b * b;
        }
        double meanL = sumL / count, meanA = sumA / count, meanB = sumB / count;
        return new LabStats(meanL, meanA, meanB, Math.Sqrt(Math.Max(0, sqL / count - meanL * meanL)),
            Math.Sqrt(Math.Max(0, sqA / count - meanA * meanA)), Math.Sqrt(Math.Max(0, sqB / count - meanB * meanB)));
    }

    /// <summary>Matches the frame's per-channel Lab mean/std to <paramref name="reference"/> in place, lerped by
    /// <paramref name="strength"/> (1 = fully matched, &lt;= 0 = untouched).</summary>
    public static void MatchToReference(byte[] frameRgb24, int width, int height, in LabStats reference, float strength)
    {
        ArgumentNullException.ThrowIfNull(frameRgb24);
        ValidateBuffer(frameRgb24.Length, width, height);
        if (strength <= 0f)
        {
            return;
        }
        LabStats own = ComputeStats(frameRgb24, width, height);
        double scaleL = own.StdL > MinStd ? reference.StdL / own.StdL : 1.0;
        double scaleA = own.StdA > MinStd ? reference.StdA / own.StdA : 1.0;
        double scaleB = own.StdB > MinStd ? reference.StdB / own.StdB : 1.0;
        double mix = Math.Min(strength, 1f);
        long count = (long)width * height;
        for (long i = 0; i < count; i++)
        {
            int o = (int)(i * 3);
            (double l, double a, double b) = ToLab(frameRgb24[o], frameRgb24[o + 1], frameRgb24[o + 2]);
            l += mix * ((l - own.MeanL) * scaleL + reference.MeanL - l);
            a += mix * ((a - own.MeanA) * scaleA + reference.MeanA - a);
            b += mix * ((b - own.MeanB) * scaleB + reference.MeanB - b);
            (frameRgb24[o], frameRgb24[o + 1], frameRgb24[o + 2]) = FromLab(l, a, b);
        }
    }

    private static void ValidateBuffer(int length, int width, int height)
    {
        if (width <= 0 || height <= 0 || length < (long)width * height * 3)
        {
            throw new ArgumentException($"RGB24 buffer has {length} bytes, expected {(long)width * height * 3} for {width}x{height}.");
        }
    }

    // D65 two-degree white point, sRGB primaries; the standard CIE Lab pair of transforms.
    private const double Xn = 0.95047, Yn = 1.0, Zn = 1.08883;
    private const double Delta = 6.0 / 29.0;

    private static (double L, double A, double B) ToLab(byte r8, byte g8, byte b8)
    {
        double r = SrgbToLinear[r8], g = SrgbToLinear[g8], b = SrgbToLinear[b8];
        double fx = LabF((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / Xn);
        double fy = LabF((0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / Yn);
        double fz = LabF((0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / Zn);
        return (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz));
    }

    private static (byte R, byte G, byte B) FromLab(double l, double a, double b)
    {
        double fy = (l + 16.0) / 116.0;
        double x = Xn * LabFInv(fy + a / 500.0);
        double y = Yn * LabFInv(fy);
        double z = Zn * LabFInv(fy - b / 200.0);
        return (LinearToSrgb(3.2404542 * x - 1.5371385 * y - 0.4985314 * z),
            LinearToSrgb(-0.9692660 * x + 1.8760108 * y + 0.0415560 * z),
            LinearToSrgb(0.0556434 * x - 0.2040259 * y + 1.0572252 * z));
    }

    private static double LabF(double t) => t > Delta * Delta * Delta ? Math.Cbrt(t) : t / (3.0 * Delta * Delta) + 4.0 / 29.0;

    private static double LabFInv(double f) => f > Delta ? f * f * f : 3.0 * Delta * Delta * (f - 4.0 / 29.0);

    private static double[] BuildSrgbToLinearLut()
    {
        double[] lut = new double[256];
        for (int i = 0; i < 256; i++)
        {
            double c = i / 255.0;
            lut[i] = c <= 0.04045 ? c / 12.92 : Math.Pow((c + 0.055) / 1.055, 2.4);
        }
        return lut;
    }

    private static byte LinearToSrgb(double linear)
    {
        // Clamp BEFORE the gamma pow: out-of-gamut matched Lab values go negative here, and pow of a negative is NaN.
        double c = Math.Clamp(linear, 0.0, 1.0);
        double srgb = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.Pow(c, 1.0 / 2.4) - 0.055;
        return (byte)Math.Clamp((int)Math.Round(srgb * 255.0), 0, 255);
    }
}

namespace HartsyInference.Video.Pipelines;

/// <summary>SeedVR2 colour matching back to the resized input, the same three transfers as ComfyUI's
/// <c>SeedVR2PostProcessing</c> (<c>comfy/ldm/seedvr/color_fix.py</c>). <see cref="Transplant"/> is the StableSR-style
/// wavelet low-frequency lerp, the optional <c>color_fix.py</c> the SeedVR2 reference loads IF PRESENT (absent by default).
/// <see cref="TransferLab"/> and <see cref="TransferAdaIn"/> are the ComfyUI <c>lab</c> and <c>adain</c> modes.
/// All planes are C×H×W floats in [-1,1].</summary>
public static class SeedVr2ColorFix
{
    private const int Levels = 5;
    private const float LabLuminanceWeight = 0.8f;
    private const float AdaInEpsilon = 1e-5f;

    // CIE 15 constants and D65 white point, as in comfy/ldm/seedvr/constants.py.
    private const double CielabDelta = 6.0 / 29.0;
    private const double CielabKappa = 24389.0 / 27.0;
    private const double D65WhiteX = 0.95047;
    private const double D65WhiteZ = 1.08883;

    private static readonly double[,] RgbToXyz =
    {
        { 0.4124564, 0.3575761, 0.1804375 },
        { 0.2126729, 0.7151522, 0.0721750 },
        { 0.0193339, 0.1191920, 0.9503041 },
    };

    private static readonly double[,] XyzToRgb =
    {
        { 3.2404542, -1.5371385, -0.4985314 },
        { -0.9692660, 1.8760108, 0.0415560 },
        { 0.0556434, -0.2040259, 1.0572252 },
    };

    /// <summary>Blends per-plane: <c>model + (1−strength)·(inputLow − modelLow)</c>. Both buffers are
    /// C×H×W planes in [-1,1]; <paramref name="model"/> is modified in place.</summary>
    public static void Transplant(Span<float> model, ReadOnlySpan<float> input, int channels, int h, int w, float strength)
    {
        if (strength >= 1f)
            return;
        float mix = 1f - strength;
        float[] modelLow = new float[h * w];
        float[] inputLow = new float[h * w];
        float[] scratch = new float[h * w];
        for (int c = 0; c < channels; c++)
        {
            Span<float> plane = model.Slice(c * h * w, h * w);
            LowFrequency(plane, modelLow, scratch, h, w);
            LowFrequency(input.Slice(c * h * w, h * w), inputLow, scratch, h, w);
            for (int i = 0; i < plane.Length; i++)
                plane[i] += mix * (inputLow[i] - modelLow[i]);
        }
    }

    /// <summary>ComfyUI <c>lab</c>: wavelet-recombine the model's high frequencies with the input's low frequencies,
    /// then histogram-match the CIELAB a/b channels to the input and blend L 80/20 toward its histogram. RGB in [-1,1]
    /// (3 planes); <paramref name="model"/> is overwritten with the result.</summary>
    public static void TransferLab(Span<float> model, ReadOnlySpan<float> input, int h, int w)
    {
        int n = h * w;
        float[] content = new float[3 * n];
        float[] scratch = new float[n];
        float[] modelLow = new float[n];
        float[] inputLow = new float[n];
        for (int c = 0; c < 3; c++)
        {
            Span<float> plane = model.Slice(c * n, n);
            LowFrequency(plane, modelLow, scratch, h, w);
            LowFrequency(input.Slice(c * n, n), inputLow, scratch, h, w);
            for (int i = 0; i < n; i++)
                content[c * n + i] = Math.Clamp(plane[i] - modelLow[i] + inputLow[i], -1f, 1f);
        }

        float[] contentLab = RgbToLab(content, n);
        float[] referenceLab = RgbToLab(input, n);
        float[] matchedL = new float[n];
        float[] matchedA = new float[n];
        float[] matchedB = new float[n];
        MatchHistogram(contentLab.AsSpan(n, n), referenceLab.AsSpan(n, n), matchedA);
        MatchHistogram(contentLab.AsSpan(2 * n, n), referenceLab.AsSpan(2 * n, n), matchedB);
        MatchHistogram(contentLab.AsSpan(0, n), referenceLab.AsSpan(0, n), matchedL);
        for (int i = 0; i < n; i++)
            matchedL[i] = contentLab[i] * LabLuminanceWeight + matchedL[i] * (1f - LabLuminanceWeight);
        float[] resultLab = new float[3 * n];
        matchedL.CopyTo(resultLab, 0);
        matchedA.CopyTo(resultLab, n);
        matchedB.CopyTo(resultLab, 2 * n);
        LabToRgb(resultLab, n, model);
    }

    /// <summary>ComfyUI <c>adain</c>: per frame and channel, match the model's mean and standard deviation to the input's
    /// (ε 1e-5, population variance), then clamp to [-1,1]. <paramref name="model"/> is overwritten in place.</summary>
    public static void TransferAdaIn(Span<float> model, ReadOnlySpan<float> input, int channels, int h, int w)
    {
        int n = h * w;
        for (int c = 0; c < channels; c++)
        {
            Span<float> plane = model.Slice(c * n, n);
            ReadOnlySpan<float> reference = input.Slice(c * n, n);
            MeanStd(plane, out float contentMean, out float contentStd);
            MeanStd(reference, out float styleMean, out float styleStd);
            for (int i = 0; i < n; i++)
                plane[i] = Math.Clamp((plane[i] - contentMean) / contentStd * styleStd + styleMean, -1f, 1f);
        }
    }

    private static void MeanStd(ReadOnlySpan<float> plane, out float mean, out float std)
    {
        double sum = 0;
        foreach (float v in plane)
            sum += v;
        double m = sum / plane.Length;
        double sq = 0;
        foreach (float v in plane)
            sq += (v - m) * (v - m);
        mean = (float)m;
        std = (float)Math.Sqrt(sq / plane.Length + AdaInEpsilon);
    }

    /// <summary>Histogram match with equal sizes: the k-th smallest source value takes the k-th smallest reference value.</summary>
    private static void MatchHistogram(ReadOnlySpan<float> source, ReadOnlySpan<float> reference, Span<float> dst)
    {
        int n = source.Length;
        int[] order = new int[n];
        for (int i = 0; i < n; i++)
            order[i] = i;
        float[] sourceCopy = source.ToArray();
        Array.Sort(order, (a, b) =>
        {
            int cmp = sourceCopy[a].CompareTo(sourceCopy[b]);
            return cmp != 0 ? cmp : a.CompareTo(b);
        });
        float[] referenceSorted = reference.ToArray();
        Array.Sort(referenceSorted);
        for (int k = 0; k < n; k++)
            dst[order[k]] = referenceSorted[k];
    }

    /// <summary>Planar sRGB in [-1,1] to planar CIELAB, via [0,1] clamped as ComfyUI does before the transform.</summary>
    private static float[] RgbToLab(ReadOnlySpan<float> rgb, int n)
    {
        float[] lab = new float[3 * n];
        for (int i = 0; i < n; i++)
        {
            double[] lin = new double[3];
            for (int c = 0; c < 3; c++)
            {
                double v = Math.Clamp((rgb[c * n + i] + 1.0) * 0.5, 0.0, 1.0);
                lin[c] = v > 0.04045 ? Math.Pow((v + 0.055) / 1.055, 2.4) : v / 12.92;
            }
            double x = RgbToXyz[0, 0] * lin[0] + RgbToXyz[0, 1] * lin[1] + RgbToXyz[0, 2] * lin[2];
            double y = RgbToXyz[1, 0] * lin[0] + RgbToXyz[1, 1] * lin[1] + RgbToXyz[1, 2] * lin[2];
            double z = RgbToXyz[2, 0] * lin[0] + RgbToXyz[2, 1] * lin[1] + RgbToXyz[2, 2] * lin[2];
            x /= D65WhiteX;
            z /= D65WhiteZ;
            double fx = Cbrt(x), fy = Cbrt(y), fz = Cbrt(z);
            lab[i] = (float)(116.0 * fy - 16.0);
            lab[n + i] = (float)(500.0 * (fx - fy));
            lab[2 * n + i] = (float)(200.0 * (fy - fz));
        }
        return lab;
    }

    /// <summary>Planar CIELAB to RGB in [-1,1], written to <paramref name="rgb"/>.</summary>
    private static void LabToRgb(ReadOnlySpan<float> lab, int n, Span<float> rgb)
    {
        for (int i = 0; i < n; i++)
        {
            double L = lab[i], a = lab[n + i], b = lab[2 * n + i];
            double fy = (L + 16.0) / 116.0;
            double fx = a / 500.0 + fy;
            double fz = fy - b / 200.0;
            double x = fx > CielabDelta ? fx * fx * fx : (116.0 * fx - 16.0) / CielabKappa;
            double y = fy > CielabDelta ? fy * fy * fy : (116.0 * fy - 16.0) / CielabKappa;
            double z = fz > CielabDelta ? fz * fz * fz : (116.0 * fz - 16.0) / CielabKappa;
            x *= D65WhiteX;
            z *= D65WhiteZ;
            for (int c = 0; c < 3; c++)
            {
                double linear = XyzToRgb[c, 0] * x + XyzToRgb[c, 1] * y + XyzToRgb[c, 2] * z;
                double g = linear > 0.0031308 ? 1.055 * Math.Pow(Math.Max(linear, 0.0), 1.0 / 2.4) - 0.055 : linear * 12.92;
                rgb[c * n + i] = (float)(Math.Clamp(g, 0.0, 1.0) * 2.0 - 1.0);
            }
        }
    }

    private static double Cbrt(double t) => t > CielabDelta * CielabDelta * CielabDelta
        ? Math.Cbrt(t)
        : (CielabKappa * t + 16.0) / 116.0;

    private static void LowFrequency(ReadOnlySpan<float> src, float[] dst, float[] scratch, int h, int w)
    {
        src.CopyTo(dst);
        // ComfyUI caps the dilation at min(H,W)//8 (wavelet_blur's max_safe_radius), so small planes keep their detail.
        int maxSafeRadius = Math.Max(1, Math.Min(h, w) / 8);
        for (int level = 0; level < Levels; level++)
        {
            WaveletBlur(dst, scratch, h, w, Math.Min(1 << level, maxSafeRadius));
            scratch.CopyTo(dst, 0);
        }
    }

    /// <summary>3×3 [[1,2,1],[2,4,2],[1,2,1]]/16 blur at the given dilation, replicate-padded.</summary>
    private static void WaveletBlur(ReadOnlySpan<float> src, float[] dst, int h, int w, int dilation)
    {
        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            float acc = 0;
            for (int ky = -1; ky <= 1; ky++)
            for (int kx = -1; kx <= 1; kx++)
            {
                int sy = Math.Clamp(y + ky * dilation, 0, h - 1);
                int sx = Math.Clamp(x + kx * dilation, 0, w - 1);
                float weight = (ky == 0 ? 2f : 1f) * (kx == 0 ? 2f : 1f) / 16f;
                acc += src[sy * w + sx] * weight;
            }
            dst[y * w + x] = acc;
        }
    }
}

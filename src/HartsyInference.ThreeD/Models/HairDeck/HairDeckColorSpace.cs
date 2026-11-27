namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>CIELAB (D65, 2 degree observer) to linear sRGB. The reference's <c>skimage.color.lab2rgb</c> applies the sRGB
/// transfer curve and clips; clipping the linear values gives the same result, so the output is linear and clipped to 0..1.</summary>
public static class HairDeckColorSpace
{
    private const double WhiteX = 0.95047;
    private const double WhiteY = 1.0;
    private const double WhiteZ = 1.08883;
    private const double Delta = 6.0 / 29.0;

    /// <summary>Converts <paramref name="lab"/> (L, a, b per colour) to linear RGB in <paramref name="linearRgb"/>.</summary>
    public static void LabToLinearRgb(ReadOnlySpan<float> lab, Span<float> linearRgb)
    {
        if (lab.Length % 3 != 0 || linearRgb.Length != lab.Length)
            throw new ArgumentException("Lab and RGB spans must hold the same number of triplets.");
        for (int i = 0; i < lab.Length; i += 3)
        {
            double fy = (lab[i] + 16.0) / 116.0;
            double fx = fy + lab[i + 1] / 500.0;
            double fz = fy - lab[i + 2] / 200.0;
            double x = WhiteX * Inverse(fx);
            double y = WhiteY * Inverse(fy);
            double z = WhiteZ * Inverse(fz);
            linearRgb[i] = Clip(3.2404542 * x - 1.5371385 * y - 0.4985314 * z);
            linearRgb[i + 1] = Clip(-0.9692660 * x + 1.8760108 * y + 0.0415560 * z);
            linearRgb[i + 2] = Clip(0.0556434 * x - 0.2040259 * y + 1.0572252 * z);
        }
    }

    private static double Inverse(double f) => f > Delta ? f * f * f : 3.0 * Delta * Delta * (f - 4.0 / 29.0);

    private static float Clip(double v) => (float)Math.Clamp(v, 0.0, 1.0);
}

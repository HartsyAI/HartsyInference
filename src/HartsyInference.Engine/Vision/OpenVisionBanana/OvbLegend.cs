namespace HartsyInference.Engine.Vision.OpenVisionBanana;

/// <summary>Colour draws for a segmentation legend, after the release's <c>p200_randomnegdraw</c> mode (the one the evaluation
/// code probes the model with): shuffle the 200-colour palette, take colours in that order that are at least
/// <see cref="MinDeltaE"/> apart in CIELAB from every colour already chosen, then relax with the skipped ones if the legend is
/// larger than the separated set. Python's shuffle is not reproduced: the model was trained on random draws, so any
/// seeded draw is in distribution, and a fixed seed keeps a run repeatable.</summary>
public static class OvbLegend
{
    /// <summary>The release's separation floor between two legend colours (CIE76 delta E).</summary>
    public const double MinDeltaE = 25.0;

    private static readonly double[] Lab = BuildLab();

    /// <summary>The colours for <paramref name="count"/> classes, in class order.</summary>
    public static (byte R, byte G, byte B)[] Draw(int count, int seed, double minDeltaE = MinDeltaE)
    {
        if (count < 1 || count > OvbPalette.Count)
            throw new ArgumentOutOfRangeException(nameof(count), count, $"A legend holds 1 to {OvbPalette.Count} classes.");
        int[] order = new int[OvbPalette.Count];
        for (int i = 0; i < order.Length; i++) order[i] = i;
        SplitMix rng = new((ulong)(uint)seed);
        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = (int)(rng.Next() % (ulong)(i + 1));
            (order[i], order[j]) = (order[j], order[i]);
        }
        List<int> chosen = new(count);
        List<int> skipped = new();
        double floor = minDeltaE * minDeltaE;
        foreach (int candidate in order)
        {
            bool separated = true;
            foreach (int taken in chosen)
            {
                if (SquaredDistance(candidate, taken) < floor)
                {
                    separated = false;
                    break;
                }
            }
            if (separated)
            {
                chosen.Add(candidate);
                if (chosen.Count == count)
                    break;
            }
            else
            {
                skipped.Add(candidate);
            }
        }
        for (int i = 0; chosen.Count < count && i < skipped.Count; i++)
            chosen.Add(skipped[i]);
        (byte, byte, byte)[] colors = new (byte, byte, byte)[count];
        ReadOnlySpan<byte> rgb = OvbPalette.Rgb;
        for (int i = 0; i < count; i++)
            colors[i] = (rgb[chosen[i] * 3], rgb[chosen[i] * 3 + 1], rgb[chosen[i] * 3 + 2]);
        return colors;
    }

    /// <summary>A legend for <paramref name="classes"/>, drawn with <paramref name="seed"/>.</summary>
    public static IReadOnlyList<OvbLegendEntry> Build(IReadOnlyList<string> classes, int seed)
    {
        ArgumentNullException.ThrowIfNull(classes);
        (byte R, byte G, byte B)[] colors = Draw(classes.Count, seed);
        OvbLegendEntry[] legend = new OvbLegendEntry[classes.Count];
        for (int i = 0; i < legend.Length; i++)
            legend[i] = new OvbLegendEntry(classes[i], colors[i].R, colors[i].G, colors[i].B);
        return legend;
    }

    private static double SquaredDistance(int a, int b)
    {
        double dl = Lab[a * 3] - Lab[b * 3], da = Lab[a * 3 + 1] - Lab[b * 3 + 1], db = Lab[a * 3 + 2] - Lab[b * 3 + 2];
        return dl * dl + da * da + db * db;
    }

    private static double[] BuildLab()
    {
        double[] lab = new double[OvbPalette.Count * 3];
        ReadOnlySpan<byte> rgb = OvbPalette.Rgb;
        for (int i = 0; i < OvbPalette.Count; i++)
            SrgbToLab(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], lab.AsSpan(i * 3, 3));
        return lab;
    }

    /// <summary>sRGB (0-255) to CIELAB under D65, as the release's <c>build_palette.srgb_to_lab</c>.</summary>
    internal static void SrgbToLab(int r, int g, int b, Span<double> lab)
    {
        static double Linear(double c) => c <= 0.04045 ? c / 12.92 : Math.Pow((c + 0.055) / 1.055, 2.4);
        double lr = Linear(r / 255.0), lg = Linear(g / 255.0), lb = Linear(b / 255.0);
        double x = (0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb) / 0.95047;
        double y = 0.2126729 * lr + 0.7151522 * lg + 0.0721750 * lb;
        double z = (0.0193339 * lr + 0.1191920 * lg + 0.9503041 * lb) / 1.08883;
        const double eps = 216.0 / 24389.0, kappa = 24389.0 / 27.0;
        static double F(double t) => t > eps ? Math.Cbrt(t) : (kappa * t + 16) / 116;
        double fx = F(x), fy = F(y), fz = F(z);
        lab[0] = 116 * fy - 16;
        lab[1] = 500 * (fx - fy);
        lab[2] = 200 * (fy - fz);
    }

    private struct SplitMix(ulong state)
    {
        private ulong _state = state;

        public ulong Next()
        {
            ulong z = _state += 0x9E3779B97F4A7C15UL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }
}

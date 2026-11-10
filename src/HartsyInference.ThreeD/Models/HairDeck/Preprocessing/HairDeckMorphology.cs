using HartsyInference.Core.Imaging;
using HartsyInference.Vision.Codec;

namespace HartsyInference.ThreeD.Models.HairDeck.Preprocessing;

/// <summary>Mask operators of the HairDeck recipe, matching the scipy.ndimage calls it makes: square dilation with a zero
/// outside, 4-connected labelling, the reflecting Gaussian of gaussian_filter, and uniform_filter with a zero outside.</summary>
internal static class HairDeckMorphology
{
    /// <summary>The integer radius of a recipe distance: <c>max(1, round(value))</c>, rounding half to even as Python does.</summary>
    public static int Radius(double value) => Math.Max(1, (int)Math.Round(value, MidpointRounding.ToEven));

    /// <summary>Binary dilation by a (2r + 1) square; pixels outside the image are background.</summary>
    public static bool[] Dilate(bool[] mask, int height, int width, int radius)
    {
        bool[] horizontal = new bool[mask.Length];
        for (int y = 0; y < height; y++)
        {
            int row = y * width;
            for (int x = 0; x < width; x++)
            {
                int lo = Math.Max(0, x - radius), hi = Math.Min(width - 1, x + radius);
                for (int i = lo; i <= hi; i++)
                {
                    if (mask[row + i])
                    {
                        horizontal[row + x] = true;
                        break;
                    }
                }
            }
        }

        bool[] result = new bool[mask.Length];
        for (int y = 0; y < height; y++)
        {
            int lo = Math.Max(0, y - radius), hi = Math.Min(height - 1, y + radius);
            for (int x = 0; x < width; x++)
            {
                for (int j = lo; j <= hi; j++)
                {
                    if (horizontal[j * width + x])
                    {
                        result[y * width + x] = true;
                        break;
                    }
                }
            }
        }

        return result;
    }

    /// <summary>Labels the 4-connected components of <paramref name="mask"/> in raster order (background 0); returns the count.</summary>
    public static int Label(bool[] mask, int height, int width, int[] labels)
    {
        Array.Clear(labels);
        int count = 0;
        Queue<int> queue = new();
        for (int start = 0; start < mask.Length; start++)
        {
            if (!mask[start] || labels[start] != 0)
            {
                continue;
            }

            count++;
            labels[start] = count;
            queue.Enqueue(start);
            while (queue.Count > 0)
            {
                int p = queue.Dequeue();
                int y = p / width, x = p % width;
                if (x > 0) Visit(p - 1);
                if (x < width - 1) Visit(p + 1);
                if (y > 0) Visit(p - width);
                if (y < height - 1) Visit(p + width);
            }

            void Visit(int q)
            {
                if (mask[q] && labels[q] == 0)
                {
                    labels[q] = count;
                    queue.Enqueue(q);
                }
            }
        }

        return count;
    }

    /// <summary>scipy's gaussian_filter (truncate 4, mode 'reflect') of a float plane, computed in double and stored as float per pass.</summary>
    public static float[] GaussianReflect(float[] plane, int height, int width, double sigma)
    {
        const double Truncate = 4.0;
        int radius = (int)(Truncate * sigma + 0.5);
        double[] kernel = new double[2 * radius + 1];
        double total = 0.0;
        for (int i = -radius; i <= radius; i++)
        {
            kernel[i + radius] = Math.Exp(-0.5 / (sigma * sigma) * i * i);
            total += kernel[i + radius];
        }

        for (int i = 0; i < kernel.Length; i++)
        {
            kernel[i] /= total;
        }

        float[] vertical = new float[plane.Length];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double acc = 0.0;
                for (int k = -radius; k <= radius; k++)
                {
                    acc += kernel[k + radius] * plane[Reflect(y + k, height) * width + x];
                }

                vertical[y * width + x] = (float)acc;
            }
        }

        float[] result = new float[plane.Length];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double acc = 0.0;
                for (int k = -radius; k <= radius; k++)
                {
                    acc += kernel[k + radius] * vertical[y * width + Reflect(x + k, width)];
                }

                result[y * width + x] = (float)acc;
            }
        }

        return result;
    }

    /// <summary>Mean of a (size x size) box around every pixel of a binary plane, with the plane zero outside (uniform_filter, mode 'constant').</summary>
    public static double[] BoxMean(bool[] mask, int height, int width, int size)
    {
        int[] integral = new int[(height + 1) * (width + 1)];
        for (int y = 0; y < height; y++)
        {
            int run = 0;
            for (int x = 0; x < width; x++)
            {
                run += mask[y * width + x] ? 1 : 0;
                integral[(y + 1) * (width + 1) + x + 1] = integral[y * (width + 1) + x + 1] + run;
            }
        }

        int half = size / 2;
        double area = (double)size * size;
        double[] means = new double[mask.Length];
        for (int y = 0; y < height; y++)
        {
            int y0 = Math.Clamp(y - half, 0, height), y1 = Math.Clamp(y - half + size, 0, height);
            for (int x = 0; x < width; x++)
            {
                int x0 = Math.Clamp(x - half, 0, width), x1 = Math.Clamp(x - half + size, 0, width);
                int sum = integral[y1 * (width + 1) + x1] - integral[y0 * (width + 1) + x1]
                    - integral[y1 * (width + 1) + x0] + integral[y0 * (width + 1) + x0];
                means[y * width + x] = sum / area;
            }
        }

        return means;
    }

    /// <summary>Upsamples low-resolution logits bilinearly (torch align_corners=False) and thresholds them at zero.</summary>
    public static bool[] LogitsToMask(float[] logits, int side, int height, int width)
    {
        float[] up = new float[(long)height * width];
        Resample.BilinearPlaneF32(logits, side, side, up, width, height);
        bool[] mask = new bool[up.Length];
        for (int i = 0; i < up.Length; i++)
        {
            mask[i] = up[i] > 0f;
        }

        return mask;
    }

    /// <summary>Reflects an index into <c>[0, n)</c> the way scipy's 'reflect' boundary does (d c b a | a b c d | d c b a).</summary>
    private static int Reflect(int index, int n)
    {
        while (index < 0 || index >= n)
        {
            if (index < 0)
            {
                index = -index - 1;
            }

            if (index >= n)
            {
                index = 2 * n - index - 1;
            }
        }

        return index;
    }
}

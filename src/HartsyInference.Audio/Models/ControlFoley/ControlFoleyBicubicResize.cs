namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Antialiased bicubic resize of interleaved RGB24 frames, bit-exact with torch's uint8 CPU kernel
/// (<c>v2.Resize(..., BICUBIC)</c> on a uint8 tensor): Pillow filter weights quantised to int16 with the largest
/// precision that fits, horizontal pass first, uint8 intermediate.</summary>
internal static class ControlFoleyBicubicResize
{
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<(int In, int Out), Axis> Axes = new();

    private sealed record Axis(int[] Start, int[] Count, short[] Weights, int Stride, int Precision);

    internal static byte[] Resize(byte[] src, int height, int width, int outHeight, int outWidth)
    {
        if (src.Length != height * width * 3)
        {
            throw new ArgumentException($"Expected {height * width * 3} bytes, got {src.Length}.", nameof(src));
        }

        Axis horizontal = Axes.GetOrAdd((width, outWidth), static key => BuildAxis(key.In, key.Out));
        Axis vertical = Axes.GetOrAdd((height, outHeight), static key => BuildAxis(key.In, key.Out));
        byte[] mid = new byte[height * outWidth * 3];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < outWidth; x++)
            {
                int start = horizontal.Start[x], count = horizontal.Count[x], wo = x * horizontal.Stride;
                for (int ch = 0; ch < 3; ch++)
                {
                    long sum = 1L << (horizontal.Precision - 1);
                    for (int k = 0; k < count; k++)
                    {
                        sum += src[(y * width + start + k) * 3 + ch] * (long)horizontal.Weights[wo + k];
                    }

                    mid[(y * outWidth + x) * 3 + ch] = Clip(sum >> horizontal.Precision);
                }
            }
        }

        byte[] dst = new byte[outHeight * outWidth * 3];
        for (int y = 0; y < outHeight; y++)
        {
            int start = vertical.Start[y], count = vertical.Count[y], wo = y * vertical.Stride;
            for (int i = 0; i < outWidth * 3; i++)
            {
                long sum = 1L << (vertical.Precision - 1);
                for (int k = 0; k < count; k++)
                {
                    sum += mid[(start + k) * outWidth * 3 + i] * (long)vertical.Weights[wo + k];
                }

                dst[y * outWidth * 3 + i] = Clip(sum >> vertical.Precision);
            }
        }

        return dst;
    }

    private static byte Clip(long v) => (byte)(v < 0 ? 0 : v > 255 ? 255 : v);

    private static double Cubic(double x)
    {
        const double a = -0.5;
        x = Math.Abs(x);
        if (x < 1.0)
        {
            return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
        }

        return x < 2.0 ? (((x - 5.0) * x + 8.0) * x - 4.0) * a : 0.0;
    }

    private static Axis BuildAxis(int inSize, int outSize)
    {
        double scale = (double)inSize / outSize;
        double support = scale >= 1.0 ? 2.0 * scale : 2.0, invScale = scale >= 1.0 ? 1.0 / scale : 1.0;
        int stride = (int)Math.Ceiling(support) * 2 + 1;
        int[] starts = new int[outSize], counts = new int[outSize];
        double[] weights = new double[outSize * stride];
        double maxWeight = 0;
        for (int i = 0; i < outSize; i++)
        {
            double center = scale * (i + 0.5);
            int min = Math.Max((int)(center - support + 0.5), 0);
            int size = Math.Min((int)(center + support + 0.5), inSize) - min;
            double total = 0;
            for (int j = 0; j < size; j++)
            {
                double w = Cubic((j + min - center + 0.5) * invScale);
                weights[i * stride + j] = w;
                total += w;
            }

            for (int j = 0; j < size; j++)
            {
                weights[i * stride + j] /= total;
                maxWeight = Math.Max(maxWeight, Math.Abs(weights[i * stride + j]));
            }

            starts[i] = min;
            counts[i] = size;
        }

        int precision = 0;
        while (precision < 22 && (int)(0.5 + maxWeight * (1 << precision)) < (1 << 15))
        {
            precision++;
        }

        precision--;
        short[] fixedWeights = new short[weights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            double w = weights[i] * (1 << precision);
            fixedWeights[i] = (short)(w < 0 ? -0.5 + w : 0.5 + w);
        }

        return new Axis(starts, counts, fixedWeights, stride, precision);
    }
}

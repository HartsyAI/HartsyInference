using HartsyInference.Core.Imaging;

namespace HartsyInference.ThreeD.Preprocessing;

/// <summary>Resizes interleaved 8-bit pixels the way OpenCV's <c>INTER_AREA</c> does when shrinking: each output pixel is the mean of the source span it covers, with fractional edge weights. Enlarging falls back to bilinear, which is what OpenCV does for a zoom with a non-integer factor.</summary>
internal static class AreaResample
{
    /// <summary>Resizes <paramref name="src"/> of <paramref name="channels"/> interleaved channels to <paramref name="dstWidth"/> by <paramref name="dstHeight"/>.</summary>
    public static byte[] Resize(byte[] src, int srcWidth, int srcHeight, int channels, int dstWidth, int dstHeight)
    {
        if (dstWidth <= 0 || dstHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(dstWidth), "The target size must be positive.");
        byte[] dst = new byte[dstWidth * dstHeight * channels];
        if (dstWidth >= srcWidth || dstHeight >= srcHeight)
        {
            float[] wide = new float[dst.Length];
            Resample.BilinearHwc8(src, srcWidth, srcHeight, channels, wide, dstWidth, dstHeight, antialias: true, quantizeU8: true);
            for (int i = 0; i < dst.Length; i++)
                dst[i] = (byte)Math.Clamp((int)MathF.Round(wide[i]), 0, 255);
            return dst;
        }
        float[] horizontal = new float[dstWidth * srcHeight * channels];
        for (int x = 0; x < dstWidth; x++)
        {
            (int start, float[] weights) = Span(x, srcWidth, dstWidth);
            for (int y = 0; y < srcHeight; y++)
            {
                for (int c = 0; c < channels; c++)
                {
                    float sum = 0;
                    for (int k = 0; k < weights.Length; k++)
                        sum += weights[k] * src[(y * srcWidth + start + k) * channels + c];
                    horizontal[(y * dstWidth + x) * channels + c] = sum;
                }
            }
        }
        for (int y = 0; y < dstHeight; y++)
        {
            (int start, float[] weights) = Span(y, srcHeight, dstHeight);
            for (int x = 0; x < dstWidth; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    float sum = 0;
                    for (int k = 0; k < weights.Length; k++)
                        sum += weights[k] * horizontal[((start + k) * dstWidth + x) * channels + c];
                    dst[(y * dstWidth + x) * channels + c] = (byte)Math.Clamp((int)MathF.Round(sum), 0, 255);
                }
            }
        }
        return dst;
    }

    private static (int Start, float[] Weights) Span(int index, int srcSize, int dstSize)
    {
        double scale = (double)srcSize / dstSize;
        double from = index * scale;
        double to = Math.Min((index + 1) * scale, srcSize);
        int start = (int)Math.Floor(from);
        int end = (int)Math.Ceiling(to);
        float[] weights = new float[end - start];
        for (int i = start; i < end; i++)
            weights[i - start] = (float)((Math.Min(i + 1, to) - Math.Max(i, from)) / scale);
        return (start, weights);
    }
}

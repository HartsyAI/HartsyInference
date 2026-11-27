namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// The resizes of the released preprocessor on planar <c>C × H × W</c> floats: <c>Resize(518)</c> of an image is bilinear with antialiasing (torchvision's tensor default, the
/// PIL-style triangle filter whose support grows with the downscale factor; unverified against the version upstream ran), of a mask or pointmap nearest.
/// </summary>
public static class SamImageResize
{
    /// <summary>
    /// Bilinear resize with half-pixel centres and, when shrinking, the triangle filter widened by the scale (torch <c>interpolate(mode="bilinear", antialias=True)</c>). Enlarging is plain bilinear.
    /// </summary>
    public static float[] BilinearAntialias(ReadOnlySpan<float> image, int channels, int height, int width, int outHeight, int outWidth)
    {
        if (channels <= 0 || height <= 0 || width <= 0 || outHeight <= 0 || outWidth <= 0 || image.Length != channels * height * width)
            throw new ArgumentException($"An image of {channels} x {height} x {width} holds {channels * height * width} values; got {image.Length}.", nameof(image));
        (int[] firstX, float[][] weightsX) = Weights(width, outWidth);
        (int[] firstY, float[][] weightsY) = Weights(height, outHeight);
        float[] horizontal = new float[channels * height * outWidth];
        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < height; y++)
            {
                ReadOnlySpan<float> row = image.Slice((c * height + y) * width, width);
                Span<float> target = horizontal.AsSpan((c * height + y) * outWidth, outWidth);
                for (int x = 0; x < outWidth; x++)
                {
                    float sum = 0f;
                    float[] w = weightsX[x];
                    for (int k = 0; k < w.Length; k++)
                        sum += w[k] * row[firstX[x] + k];
                    target[x] = sum;
                }
            }
        }

        float[] result = new float[channels * outHeight * outWidth];
        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < outHeight; y++)
            {
                float[] w = weightsY[y];
                Span<float> target = result.AsSpan((c * outHeight + y) * outWidth, outWidth);
                for (int k = 0; k < w.Length; k++)
                {
                    ReadOnlySpan<float> row = horizontal.AsSpan((c * height + firstY[y] + k) * outWidth, outWidth);
                    float weight = w[k];
                    for (int x = 0; x < outWidth; x++)
                        target[x] += weight * row[x];
                }
            }
        }

        return result;
    }

    /// <summary>Nearest-neighbour resize of every channel (<c>floor(dst · src / dst size)</c>, torchvision's rule).</summary>
    public static float[] Nearest(ReadOnlySpan<float> image, int channels, int height, int width, int outHeight, int outWidth)
    {
        if (channels <= 0 || height <= 0 || width <= 0 || outHeight <= 0 || outWidth <= 0 || image.Length != channels * height * width)
            throw new ArgumentException($"An image of {channels} x {height} x {width} holds {channels * height * width} values; got {image.Length}.", nameof(image));
        float[] result = new float[channels * outHeight * outWidth];
        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < outHeight; y++)
            {
                int sy = Math.Min(height - 1, (int)((long)y * height / outHeight));
                for (int x = 0; x < outWidth; x++)
                    result[(c * outHeight + y) * outWidth + x] = image[(c * height + sy) * width + Math.Min(width - 1, (int)((long)x * width / outWidth))];
            }
        }

        return result;
    }

    /// <summary>Per output index: the first contributing input index and the normalised triangle weights (PIL's <c>precompute_coeffs</c> for the bilinear filter).</summary>
    private static (int[] First, float[][] Weights) Weights(int inSize, int outSize)
    {
        double scale = (double)inSize / outSize, filterScale = Math.Max(scale, 1.0), support = filterScale;
        int[] first = new int[outSize];
        float[][] weights = new float[outSize][];
        for (int i = 0; i < outSize; i++)
        {
            double center = (i + 0.5) * scale;
            int min = Math.Max((int)(center - support + 0.5), 0), max = Math.Min((int)(center + support + 0.5), inSize);
            double[] raw = new double[max - min];
            double total = 0;
            for (int j = 0; j < raw.Length; j++)
            {
                double x = Math.Abs((j + min - center + 0.5) / filterScale);
                raw[j] = x < 1.0 ? 1.0 - x : 0.0;
                total += raw[j];
            }

            first[i] = min;
            weights[i] = raw.Select(v => (float)(v / total)).ToArray();
        }

        return (first, weights);
    }
}

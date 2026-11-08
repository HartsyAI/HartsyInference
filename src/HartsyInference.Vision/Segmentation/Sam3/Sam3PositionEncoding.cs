using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Sine position encoding over a feature grid. Y channels come first, then X; no learnable parameters.</summary>
public static class Sam3PositionEncoding
{
    private const double Temperature = 10000.0;
    private const double NormalizeEpsilon = 1e-6;

    /// <summary>Returns a new F32 <c>[1, dModel, height, width]</c> encoding owned by the caller.</summary>
    /// <param name="dModel">Total channels; half encode y and half encode x. Must be a positive multiple of 4.</param>
    /// <param name="height">Grid height in feature cells.</param>
    /// <param name="width">Grid width in feature cells.</param>
    public static Tensor Create(int dModel, int height, int width)
    {
        if (dModel <= 0 || dModel % 4 != 0)
            throw new ArgumentOutOfRangeException(nameof(dModel), dModel, "Position encoding width must be a positive multiple of 4.");
        if (height <= 0)
            throw new ArgumentOutOfRangeException(nameof(height), height, "Grid height must be positive.");
        if (width <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), width, "Grid width must be positive.");

        int half = dModel / 2;
        Tensor output = new Tensor(new TensorShape(1, dModel, height, width), DType.F32);
        Span<float> dst = output.AsSpan<float>();

        // Positions are 1-indexed and divided by the side length plus epsilon, as upstream does.
        double yScale = 2.0 * Math.PI / (height + NormalizeEpsilon);
        double xScale = 2.0 * Math.PI / (width + NormalizeEpsilon);

        for (int c = 0; c < half; c++)
        {
            double divisor = Math.Pow(Temperature, 2.0 * (c / 2) / half);

            for (int y = 0; y < height; y++)
            {
                float value = Basis(c, (y + 1) * yScale / divisor);
                dst.Slice((c * height + y) * width, width).Fill(value);
            }

            int xChannelStart = (half + c) * height;
            for (int x = 0; x < width; x++)
            {
                float value = Basis(c, (x + 1) * xScale / divisor);
                for (int y = 0; y < height; y++)
                {
                    dst[(xChannelStart + y) * width + x] = value;
                }
            }
        }

        return output;
    }

    // Sine on even channels and cosine on odd channels, interleaved within each axis.
    private static float Basis(int channel, double angle)
    {
        return (channel & 1) == 0 ? (float)Math.Sin(angle) : (float)Math.Cos(angle);
    }
}

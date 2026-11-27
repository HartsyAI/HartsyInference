using HartsyInference.Core.Imaging;

namespace HartsyInference.ThreeD.Preprocessing;

/// <summary>Hunyuan3D-2's input preparation (<c>ImageProcessorV2</c> in <c>hy3dgen/shapegen/preprocessors.py</c>): the foreground is recentred on a square canvas with a border, composited on white and resized to the model's input size. The model is fed the RGB only; the mask the upstream class also returns is not used by the shape pipeline.</summary>
public static class Hunyuan3DImageProcessor
{
    /// <summary>The upstream default border around the foreground, as a fraction of the canvas.</summary>
    public const double DefaultBorderRatio = 0.15;

    /// <summary>The model's input edge in pixels.</summary>
    public const int DefaultSize = 512;

    /// <summary>Prepares one image. A null <paramref name="alpha"/> makes the whole image the foreground, as upstream does for an RGB input.</summary>
    /// <param name="rgb">Row-major interleaved RGB24.</param>
    /// <param name="alpha">Row-major coverage, or null.</param>
    /// <exception cref="ArgumentException">The foreground is empty or a single row or column, which upstream also rejects.</exception>
    public static PreparedImage Process(byte[] rgb, byte[]? alpha, int width, int height, int size = DefaultSize, double borderRatio = DefaultBorderRatio)
    {
        byte[] coverage = ForegroundPixels.Validate(rgb, alpha, width, height);
        if (size <= 0 || borderRatio is < 0 or >= 1)
            throw new ArgumentOutOfRangeException(nameof(size), "The size must be positive and the border ratio in [0, 1).");
        (int MinX, int MinY, int MaxX, int MaxY)? box = ForegroundPixels.Bounds(coverage, width, height, threshold: 0);
        // Upstream measures the box as max - min, so a foreground one pixel wide has zero extent and is refused.
        if (box is null || box.Value.MaxX == box.Value.MinX || box.Value.MaxY == box.Value.MinY)
            throw new ArgumentException("The image has no foreground at least two pixels in each direction; supply a cutout or let the engine matte it.", nameof(alpha));
        int boxHeight = box.Value.MaxY - box.Value.MinY;
        int boxWidth = box.Value.MaxX - box.Value.MinX;

        int canvas = Math.Max(height, width);
        double scale = (int)(canvas * (1.0 - borderRatio)) / (double)Math.Max(boxHeight, boxWidth);
        int scaledHeight = (int)(boxHeight * scale);
        int scaledWidth = (int)(boxWidth * scale);
        if (scaledHeight == 0 || scaledWidth == 0)
            throw new ArgumentException("The foreground is too thin to scale to the canvas.", nameof(alpha));

        byte[] rgba = ForegroundPixels.ToRgba(rgb, coverage);
        byte[] crop = ForegroundPixels.CropRgba(rgba, width, height, box.Value.MinX, box.Value.MinY, box.Value.MaxX, box.Value.MaxY);
        byte[] scaled = AreaResample.Resize(crop, boxWidth, boxHeight, 4, scaledWidth, scaledHeight);

        // Composite on white at the canvas size: rgb * a + 255 * (1 - a), clipped and truncated like numpy's astype(uint8).
        int top = (canvas - scaledHeight) / 2;
        int left = (canvas - scaledWidth) / 2;
        byte[] composite = new byte[canvas * canvas * 3];
        Array.Fill(composite, (byte)255);
        for (int y = 0; y < scaledHeight; y++)
        {
            for (int x = 0; x < scaledWidth; x++)
            {
                int source = (y * scaledWidth + x) * 4;
                float a = scaled[source + 3] / 255f;
                int target = ((top + y) * canvas + left + x) * 3;
                for (int c = 0; c < 3; c++)
                    composite[target + c] = (byte)Math.Clamp((int)(scaled[source + c] * a + 255f * (1f - a)), 0, 255);
            }
        }
        if (canvas == size)
            return new PreparedImage(composite, size, size);

        // cv2.INTER_CUBIC (a = -0.75), no antialiasing: the same kernel as OpenCV, with this library's border handling.
        float[] wide = new float[size * size * 3];
        Resample.BicubicHwc8(composite, canvas, canvas, 3, wide, size, size, a: -0.75f, antialias: false);
        byte[] result = new byte[wide.Length];
        for (int i = 0; i < result.Length; i++)
            result[i] = (byte)Math.Clamp((int)MathF.Round(wide[i]), 0, 255);
        return new PreparedImage(result, size, size);
    }
}

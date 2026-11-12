namespace HartsyInference.ThreeD.Preprocessing;

/// <summary>TripoSR's input preparation (<c>run.py</c> and <c>tsr/utils.py</c> of <c>stabilityai/TripoSR</c>): <c>resize_foreground</c> crops the foreground, pads it to a square and then to the foreground ratio, and the result is composited on mid-gray. The model's own processor resizes to its input size afterwards.</summary>
public static class TripoSrImageProcessor
{
    /// <summary>The upstream <c>--foreground-ratio</c> default.</summary>
    public const double DefaultForegroundRatio = 0.85;

    /// <summary>Prepares one image. A null <paramref name="alpha"/> makes the whole image the foreground.</summary>
    /// <exception cref="ArgumentException">The image has no foreground.</exception>
    public static PreparedImage Process(byte[] rgb, byte[]? alpha, int width, int height, double foregroundRatio = DefaultForegroundRatio)
    {
        byte[] coverage = ForegroundPixels.Validate(rgb, alpha, width, height);
        if (!(foregroundRatio > 0 && foregroundRatio <= 1))
            throw new ArgumentOutOfRangeException(nameof(foregroundRatio), foregroundRatio, "The foreground ratio must be in (0, 1].");
        (int MinX, int MinY, int MaxX, int MaxY)? box = ForegroundPixels.Bounds(coverage, width, height, threshold: 0);
        if (box is null)
            throw new ArgumentException("The image has no foreground; supply a cutout or let the engine matte it.", nameof(alpha));

        // Upstream slices image[y1:y2, x1:x2] with the inclusive extremes, so the last foreground row and column fall outside the crop.
        int cropWidth = Math.Max(box.Value.MaxX - box.Value.MinX, 1);
        int cropHeight = Math.Max(box.Value.MaxY - box.Value.MinY, 1);
        byte[] rgba = ForegroundPixels.ToRgba(rgb, coverage);
        byte[] crop = ForegroundPixels.CropRgba(rgba, width, height, box.Value.MinX, box.Value.MinY, box.Value.MinX + cropWidth, box.Value.MinY + cropHeight);

        int side = Math.Max(cropWidth, cropHeight);
        int outSide = (int)(side / foregroundRatio);
        int padTop = (side - cropHeight) / 2 + (outSide - side) / 2;
        int padLeft = (side - cropWidth) / 2 + (outSide - side) / 2;
        byte[] output = new byte[outSide * outSide * 3];
        Array.Fill(output, (byte)127);
        // Padding is transparent black, which composites to mid-gray; the pasted crop composites rgb * a + (1 - a) * 0.5 in float, truncated.
        for (int y = 0; y < cropHeight; y++)
        {
            for (int x = 0; x < cropWidth; x++)
            {
                int source = (y * cropWidth + x) * 4;
                float a = crop[source + 3] / 255f;
                int target = ((padTop + y) * outSide + padLeft + x) * 3;
                for (int c = 0; c < 3; c++)
                    output[target + c] = Composite(crop[source + c] / 255f, a);
            }
        }
        return new PreparedImage(output, outSide, outSide);
    }

    private static byte Composite(float channel, float alpha) => (byte)Math.Clamp((int)((channel * alpha + (1f - alpha) * 0.5f) * 255f), 0, 255);
}

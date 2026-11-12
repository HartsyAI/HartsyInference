namespace HartsyInference.ThreeD.Preprocessing;

/// <summary>Pixel helpers shared by the model-specific input preparations: validation, the foreground box, the RGBA interleave and a cropping that pads outside the image with zeros.</summary>
internal static class ForegroundPixels
{
    /// <summary>Checks the buffers against the size and returns the alpha plane, or an opaque one when <paramref name="alpha"/> is null.</summary>
    public static byte[] Validate(byte[] rgb, byte[]? alpha, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(rgb);
        if (width <= 0 || height <= 0 || rgb.Length != (long)width * height * 3)
            throw new ArgumentException($"A {width} x {height} image holds {(long)width * height * 3} RGB bytes; got {rgb.Length}.", nameof(rgb));
        if (alpha is not null && alpha.Length != width * height)
            throw new ArgumentException($"A {width} x {height} alpha plane holds {width * height} bytes; got {alpha.Length}.", nameof(alpha));
        return alpha ?? Enumerable.Repeat((byte)255, width * height).ToArray();
    }

    /// <summary>The inclusive box of the pixels whose alpha is above <paramref name="threshold"/>, or null when there are none.</summary>
    public static (int MinX, int MinY, int MaxX, int MaxY)? Bounds(byte[] alpha, int width, int height, int threshold)
    {
        int minX = width, minY = height, maxX = -1, maxY = -1;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (alpha[y * width + x] <= threshold)
                    continue;
                minX = Math.Min(minX, x);
                maxX = Math.Max(maxX, x);
                minY = Math.Min(minY, y);
                maxY = y;
            }
        }
        return maxX < 0 ? null : (minX, minY, maxX, maxY);
    }

    /// <summary>Interleaves RGB and alpha into RGBA.</summary>
    public static byte[] ToRgba(byte[] rgb, byte[] alpha)
    {
        byte[] rgba = new byte[alpha.Length * 4];
        for (int i = 0; i < alpha.Length; i++)
        {
            rgba[i * 4] = rgb[i * 3];
            rgba[i * 4 + 1] = rgb[i * 3 + 1];
            rgba[i * 4 + 2] = rgb[i * 3 + 2];
            rgba[i * 4 + 3] = alpha[i];
        }
        return rgba;
    }

    /// <summary>The window <c>[x0, x1) x [y0, y1)</c> of an RGBA image, with zeros (transparent black) where it leaves the image.</summary>
    public static byte[] CropRgba(byte[] rgba, int width, int height, int x0, int y0, int x1, int y1)
    {
        int outWidth = x1 - x0;
        int outHeight = y1 - y0;
        byte[] result = new byte[outWidth * outHeight * 4];
        for (int y = Math.Max(y0, 0); y < Math.Min(y1, height); y++)
        {
            int xFrom = Math.Max(x0, 0);
            int xTo = Math.Min(x1, width);
            if (xTo <= xFrom)
                continue;
            Buffer.BlockCopy(rgba, (y * width + xFrom) * 4, result, ((y - y0) * outWidth + (xFrom - x0)) * 4, (xTo - xFrom) * 4);
        }
        return result;
    }
}

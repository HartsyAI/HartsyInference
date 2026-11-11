namespace HartsyInference.ThreeD.Geometry;

/// <summary>An 8-bit image with 1, 3 or 4 interleaved channels, row-major, first row at the top.</summary>
public sealed class TextureImage
{
    /// <summary>Wraps <paramref name="pixels"/> (not copied) after validating its length.</summary>
    public TextureImage(int width, int height, int channels, byte[] pixels)
    {
        ArgumentNullException.ThrowIfNull(pixels);
        if (width <= 0 || height <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), "Image dimensions must be positive.");
        if (channels is not (1 or 3 or 4))
            throw new ArgumentOutOfRangeException(nameof(channels), "Only 1, 3 or 4 channels are supported.");
        if (pixels.LongLength != (long)width * height * channels)
            throw new ArgumentException($"Expected {(long)width * height * channels} bytes, got {pixels.Length}.", nameof(pixels));
        Width = width;
        Height = height;
        Channels = channels;
        Pixels = pixels;
    }

    /// <summary>Width in pixels.</summary>
    public int Width { get; }

    /// <summary>Height in pixels.</summary>
    public int Height { get; }

    /// <summary>Interleaved channels per pixel.</summary>
    public int Channels { get; }

    /// <summary>Pixel bytes, <c>(y * Width + x) * Channels + c</c>.</summary>
    public byte[] Pixels { get; }

    /// <summary>Quantizes <paramref name="count"/> channels of a float image, starting at <paramref name="offset"/> of <paramref name="stride"/>
    /// channels per texel, as <c>clamp(round(v * 255), 0, 255)</c>.</summary>
    public static TextureImage FromFloats(ReadOnlySpan<float> data, int width, int height, int stride, int offset, int count)
    {
        if ((long)width * height * stride != data.Length || offset < 0 || count <= 0 || offset + count > stride)
            throw new ArgumentException("Float image shape does not match the requested channel slice.");
        byte[] pixels = new byte[width * height * count];
        for (int i = 0; i < width * height; i++)
            for (int c = 0; c < count; c++)
                pixels[i * count + c] = Quantize(data[i * stride + offset + c]);
        return new TextureImage(width, height, count, pixels);
    }

    private static byte Quantize(float v)
    {
        float s = MathF.Round(v * 255f);
        if (!(s > 0f)) return 0;
        return s >= 255f ? (byte)255 : (byte)s;
    }
}

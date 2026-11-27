using System.Buffers.Binary;

namespace HartsyInference.Vision.Codec;

/// <summary>EXIF orientation (tag 0x0112) of a TIFF block, and the pixel transform PIL's <c>ImageOps.exif_transpose</c> applies for it.</summary>
public static class ExifOrientation
{
    /// <summary>The orientation of an upright image; also what an absent or unreadable tag means.</summary>
    public const int Upright = 1;

    private const ushort OrientationTag = 0x0112;

    /// <summary>The orientation of a TIFF block (a PNG <c>eXIf</c> chunk body), or <see cref="Upright"/> when absent or malformed.</summary>
    public static int ReadTiff(byte[]? tiff)
    {
        if (tiff is null || tiff.Length < 8)
            return Upright;
        bool littleEndian = tiff[0] == (byte)'I' && tiff[1] == (byte)'I';
        bool bigEndian = tiff[0] == (byte)'M' && tiff[1] == (byte)'M';
        if (!littleEndian && !bigEndian)
            return Upright;
        ReadOnlySpan<byte> data = tiff;
        if (ReadUInt16(data, 2, littleEndian) != 42)
            return Upright;
        long ifd = ReadUInt32(data, 4, littleEndian);
        if (ifd < 8 || ifd + 2 > data.Length)
            return Upright;
        int count = ReadUInt16(data, (int)ifd, littleEndian);
        for (int i = 0; i < count; i++)
        {
            long entry = ifd + 2 + 12L * i;
            if (entry + 12 > data.Length)
                break;
            int at = (int)entry;
            if (ReadUInt16(data, at, littleEndian) != OrientationTag)
                continue;
            int value = ReadUInt16(data, at + 8, littleEndian);
            return value is >= 1 and <= 8 ? value : Upright;
        }
        return Upright;
    }

    /// <summary>Applies an orientation to RGB24 pixels and their optional alpha plane, returning the oriented planes and size.</summary>
    public static (byte[] Rgb, byte[]? Alpha, int Width, int Height) Apply(byte[] rgb, byte[]? alpha, int width, int height, int orientation)
    {
        ArgumentNullException.ThrowIfNull(rgb);
        if (orientation is < 1 or > 8)
            throw new ArgumentOutOfRangeException(nameof(orientation), orientation, "EXIF orientation must be 1 to 8.");
        if (rgb.Length != (long)width * height * 3)
            throw new ArgumentException($"RGB buffer holds {rgb.Length} bytes; a {width}x{height} image needs {(long)width * height * 3}.", nameof(rgb));
        if (alpha is not null && alpha.Length != (long)width * height)
            throw new ArgumentException($"Alpha plane holds {alpha.Length} bytes; a {width}x{height} image needs {(long)width * height}.", nameof(alpha));
        if (orientation == Upright)
            return (rgb, alpha, width, height);

        bool swaps = orientation >= 5;
        int outWidth = swaps ? height : width;
        int outHeight = swaps ? width : height;
        byte[] outRgb = new byte[(long)outWidth * outHeight * 3];
        byte[]? outAlpha = alpha is null ? null : new byte[(long)outWidth * outHeight];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                MapPixel(orientation, x, y, width, height, out int nx, out int ny);
                int src = y * width + x;
                int dst = ny * outWidth + nx;
                outRgb[dst * 3] = rgb[src * 3];
                outRgb[dst * 3 + 1] = rgb[src * 3 + 1];
                outRgb[dst * 3 + 2] = rgb[src * 3 + 2];
                if (outAlpha is not null)
                    outAlpha[dst] = alpha![src];
            }
        }
        return (outRgb, outAlpha, outWidth, outHeight);
    }

    /// <summary>Destination of input pixel (x, y) under the orientation, in PIL's transpose / rotate convention.</summary>
    private static void MapPixel(int orientation, int x, int y, int width, int height, out int nx, out int ny)
    {
        switch (orientation)
        {
            case 2: nx = width - 1 - x; ny = y; break;
            case 3: nx = width - 1 - x; ny = height - 1 - y; break;
            case 4: nx = x; ny = height - 1 - y; break;
            case 5: nx = y; ny = x; break;
            case 6: nx = height - 1 - y; ny = x; break;
            case 7: nx = height - 1 - y; ny = width - 1 - x; break;
            default: nx = y; ny = width - 1 - x; break; // 8
        }
    }

    private static int ReadUInt16(ReadOnlySpan<byte> data, int at, bool littleEndian) =>
        littleEndian ? BinaryPrimitives.ReadUInt16LittleEndian(data[at..]) : BinaryPrimitives.ReadUInt16BigEndian(data[at..]);

    private static long ReadUInt32(ReadOnlySpan<byte> data, int at, bool littleEndian) =>
        littleEndian ? BinaryPrimitives.ReadUInt32LittleEndian(data[at..]) : BinaryPrimitives.ReadUInt32BigEndian(data[at..]);
}

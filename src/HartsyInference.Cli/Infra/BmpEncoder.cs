namespace HartsyInference.Cli.Infra;

/// <summary>Encodes tightly-packed 24-bit RGB pixel data into an in-memory BMP file, so image artifacts can be
/// returned as bytes without pulling in an external image library (the engine only exposes a file-writing helper).</summary>
public static class BmpEncoder
{
    /// <summary>Encodes <paramref name="rgb"/> (row-major, top-to-bottom, 3 bytes/pixel R,G,B) as a 24-bit BMP.</summary>
    public static byte[] Encode(byte[] rgb, int width, int height)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentException($"Invalid image dimensions {width}x{height}.");
        if (rgb.Length < width * height * 3)
            throw new ArgumentException($"RGB buffer too small: {rgb.Length} < {width * height * 3}.");

        int rowStride = width * 3;
        int rowPadded = (rowStride + 3) & ~3;
        int pixelDataSize = rowPadded * height;
        const int headerSize = 54;
        byte[] bmp = new byte[headerSize + pixelDataSize];

        // BITMAPFILEHEADER
        bmp[0] = (byte)'B';
        bmp[1] = (byte)'M';
        WriteI32(bmp, 2, headerSize + pixelDataSize);
        WriteI32(bmp, 10, headerSize);

        // BITMAPINFOHEADER
        WriteI32(bmp, 14, 40);
        WriteI32(bmp, 18, width);
        WriteI32(bmp, 22, height);
        WriteI16(bmp, 26, 1);
        WriteI16(bmp, 28, 24);
        WriteI32(bmp, 34, pixelDataSize);
        WriteI32(bmp, 38, 2835);
        WriteI32(bmp, 42, 2835);

        // Pixel rows are stored bottom-up as BGR.
        for (int y = 0; y < height; y++)
        {
            int srcRow = y * rowStride;
            int dstRow = headerSize + (height - 1 - y) * rowPadded;
            for (int x = 0; x < width; x++)
            {
                int src = srcRow + x * 3;
                int dst = dstRow + x * 3;
                bmp[dst + 0] = rgb[src + 2];
                bmp[dst + 1] = rgb[src + 1];
                bmp[dst + 2] = rgb[src + 0];
            }
        }

        return bmp;
    }

    private static void WriteI32(byte[] buffer, int offset, int value)
    {
        buffer[offset + 0] = (byte)value;
        buffer[offset + 1] = (byte)(value >> 8);
        buffer[offset + 2] = (byte)(value >> 16);
        buffer[offset + 3] = (byte)(value >> 24);
    }

    private static void WriteI16(byte[] buffer, int offset, int value)
    {
        buffer[offset + 0] = (byte)value;
        buffer[offset + 1] = (byte)(value >> 8);
    }
}

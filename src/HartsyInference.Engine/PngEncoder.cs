using System.Buffers.Binary;
using System.IO.Compression;

namespace HartsyInference.Engine;

/// <summary>Encodes tightly-packed 24-bit RGB pixel data into an in-memory PNG (8-bit truecolor), using the runtime's
/// built-in zlib so no external image library is pulled in. PNG is preferred over BMP for saved artifacts: it is
/// universally viewable, lossless, and far smaller.</summary>
public static class PngEncoder
{
    private static readonly byte[] Signature = { 0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A };

    /// <summary>Encodes <paramref name="rgb"/> (row-major, top-to-bottom, 3 bytes/pixel R,G,B) as a color-type-2 PNG.</summary>
    public static byte[] Encode(byte[] rgb, int width, int height)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentException($"Invalid image dimensions {width}x{height}.");
        if (rgb.Length < (long)width * height * 3)
            throw new ArgumentException($"RGB buffer too small: {rgb.Length} < {(long)width * height * 3}.");

        using MemoryStream png = new MemoryStream();
        png.Write(Signature, 0, Signature.Length);

        Span<byte> ihdr = stackalloc byte[13];
        BinaryPrimitives.WriteInt32BigEndian(ihdr[..4], width);
        BinaryPrimitives.WriteInt32BigEndian(ihdr.Slice(4, 4), height);
        ihdr[8] = 8;  // bit depth
        ihdr[9] = 2;  // color type: truecolor RGB
        ihdr[10] = 0; // deflate
        ihdr[11] = 0; // adaptive filtering
        ihdr[12] = 0; // no interlace
        WriteChunk(png, "IHDR", ihdr);

        WriteChunk(png, "IDAT", Deflate(FilterScanlines(rgb, width, height)));
        WriteChunk(png, "IEND", ReadOnlySpan<byte>.Empty);
        return png.ToArray();
    }

    // Prepend the per-scanline filter byte (0 = None) that PNG requires ahead of each row's raw pixels.
    private static byte[] FilterScanlines(byte[] rgb, int width, int height)
    {
        int stride = width * 3;
        byte[] raw = new byte[(stride + 1) * height];
        for (int y = 0; y < height; y++)
        {
            int dst = y * (stride + 1);
            raw[dst] = 0;
            Array.Copy(rgb, y * stride, raw, dst + 1, stride);
        }
        return raw;
    }

    private static byte[] Deflate(byte[] data)
    {
        using MemoryStream compressed = new MemoryStream();
        using (ZLibStream zlib = new ZLibStream(compressed, CompressionLevel.Optimal, leaveOpen: true))
            zlib.Write(data, 0, data.Length);
        return compressed.ToArray();
    }

    private static void WriteChunk(Stream png, string type, ReadOnlySpan<byte> payload)
    {
        Span<byte> length = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(length, payload.Length);
        png.Write(length);

        byte[] typeAndData = new byte[4 + payload.Length];
        typeAndData[0] = (byte)type[0];
        typeAndData[1] = (byte)type[1];
        typeAndData[2] = (byte)type[2];
        typeAndData[3] = (byte)type[3];
        payload.CopyTo(typeAndData.AsSpan(4));
        png.Write(typeAndData, 0, typeAndData.Length);

        Span<byte> crc = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(crc, Crc32(typeAndData));
        png.Write(crc);
    }

    private static readonly uint[] CrcTable = BuildCrcTable();

    private static uint[] BuildCrcTable()
    {
        uint[] table = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++)
                c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[n] = c;
        }
        return table;
    }

    private static uint Crc32(ReadOnlySpan<byte> data)
    {
        uint c = 0xFFFFFFFFu;
        foreach (byte b in data)
            c = CrcTable[(c ^ b) & 0xFF] ^ (c >> 8);
        return c ^ 0xFFFFFFFFu;
    }
}

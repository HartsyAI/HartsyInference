using System.Buffers.Binary;
using System.IO.Compression;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>Minimal 8-bit PNG encoder (gray, RGB or RGBA with Sub filtering) for GLB textures. The engine's own PngEncoder lives in a package
/// that depends on this one, so the texture path carries its own.</summary>
internal static class PngTextureEncoder
{
    private static readonly byte[] Signature = [0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A];
    private static readonly uint[] CrcTable = BuildCrcTable();

    public static byte[] Encode(TextureImage image)
    {
        ArgumentNullException.ThrowIfNull(image);
        int channels = image.Channels;
        int stride = image.Width * channels;
        byte[] raw = new byte[(stride + 1) * image.Height];
        for (int y = 0; y < image.Height; y++)
        {
            int dst = y * (stride + 1);
            int src = y * stride;
            raw[dst] = 1;
            for (int i = 0; i < stride; i++)
            {
                byte left = i >= channels ? image.Pixels[src + i - channels] : (byte)0;
                raw[dst + 1 + i] = (byte)(image.Pixels[src + i] - left);
            }
        }

        using MemoryStream png = new();
        png.Write(Signature);
        Span<byte> header = stackalloc byte[13];
        BinaryPrimitives.WriteInt32BigEndian(header, image.Width);
        BinaryPrimitives.WriteInt32BigEndian(header[4..], image.Height);
        header[8] = 8;
        header[9] = channels switch { 1 => 0, 3 => 2, _ => 6 };
        WriteChunk(png, "IHDR", header);
        using (MemoryStream compressed = new())
        {
            using (ZLibStream zlib = new(compressed, CompressionLevel.Optimal, leaveOpen: true))
                zlib.Write(raw);
            WriteChunk(png, "IDAT", compressed.ToArray());
        }
        WriteChunk(png, "IEND", ReadOnlySpan<byte>.Empty);
        return png.ToArray();
    }

    private static void WriteChunk(Stream png, string type, ReadOnlySpan<byte> payload)
    {
        Span<byte> word = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(word, payload.Length);
        png.Write(word);
        byte[] body = new byte[4 + payload.Length];
        for (int i = 0; i < 4; i++) body[i] = (byte)type[i];
        payload.CopyTo(body.AsSpan(4));
        png.Write(body);
        BinaryPrimitives.WriteUInt32BigEndian(word, Crc32(body));
        png.Write(word);
    }

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

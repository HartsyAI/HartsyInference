namespace HartsyInference.Server.Imaging;

/// <summary>Minimal in-memory PNG encoder for RGB images. Uses uncompressed (stored) DEFLATE blocks, so
/// there is no dependency on a compression library and the output is deterministic. Produces a valid
/// 8-bit truecolor PNG that any standard decoder (and the engine's own <c>PngDecoder</c>) can read.</summary>
public static class PngImageWriter
{
    private static readonly byte[] Signature = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    /// <summary>Encodes HWC RGB bytes [0,255] (<c>width*height*3</c>) into a PNG byte stream.</summary>
    public static byte[] Encode(ReadOnlySpan<byte> rgb, int width, int height)
    {
        if (rgb.Length < width * height * 3)
            throw new ArgumentException($"RGB buffer too small: {rgb.Length} < {width * height * 3}.", nameof(rgb));

        using MemoryStream ms = new MemoryStream();
        ms.Write(Signature);

        // IHDR
        byte[] ihdr = new byte[13];
        WriteBe32(ihdr, 0, (uint)width);
        WriteBe32(ihdr, 4, (uint)height);
        ihdr[8] = 8;  // bit depth
        ihdr[9] = 2;  // color type: truecolor RGB
        ihdr[10] = 0; // compression
        ihdr[11] = 0; // filter
        ihdr[12] = 0; // interlace
        WriteChunk(ms, "IHDR", ihdr);

        // Raw scanlines: filter byte 0 + RGB row.
        int stride = width * 3;
        byte[] raw = new byte[height * (stride + 1)];
        for (int y = 0; y < height; y++)
        {
            int dst = y * (stride + 1);
            raw[dst] = 0; // filter type none
            rgb.Slice(y * stride, stride).CopyTo(raw.AsSpan(dst + 1, stride));
        }

        byte[] idat = ZlibStore(raw);
        WriteChunk(ms, "IDAT", idat);
        WriteChunk(ms, "IEND", []);

        return ms.ToArray();
    }

    /// <summary>Wraps raw bytes in a zlib stream using uncompressed DEFLATE stored blocks.</summary>
    private static byte[] ZlibStore(byte[] data)
    {
        using MemoryStream ms = new MemoryStream();
        ms.WriteByte(0x78); // CMF: deflate, 32K window
        ms.WriteByte(0x01); // FLG: no dict, check bits

        const int maxBlock = 65535;
        int offset = 0;
        while (offset < data.Length || data.Length == 0)
        {
            int len = Math.Min(maxBlock, data.Length - offset);
            bool last = offset + len >= data.Length;
            ms.WriteByte((byte)(last ? 1 : 0)); // BFINAL + BTYPE=00 (stored)
            ms.WriteByte((byte)(len & 0xFF));
            ms.WriteByte((byte)((len >> 8) & 0xFF));
            int nlen = ~len;
            ms.WriteByte((byte)(nlen & 0xFF));
            ms.WriteByte((byte)((nlen >> 8) & 0xFF));
            ms.Write(data, offset, len);
            offset += len;
            if (data.Length == 0) break;
        }

        WriteBe32Stream(ms, Adler32(data));
        return ms.ToArray();
    }

    private static void WriteChunk(Stream s, string type, ReadOnlySpan<byte> data)
    {
        Span<byte> lenBuf = stackalloc byte[4];
        WriteBe32(lenBuf, 0, (uint)data.Length);
        s.Write(lenBuf);

        byte[] typeBytes = [(byte)type[0], (byte)type[1], (byte)type[2], (byte)type[3]];
        s.Write(typeBytes);
        s.Write(data);

        uint crc = Crc32(typeBytes, data);
        WriteBe32(lenBuf, 0, crc);
        s.Write(lenBuf);
    }

    private static void WriteBe32(Span<byte> buf, int offset, uint value)
    {
        buf[offset] = (byte)(value >> 24);
        buf[offset + 1] = (byte)(value >> 16);
        buf[offset + 2] = (byte)(value >> 8);
        buf[offset + 3] = (byte)value;
    }

    private static void WriteBe32Stream(Stream s, uint value)
    {
        s.WriteByte((byte)(value >> 24));
        s.WriteByte((byte)(value >> 16));
        s.WriteByte((byte)(value >> 8));
        s.WriteByte((byte)value);
    }

    private static uint Adler32(ReadOnlySpan<byte> data)
    {
        const uint mod = 65521;
        uint a = 1, b = 0;
        foreach (byte d in data)
        {
            a = (a + d) % mod;
            b = (b + a) % mod;
        }
        return (b << 16) | a;
    }

    private static readonly uint[] CrcTable = BuildCrcTable();

    private static uint[] BuildCrcTable()
    {
        uint[] table = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++)
            {
                c = (c & 1) != 0 ? 0xEDB88320 ^ (c >> 1) : c >> 1;
            }
            table[n] = c;
        }
        return table;
    }

    private static uint Crc32(ReadOnlySpan<byte> type, ReadOnlySpan<byte> data)
    {
        uint crc = 0xFFFFFFFF;
        foreach (byte b in type) crc = CrcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
        foreach (byte b in data) crc = CrcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
        return crc ^ 0xFFFFFFFF;
    }
}

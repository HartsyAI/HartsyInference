using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using HartsyInference.Engine;

namespace HartsyInference.BenchmarkRunner.Evidence;
/// <summary>Bounded validation of the exact PNG format emitted by the engine; never unbounded image decompression.</summary>
public static class OutputChecks
{
    public static bool Text(string path)
    {
        string text = new UTF8Encoding(false, true).GetString(File.ReadAllBytes(path));
        return !string.IsNullOrWhiteSpace(text) && !text.Contains('�');
    }

    public static bool Image(string path, int width, int height)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length < 57 || !bytes.AsSpan(0, 8).SequenceEqual(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 }))
            throw new InvalidDataException("Invalid PNG signature.");
        int offset = 8;
        ReadOnlySpan<byte> header = Chunk(bytes, ref offset, "IHDR");
        if (header.Length != 13 || BinaryPrimitives.ReadInt32BigEndian(header) != width || BinaryPrimitives.ReadInt32BigEndian(
            header[4..]) != height || !header[8..].SequenceEqual(new byte[] { 8, 2, 0, 0, 0 }))
            throw new InvalidDataException("PNG shape or format differs from the frozen workload.");
        byte[] compressed = Chunk(bytes, ref offset, "IDAT").ToArray();
        if (Chunk(bytes, ref offset, "IEND").Length != 0 || offset != bytes.Length)
            throw new InvalidDataException("Unexpected PNG content.");
        int stride = checked(width * 3);
        byte[] raw = new byte[checked((stride + 1) * height)];
        using MemoryStream input = new(compressed);
        using ZLibStream zlib = new(input, CompressionMode.Decompress);
        zlib.ReadExactly(raw);
        if (zlib.ReadByte() != -1)
            throw new InvalidDataException("PNG decompression exceeds declared shape.");
        byte[] pixels = new byte[checked(stride * height)];
        for (int y = 0; y < height; y++)
        {
            if (raw[y * (stride + 1)] != 0)
                throw new InvalidDataException("Unexpected PNG filter.");
            raw.AsSpan(y * (stride + 1) + 1, stride).CopyTo(pixels.AsSpan(y * stride));
        }

        double mean = pixels.Average(b => (double)b);
        return pixels.Average(b => (b - mean) * (b - mean)) > 1;
    }

    private static ReadOnlySpan<byte> Chunk(byte[] bytes, ref int offset, string expected)
    {
        if (offset > bytes.Length - 12)
            throw new InvalidDataException("Truncated PNG chunk.");
        int count = BinaryPrimitives.ReadInt32BigEndian(bytes.AsSpan(offset));
        if (count < 0 || count > bytes.Length - offset - 12 || Encoding.ASCII.GetString(bytes, offset + 4, 4) != expected)
            throw new InvalidDataException("Unexpected PNG chunk.");
        uint expectedCrc = BinaryPrimitives.ReadUInt32BigEndian(bytes.AsSpan(offset + 8 + count));
        if (PngEncoder.Crc32(bytes.AsSpan(offset + 4, count + 4)) != expectedCrc)
            throw new InvalidDataException("PNG chunk checksum mismatch.");
        ReadOnlySpan<byte> data = bytes.AsSpan(offset + 8, count);
        offset += count + 12;
        return data;
    }
}

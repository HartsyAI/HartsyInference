using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.LLM.Tests;

/// <summary>Reads the little-endian float32 <c>.npy</c> files the Python reference dumps write.</summary>
internal static class NpyReader
{
    /// <summary>Reads a little-endian float32 <c>.npy</c> (version 1 or 2 header, C order).</summary>
    public static (long[] Shape, float[] Data) ReadFloat32(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        int headerLength = bytes[8] | (bytes[9] << 8);
        int dataOffset = 10 + headerLength;
        string header = Encoding.ASCII.GetString(bytes, 10, headerLength);
        Assert.Contains("'<f4'", header);
        Assert.Contains("'fortran_order': False", header);

        Match shapeMatch = Regex.Match(header, @"'shape':\s*\(([^)]*)\)");
        long[] shape = shapeMatch.Groups[1].Value
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(long.Parse).ToArray();

        float[] data = new float[(bytes.Length - dataOffset) / sizeof(float)];
        Buffer.BlockCopy(bytes, dataOffset, data, 0, data.Length * sizeof(float));
        return (shape, data);
    }

    /// <summary>Reads a boolean <c>.npy</c> (<c>|b1</c>, C order).</summary>
    public static (long[] Shape, bool[] Data) ReadBool(string path)
    {
        (long[] shape, byte[] raw) = ReadRaw(path, "'|b1'");
        return (shape, [.. raw.Select(static b => b != 0)]);
    }

    /// <summary>Reads a little-endian int64 <c>.npy</c> (<c>&lt;i8</c>, C order).</summary>
    public static (long[] Shape, long[] Data) ReadInt64(string path)
    {
        (long[] shape, byte[] raw) = ReadRaw(path, "'<i8'");
        long[] data = new long[raw.Length / sizeof(long)];
        Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
        return (shape, data);
    }

    /// <summary>Reads a little-endian int32 <c>.npy</c> (<c>&lt;i4</c>, C order).</summary>
    public static (long[] Shape, int[] Data) ReadInt32(string path)
    {
        (long[] shape, byte[] raw) = ReadRaw(path, "'<i4'");
        int[] data = new int[raw.Length / sizeof(int)];
        Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
        return (shape, data);
    }

    private static (long[] Shape, byte[] Data) ReadRaw(string path, string descr)
    {
        byte[] bytes = File.ReadAllBytes(path);
        int headerLength = bytes[8] | (bytes[9] << 8);
        int dataOffset = 10 + headerLength;
        string header = Encoding.ASCII.GetString(bytes, 10, headerLength);
        Assert.Contains(descr, header);
        Assert.Contains("'fortran_order': False", header);
        Match shapeMatch = Regex.Match(header, @"'shape':\s*\(([^)]*)\)");
        long[] shape = shapeMatch.Groups[1].Value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(long.Parse).ToArray();
        return (shape, bytes[dataOffset..]);
    }
}

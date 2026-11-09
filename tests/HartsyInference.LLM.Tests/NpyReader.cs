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
}

using System.Globalization;
using System.Text.RegularExpressions;
using Xunit;

namespace HartsyInference.Audio.Tests.VoiceIsolator.Parity;

/// <summary>Reads a one-dimensional little-endian float32 <c>.npy</c> file.</summary>
internal static class NpyFloat32
{
    public static float[] Read(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        Assert.True(bytes.Length > 10 && bytes[0] == 0x93 && bytes[1] == (byte)'N', "not an .npy file");
        int headerStart = bytes[6] == 1 ? 10 : 12;
        int headerLength = bytes[6] == 1 ? BitConverter.ToUInt16(bytes, 8) : (int)BitConverter.ToUInt32(bytes, 8);
        int dataStart = headerStart + headerLength;
        string header = System.Text.Encoding.ASCII.GetString(bytes, headerStart, headerLength);
        Assert.Contains("'<f4'", header);
        Assert.Contains("'fortran_order': False", header);
        Match shape = Regex.Match(header, @"'shape': \((\d+),?\)");
        Assert.True(shape.Success, $"unexpected npy shape in {header}");
        int count = int.Parse(shape.Groups[1].Value, CultureInfo.InvariantCulture);
        Assert.Equal(count * 4, bytes.Length - dataStart);
        float[] values = new float[count];
        Buffer.BlockCopy(bytes, dataStart, values, 0, count * 4);
        return values;
    }
}

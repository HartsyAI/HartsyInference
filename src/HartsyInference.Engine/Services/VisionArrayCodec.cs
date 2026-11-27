using System.Buffers.Binary;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using HartsyInference.Core.Numerics;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Byte forms of a <see cref="VisionArray"/>: raw little-endian float32 (what the API base64-encodes next to
/// <c>shape</c> and <c>dtype</c>) and NumPy <c>.npy</c> v1.0 (what the CLI writes). Both carry the same payload bytes, so
/// a CLI file and an API response for one result decode to identical arrays.</summary>
public static class VisionArrayCodec
{
    /// <summary>The dtype name both forms declare.</summary>
    public const string Dtype = "float32";

    /// <summary>The values as little-endian float32 bytes, row-major.</summary>
    public static byte[] ToLittleEndianBytes(VisionArray array)
    {
        ArgumentNullException.ThrowIfNull(array);
        byte[] bytes = new byte[array.Data.Length * sizeof(float)];
        if (BitConverter.IsLittleEndian)
        {
            MemoryMarshal.AsBytes(array.Data.AsSpan()).CopyTo(bytes);
        }
        else
        {
            for (int i = 0; i < array.Data.Length; i++)
                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * sizeof(float)), array.Data[i]);
        }
        return bytes;
    }

    /// <summary>Reads little-endian float32 bytes back into an array of <paramref name="shape"/>.</summary>
    public static VisionArray FromLittleEndianBytes(ReadOnlySpan<byte> bytes, int[] shape)
    {
        if (bytes.Length % sizeof(float) != 0)
            throw new ArgumentException($"{bytes.Length} bytes is not a whole number of float32 values.", nameof(bytes));
        float[] data = new float[bytes.Length / sizeof(float)];
        for (int i = 0; i < data.Length; i++)
            data[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes[(i * sizeof(float))..]);
        return new VisionArray(data, shape);
    }

    /// <summary>The array as a NumPy <c>.npy</c> v1.0 file: <c>&lt;f4</c>, C order.</summary>
    public static byte[] ToNpy(VisionArray array)
    {
        ArgumentNullException.ThrowIfNull(array);
        return NpyFile.Float32(array.Data, array.Shape);
    }

    /// <summary>Reads a <c>.npy</c> v1.0 file written by <see cref="ToNpy"/> (little-endian float32, C order).</summary>
    public static VisionArray FromNpy(ReadOnlySpan<byte> file)
    {
        if (file.Length < 10 || !file[..6].SequenceEqual(NpyFile.Magic) || file[6] != 1)
            throw new ArgumentException("Not a .npy v1.0 file.", nameof(file));
        int headerLength = BinaryPrimitives.ReadUInt16LittleEndian(file[8..]);
        string header = Encoding.ASCII.GetString(file.Slice(10, headerLength));
        if (!header.Contains("'descr': '<f4'", StringComparison.Ordinal) || !header.Contains("'fortran_order': False", StringComparison.Ordinal))
            throw new ArgumentException($"Only little-endian float32 C-order arrays are read; header is {header.Trim()}.", nameof(file));
        int open = header.IndexOf('(', StringComparison.Ordinal), close = header.IndexOf(')', open);
        int[] shape = header[(open + 1)..close]
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(d => int.Parse(d, CultureInfo.InvariantCulture))
            .ToArray();
        return FromLittleEndianBytes(file[(10 + headerLength)..], shape);
    }
}

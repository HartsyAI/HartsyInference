using System.Buffers.Binary;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;

namespace HartsyInference.Core.Numerics;

/// <summary>NumPy <c>.npy</c> v1.0 files of little-endian float32 in C order.</summary>
public static class NpyFile
{
    /// <summary>The six bytes every <c>.npy</c> file starts with.</summary>
    public static ReadOnlySpan<byte> Magic => [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];

    /// <summary>The file bytes of <paramref name="values"/> under <paramref name="shape"/>; an empty shape is a scalar of one value.</summary>
    public static byte[] Float32(ReadOnlySpan<float> values, ReadOnlySpan<int> shape)
    {
        long count = 1;
        StringBuilder dimensions = new();
        foreach (int dimension in shape)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(dimension, nameof(shape));
            count *= dimension;
            if (dimensions.Length > 0)
                dimensions.Append(", ");
            dimensions.Append(dimension.ToString(CultureInfo.InvariantCulture));
        }
        if (count != values.Length)
            throw new ArgumentException($"Shape ({dimensions}) holds {count} values but the data has {values.Length}.", nameof(values));
        // Python spells a one-element tuple with a trailing comma.
        string dict = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({dimensions}{(shape.Length == 1 ? "," : "")}), }}";
        // magic (6) + version (2) + header length (2) + header, padded with spaces and a newline to a multiple of 64.
        int unpadded = Magic.Length + 2 + 2 + dict.Length + 1;
        int padding = (64 - unpadded % 64) % 64;
        string header = dict + new string(' ', padding) + "\n";
        int payload = Magic.Length + 4 + header.Length;
        byte[] file = new byte[payload + values.Length * sizeof(float)];
        Magic.CopyTo(file);
        file[6] = 1;
        file[7] = 0;
        BinaryPrimitives.WriteUInt16LittleEndian(file.AsSpan(8), checked((ushort)header.Length));
        Encoding.ASCII.GetBytes(header, file.AsSpan(10));
        if (BitConverter.IsLittleEndian)
        {
            MemoryMarshal.AsBytes(values).CopyTo(file.AsSpan(payload));
        }
        else
        {
            for (int i = 0; i < values.Length; i++)
                BinaryPrimitives.WriteSingleLittleEndian(file.AsSpan(payload + i * sizeof(float)), values[i]);
        }
        return file;
    }
}

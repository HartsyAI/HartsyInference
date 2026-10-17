using System.Text;
using System.Text.Json;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.SafeTensors;

/// <summary>Writes a safetensors file one tensor at a time. Each entry is materialized, written and released before the
/// next, so the size of the file does not bound host memory the way <see cref="SafeTensorsWriter.Save"/> does.</summary>
public static unsafe class SafeTensorsStreamWriter
{
    /// <summary>One tensor to write. <paramref name="Materialize"/> returns the tensor's bytes; <c>Owned</c> says whether the writer
    /// releases it after writing (a conversion) or leaves it alone (a source view).</summary>
    public readonly record struct Entry(string Name, DType DType, TensorShape Shape, Func<(Tensor Tensor, bool Owned)> Materialize);

    /// <summary>Writes <paramref name="entries"/> in order to <paramref name="path"/> through a staging file that is moved into place.</summary>
    public static void Write(string path, IReadOnlyList<Entry> entries, IReadOnlyDictionary<string, string>? metadata = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        ArgumentNullException.ThrowIfNull(entries);

        using MemoryStream headerStream = new();
        using (Utf8JsonWriter header = new(headerStream))
        {
            header.WriteStartObject();
            if (metadata is { Count: > 0 })
            {
                header.WritePropertyName("__metadata__");
                header.WriteStartObject();
                foreach (KeyValuePair<string, string> entry in metadata)
                {
                    header.WriteString(entry.Key, entry.Value);
                }
                header.WriteEndObject();
            }
            long offset = 0;
            foreach (Entry entry in entries)
            {
                long bytes = entry.DType.ComputeByteCount(entry.Shape.ElementCount);
                header.WritePropertyName(entry.Name);
                header.WriteStartObject();
                header.WriteString("dtype", SafeTensorName(entry.DType));
                header.WritePropertyName("shape");
                header.WriteStartArray();
                for (int d = 0; d < entry.Shape.Rank; d++)
                {
                    header.WriteNumberValue(entry.Shape[d]);
                }
                header.WriteEndArray();
                header.WritePropertyName("data_offsets");
                header.WriteStartArray();
                header.WriteNumberValue(offset);
                header.WriteNumberValue(offset + bytes);
                header.WriteEndArray();
                header.WriteEndObject();
                offset += bytes;
            }
            header.WriteEndObject();
        }

        // The header is padded with spaces to an 8-byte boundary, as the safetensors format expects.
        byte[] headerBytes = headerStream.ToArray();
        int padded = (headerBytes.Length + 7) & ~7;
        byte[] header8 = new byte[padded];
        Array.Fill(header8, (byte)' ');
        Buffer.BlockCopy(headerBytes, 0, header8, 0, headerBytes.Length);

        string staging = $"{path}.{Guid.NewGuid():N}.staging";
        try
        {
            using (FileStream file = new(staging, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1 << 20))
            {
                Span<byte> length = stackalloc byte[8];
                BitConverter.TryWriteBytes(length, (long)padded);
                file.Write(length);
                file.Write(header8, 0, header8.Length);
                foreach (Entry entry in entries)
                {
                    (Tensor tensor, bool owned) = entry.Materialize();
                    try
                    {
                        long bytes = entry.DType.ComputeByteCount(entry.Shape.ElementCount);
                        long available = tensor.DType.ComputeByteCount(tensor.ElementCount);
                        if (tensor.DType != entry.DType || available != bytes)
                        {
                            throw new InvalidOperationException(
                                $"Tensor '{entry.Name}' materialized as {tensor.DType} ({available} bytes), expected {entry.DType} ({bytes} bytes).");
                        }
                        file.Write(new ReadOnlySpan<byte>((void*)tensor.DataPointer, checked((int)bytes)));
                    }
                    finally
                    {
                        if (owned)
                        {
                            tensor.Dispose();
                        }
                    }
                }
            }
            File.Move(staging, path, overwrite: true);
        }
        finally
        {
            if (File.Exists(staging))
            {
                File.Delete(staging);
            }
        }
    }

    private static string SafeTensorName(DType dtype) => dtype == DType.F32 ? "F32"
        : dtype == DType.BF16 ? "BF16"
        : dtype == DType.F16 ? "F16"
        : throw new NotSupportedException($"Streaming safetensors writer does not support {dtype}.");
}

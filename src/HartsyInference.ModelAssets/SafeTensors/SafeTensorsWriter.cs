using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.SafeTensors;

public static class SafeTensorsWriter
{
    /// <summary>Layout is an 8-byte little-endian header length, the JSON header (dtype/shape/data_offsets per tensor), then the tensor bytes back-to-back at those offsets — no padding between tensors.</summary>
    /// <param name="metadata">Optional string map emitted as the header's <c>__metadata__</c> object (the spec allows string values only). Keys must be non-empty; values must be non-null.</param>
    public static unsafe void Save(string filePath, IReadOnlyDictionary<string, Tensor> tensors, IReadOnlyDictionary<string, string>? metadata = null)
    {
        // Build header JSON
        using MemoryStream headerStream = new MemoryStream();
        using Utf8JsonWriter writer = new Utf8JsonWriter(headerStream);
        writer.WriteStartObject();

        if (metadata is not null && metadata.Count > 0)
        {
            writer.WritePropertyName("__metadata__");
            writer.WriteStartObject();
            foreach (KeyValuePair<string, string> entry in metadata)
            {
                if (string.IsNullOrEmpty(entry.Key) || entry.Value is null)
                    throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"Invalid __metadata__ entry for '{filePath}': keys must be non-empty and values non-null (key='{entry.Key}').");
                writer.WriteString(entry.Key, entry.Value);
            }
            writer.WriteEndObject();
        }

        long currentOffset = 0;
        List<(string Name, Tensor Tensor, long Start, long End)> entries = [];

        foreach (KeyValuePair<string, Tensor> kvp in tensors)
        {
            long byteSize = Tensor.ComputeByteSize(kvp.Value.Shape, kvp.Value.DType);
            entries.Add((kvp.Key, kvp.Value, currentOffset, currentOffset + byteSize));
            currentOffset += byteSize;
        }

        foreach ((string name, Tensor tensor, long start, long end) in entries)
        {
            writer.WritePropertyName(name);
            writer.WriteStartObject();

            writer.WriteString("dtype", FormatDType(tensor.DType));

            writer.WritePropertyName("shape");
            writer.WriteStartArray();
            for (int i = 0; i < tensor.Shape.Rank; i++)
                writer.WriteNumberValue(tensor.Shape[i]);
            writer.WriteEndArray();

            writer.WritePropertyName("data_offsets");
            writer.WriteStartArray();
            writer.WriteNumberValue(start);
            writer.WriteNumberValue(end);
            writer.WriteEndArray();

            writer.WriteEndObject();
        }

        writer.WriteEndObject();
        writer.Flush();

        byte[] headerBytes = headerStream.ToArray();
        long headerLength = headerBytes.Length;

        // Write file: 8-byte header length + header JSON + tensor data
        using FileStream fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
        using BinaryWriter bw = new BinaryWriter(fs);

        bw.Write(headerLength);
        bw.Write(headerBytes);

        foreach ((string _, Tensor tensor, long _, long _) in entries)
        {
            long byteSize = Tensor.ComputeByteSize(tensor.Shape, tensor.DType);
            ReadOnlySpan<byte> data = new ReadOnlySpan<byte>(tensor.DataPointer, (int)byteSize);
            fs.Write(data);
        }
    }

    /// <summary>SHA-256 over the tensor bytes exactly as <see cref="Save"/> lays them out (enumeration order, back-to-back), i.e. the file's payload after the header. Matches SwarmUI's tensor-data <c>modelspec.hash_sha256</c> convention; callers prefix <c>0x</c>. Compute this BEFORE <see cref="Save"/> so the result can be embedded in the header without a second file pass.</summary>
    public static unsafe string ComputeTensorDataSha256(IReadOnlyDictionary<string, Tensor> tensors)
    {
        using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (KeyValuePair<string, Tensor> kvp in tensors)
        {
            long byteSize = Tensor.ComputeByteSize(kvp.Value.Shape, kvp.Value.DType);
            byte* data = (byte*)kvp.Value.DataPointer;
            long remaining = byteSize;
            while (remaining > 0)
            {
                int chunk = (int)Math.Min(remaining, int.MaxValue);
                hash.AppendData(new ReadOnlySpan<byte>(data, chunk));
                data += chunk;
                remaining -= chunk;
            }
        }
        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    }

    private static string FormatDType(DType dtype)
    {
        if (dtype.IsQuantized)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"Cannot write quantized dtype {dtype} to safetensors.");
        return dtype.Name;
    }
}

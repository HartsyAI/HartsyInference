using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
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
            // A single span is capped at int.MaxValue bytes; a large embedding table exceeds it.
            byte* data = (byte*)tensor.DataPointer;
            long remaining = Tensor.ComputeByteSize(tensor.Shape, tensor.DType);
            while (remaining > 0)
            {
                int chunk = (int)Math.Min(remaining, int.MaxValue);
                fs.Write(new ReadOnlySpan<byte>(data, chunk));
                data += chunk;
                remaining -= chunk;
            }
        }
    }

    /// <summary>Copies an existing safetensors file to <paramref name="outputPath"/> with its <c>__metadata__</c>
    /// replaced by the existing entries overlaid with <paramref name="metadata"/>, streaming the payload untouched.
    /// An empty <c>modelspec.hash_sha256</c> value is filled with <c>0x</c> + the payload SHA-256.</summary>
    /// <returns>Lowercase hex SHA-256 of the tensor payload (the bytes after the header).</returns>
    public static string RewriteMetadata(string sourcePath, string outputPath, IReadOnlyDictionary<string, string> metadata)
    {
        ArgumentNullException.ThrowIfNull(metadata);
        const string HashKey = "modelspec.hash_sha256";
        using FileStream source = new(sourcePath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20);
        Span<byte> lengthBytes = stackalloc byte[8];
        source.ReadExactly(lengthBytes);
        long headerLength = BitConverter.ToInt64(lengthBytes);
        if (headerLength <= 0 || headerLength > source.Length - 8)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"'{sourcePath}' is not a safetensors file (header length {headerLength}).");
        byte[] header = new byte[headerLength];
        source.ReadExactly(header);
        JsonObject root = JsonNode.Parse(header) as JsonObject
            ?? throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"'{sourcePath}' has a header that is not a JSON object.");
        Dictionary<string, string> merged = new(StringComparer.Ordinal);
        if (root["__metadata__"] is JsonObject existing)
        {
            foreach (KeyValuePair<string, JsonNode?> entry in existing)
                merged[entry.Key] = entry.Value?.GetValue<string>() ?? "";
        }
        foreach (KeyValuePair<string, string> entry in metadata)
            merged[entry.Key] = entry.Value;
        // A fixed-width placeholder keeps the header length stable, so the digest can be patched in after the copy.
        bool fillHash = merged.TryGetValue(HashKey, out string? slot) && string.IsNullOrEmpty(slot);
        if (fillHash)
            merged[HashKey] = "0x" + new string('0', 64);
        byte[] newHeader = BuildHeader(root, merged);
        string? outputDir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(outputDir))
            Directory.CreateDirectory(outputDir);
        string tempPath = outputPath + ".tmp";
        string payloadHash;
        try
        {
            using (FileStream output = new(tempPath, FileMode.Create, FileAccess.ReadWrite, FileShare.None, 1 << 20))
            {
                output.Write(BitConverter.GetBytes((long)newHeader.Length));
                output.Write(newHeader);
                using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                byte[] buffer = new byte[1 << 22];
                int read;
                while ((read = source.Read(buffer, 0, buffer.Length)) > 0)
                {
                    hash.AppendData(buffer, 0, read);
                    output.Write(buffer, 0, read);
                }
                payloadHash = Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
                if (fillHash)
                {
                    merged[HashKey] = "0x" + payloadHash;
                    byte[] patched = BuildHeader(root, merged);
                    output.Seek(8, SeekOrigin.Begin);
                    output.Write(patched);
                }
            }
            File.Move(tempPath, outputPath, overwrite: true);
        }
        catch
        {
            File.Delete(tempPath);
            throw;
        }
        return payloadHash;
    }

    private static byte[] BuildHeader(JsonObject root, IReadOnlyDictionary<string, string> metadata)
    {
        JsonObject header = new() { ["__metadata__"] = new JsonObject(metadata.Select(entry =>
            new KeyValuePair<string, JsonNode?>(entry.Key, JsonValue.Create(entry.Value)))) };
        foreach (KeyValuePair<string, JsonNode?> entry in root)
        {
            if (entry.Key != "__metadata__")
                header[entry.Key] = entry.Value?.DeepClone();
        }
        return Encoding.UTF8.GetBytes(header.ToJsonString());
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

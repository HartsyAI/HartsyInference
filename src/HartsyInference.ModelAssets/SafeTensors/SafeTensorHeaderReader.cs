using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using Microsoft.Win32.SafeHandles;

namespace HartsyInference.ModelAssets.SafeTensors;

/// <summary>Reads and validates a safetensors header from the first 8+N bytes only, so opening a shard never maps or reads its data.</summary>
/// <remarks>Validation is what lets later code hand out raw pointers: every tensor must lie inside the file, tensors
/// must not overlap, and each byte length must equal what its dtype and shape imply. A header failing any of those is
/// corrupt or truncated, and mapping it would end in a read past the mapping rather than an error message.</remarks>
public static class SafeTensorHeaderReader
{
    /// <summary>Largest header accepted by default; matches the reference safetensors implementation's cap.</summary>
    public const long DefaultMaxHeaderBytes = 100L << 20;

    /// <summary>Reads the header of <paramref name="path"/> with positional reads, without memory-mapping it.</summary>
    /// <param name="path">The safetensors file.</param>
    /// <param name="maxHeaderBytes">Largest JSON header accepted.</param>
    /// <param name="verifyByteLength">Whether each tensor's span must equal what its dtype and shape imply. Weight loading
    /// keeps it on; planners that inspect data-less header stubs turn it off.</param>
    public static SafeTensorHeader Read(string path, long maxHeaderBytes = DefaultMaxHeaderBytes, bool verifyByteLength = true)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Safetensors file not found: {path}", path);

        using SafeFileHandle handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        long fileLength = RandomAccess.GetLength(handle);
        if (fileLength < 8)
            throw new HartsyInferenceException($"Safetensors file '{path}' is {fileLength} bytes, too small for a header.");

        Span<byte> prefix = stackalloc byte[8];
        ReadFully(handle, prefix, 0, path);
        long headerLength = CheckHeaderLength(BinaryPrimitives.ReadUInt64LittleEndian(prefix), fileLength, maxHeaderBytes, path);

        byte[] json = new byte[headerLength];
        ReadFully(handle, json, 8, path);
        return Parse(json, headerLength, fileLength, path, verifyByteLength);
    }

    /// <summary>Validates the 8-byte length prefix against the file and cap, returning the JSON length.</summary>
    internal static long CheckHeaderLength(ulong declared, long fileLength, long maxHeaderBytes, string path)
    {
        if (declared > (ulong)maxHeaderBytes)
        {
            throw new HartsyInferenceException(
                $"Safetensors header of '{path}' declares {declared} bytes, over the {maxHeaderBytes}-byte limit. "
                + "The file is not safetensors or is corrupt.");
        }
        long headerLength = (long)declared;
        if (headerLength < 2 || headerLength > fileLength - 8)
        {
            throw new HartsyInferenceException(
                $"Invalid safetensors header length {declared} in '{path}' ({fileLength} bytes): the header must fit "
                + "inside the file. The file is truncated or is not safetensors.");
        }
        return headerLength;
    }

    /// <summary>Parses and validates header JSON for a file of <paramref name="fileLength"/> bytes.</summary>
    internal static SafeTensorHeader Parse(ReadOnlyMemory<byte> json, long headerLength, long fileLength, string path,
        bool verifyByteLength = true)
    {
        long dataStart = 8 + headerLength;
        long dataLength = fileLength - dataStart;
        Dictionary<string, SafeTensorDescriptor> tensors = new Dictionary<string, SafeTensorDescriptor>(StringComparer.Ordinal);
        Dictionary<string, string>? metadata = null;
        List<(long Start, long End, string Name)> extents = [];

        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(json);
        }
        catch (JsonException error)
        {
            throw new HartsyInferenceException($"Safetensors header of '{path}' is not valid JSON: {error.Message}", error);
        }

        using (document)
        {
            if (document.RootElement.ValueKind != JsonValueKind.Object)
                throw new HartsyInferenceException($"Safetensors header of '{path}' is not a JSON object.");

            foreach (JsonProperty property in document.RootElement.EnumerateObject())
            {
                if (property.Name == "__metadata__")
                {
                    metadata = ReadMetadata(property.Value, path);
                    continue;
                }
                if (tensors.ContainsKey(property.Name))
                    throw new HartsyInferenceException($"Safetensors header of '{path}' lists tensor '{property.Name}' twice.");

                SafeTensorDescriptor descriptor = ReadTensor(property.Name, property.Value, dataStart, dataLength, path, verifyByteLength);
                tensors.Add(property.Name, descriptor);
                extents.Add((descriptor.DataOffset - dataStart, descriptor.DataOffset - dataStart + descriptor.ByteLength, property.Name));
            }
        }

        extents.Sort(static (a, b) => a.Start != b.Start ? a.Start.CompareTo(b.Start) : a.End.CompareTo(b.End));
        for (int i = 1; i < extents.Count; i++)
        {
            if (extents[i].Start < extents[i - 1].End)
            {
                throw new HartsyInferenceException(
                    $"Safetensors header of '{path}': tensors '{extents[i - 1].Name}' [{extents[i - 1].Start}, "
                    + $"{extents[i - 1].End}) and '{extents[i].Name}' [{extents[i].Start}, {extents[i].End}) overlap.");
            }
        }

        return new SafeTensorHeader
        {
            Tensors = tensors,
            Metadata = metadata,
            HeaderLength = headerLength,
            DataStart = dataStart,
            FileLength = fileLength,
        };
    }

    private static Dictionary<string, string> ReadMetadata(JsonElement element, string path)
    {
        if (element.ValueKind != JsonValueKind.Object)
            throw new HartsyInferenceException($"Safetensors header of '{path}': __metadata__ is not an object.");
        Dictionary<string, string> metadata = [];
        foreach (JsonProperty entry in element.EnumerateObject())
        {
            // The spec restricts __metadata__ to string values; ignore anything else rather than fail the load.
            if (entry.Value.ValueKind == JsonValueKind.String)
                metadata[entry.Name] = entry.Value.GetString()!;
        }
        return metadata;
    }

    private static SafeTensorDescriptor ReadTensor(string name, JsonElement entry, long dataStart, long dataLength, string path,
        bool verifyByteLength)
    {
        if (entry.ValueKind != JsonValueKind.Object
            || !entry.TryGetProperty("dtype", out JsonElement dtypeElement) || dtypeElement.ValueKind != JsonValueKind.String
            || !entry.TryGetProperty("shape", out JsonElement shapeElement) || shapeElement.ValueKind != JsonValueKind.Array
            || !entry.TryGetProperty("data_offsets", out JsonElement offsetsElement) || offsetsElement.ValueKind != JsonValueKind.Array)
        {
            throw new HartsyInferenceException(
                $"Tensor '{name}' in '{path}' needs a string dtype, an array shape and an array data_offsets.");
        }

        string dtypeName = dtypeElement.GetString()!;
        if (!SafeTensorDTypes.TryParse(dtypeName, out DType dtype))
            throw new HartsyInferenceException($"Unsupported safetensors dtype: {dtypeName} (tensor '{name}' in '{path}')");

        int rank = shapeElement.GetArrayLength();
        if (rank > TensorShape.MaxRank)
            throw new HartsyInferenceException($"Tensor '{name}' in '{path}' has rank {rank}; the engine supports up to {TensorShape.MaxRank}.");
        long[] dims = new long[rank];
        long elements = 1;
        int index = 0;
        foreach (JsonElement dim in shapeElement.EnumerateArray())
        {
            if (dim.ValueKind != JsonValueKind.Number || !dim.TryGetInt64(out long value) || value < 0)
                throw new HartsyInferenceException($"Tensor '{name}' in '{path}' has a non-integer or negative dimension.");
            dims[index++] = value;
            long product;
            try
            {
                product = checked(elements * value);
            }
            catch (OverflowException)
            {
                throw new HartsyInferenceException($"Tensor '{name}' in '{path}' has a shape whose element count overflows 64 bits.");
            }
            elements = product;
        }

        if (offsetsElement.GetArrayLength() != 2)
            throw new HartsyInferenceException($"Tensor '{name}' in '{path}' needs exactly two data_offsets.");
        if (!TryReadOffset(offsetsElement[0], out long start) || !TryReadOffset(offsetsElement[1], out long end) || end < start)
            throw new HartsyInferenceException($"Tensor '{name}' in '{path}' has malformed data_offsets (need 0 <= start <= end).");
        if (end > dataLength)
        {
            throw new HartsyInferenceException(
                $"Tensor '{name}' in '{path}' ends at data offset {end} but the file holds only {dataLength} data bytes. "
                + "The file is truncated (an interrupted download?).");
        }

        if (elements > long.MaxValue >> 6)
            throw new HartsyInferenceException($"Tensor '{name}' in '{path}' has {elements} elements, more than any file can hold.");
        if (verifyByteLength)
            CheckByteLength(name, dtype, elements, dims, end - start, path);

        return new SafeTensorDescriptor
        {
            Name = name,
            DType = dtype,
            Shape = new TensorShape(dims.AsSpan()),
            DataOffset = dataStart + start,
            ByteLength = end - start,
        };
    }

    private static void CheckByteLength(string name, DType dtype, long elements, long[] dims, long actual, string path)
    {
        long expected;
        try
        {
            expected = dtype.ComputeByteCount(elements);
        }
        catch (Exception error) when (error is HartsyInferenceException or OverflowException)
        {
            throw new HartsyInferenceException($"Tensor '{name}' in '{path}': {error.Message}", error);
        }
        if (actual != expected)
        {
            throw new HartsyInferenceException(
                $"Tensor '{name}' in '{path}' spans {actual} bytes but {dtype.Name} {ShapeText(dims)} needs {expected}.");
        }
    }

    private static bool TryReadOffset(JsonElement element, out long value)
    {
        value = 0;
        return element.ValueKind == JsonValueKind.Number && element.TryGetInt64(out value) && value >= 0;
    }

    private static string ShapeText(long[] dims) => "[" + string.Join(", ", dims) + "]";

    private static void ReadFully(SafeFileHandle handle, Span<byte> destination, long offset, string path)
    {
        int done = 0;
        while (done < destination.Length)
        {
            int read = RandomAccess.Read(handle, destination[done..], offset + done);
            if (read == 0)
                throw new HartsyInferenceException($"'{path}' ended while its header was being read; the file is truncated.");
            done += read;
        }
    }
}

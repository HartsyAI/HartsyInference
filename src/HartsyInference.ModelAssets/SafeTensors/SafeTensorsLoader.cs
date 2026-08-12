using System.Text.Json;
using HartsyInference.Core.Memory;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.SafeTensors;

/// <summary>Loads tensors from safetensors files via memory-mapping. The file format is: 8-byte LE uint64 header length, then JSON header, then contiguous tensor data.</summary>
public sealed class SafeTensorsLoader : IDisposable
{
    private MmapHandle? _handle;
    private int _disposed;

    /// <summary>Tensor name → descriptor mapping from the parsed header.</summary>
    public IReadOnlyDictionary<string, SafeTensorDescriptor> Descriptors { get; private set; } = new Dictionary<string, SafeTensorDescriptor>();

    /// <summary>String entries of the header's <c>__metadata__</c> object, or null when the file has none. Values that are themselves JSON (e.g. LTX's <c>config</c>) are returned verbatim for the caller to parse.</summary>
    public IReadOnlyDictionary<string, string>? Metadata { get; private set; }

    /// <summary>File path of the loaded safetensors file.</summary>
    public string FilePath { get; private set; } = string.Empty;

    /// <summary>Loads a safetensors file, parsing the header and memory-mapping the data region.</summary>
    public unsafe void Load(string filePath)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        _handle?.Dispose();
        _handle = MmapHandle.OpenRead(filePath);
        FilePath = filePath;

        byte* basePtr = _handle.BasePointer;
        long fileLength = _handle.ByteLength;

        if (fileLength < 8)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"Safetensors file too small: {fileLength} bytes.");

        // Read 8-byte LE header length
        long headerLength = *(long*)basePtr;
        if (headerLength <= 0 || headerLength > fileLength - 8)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"Invalid safetensors header length: {headerLength}.");

        long dataOffset = 8 + headerLength;

        // Parse JSON header
        ReadOnlySpan<byte> headerJson = new ReadOnlySpan<byte>(basePtr + 8, (int)headerLength);
        Dictionary<string, SafeTensorDescriptor> descriptors = ParseHeader(headerJson, dataOffset, out Dictionary<string, string>? metadata);
        Descriptors = descriptors;
        Metadata = metadata;
    }

    /// <summary>Returns a tensor backed by memory-mapped data for the given tensor name. The returned tensor borrows memory from the mmap — do not dispose the loader while using it.</summary>
    public unsafe Tensor GetTensor(string name)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        if (_handle is null)
            throw new InvalidOperationException("No safetensors file loaded. Call Load() first.");

        if (!Descriptors.TryGetValue(name, out SafeTensorDescriptor? descriptor))
            throw new KeyNotFoundException($"Tensor '{name}' not found in safetensors file '{FilePath}'.");

        void* dataPtr = _handle.PointerAt(descriptor.DataOffset);
        Tensor tensor = new Tensor(dataPtr, descriptor.Shape, descriptor.DType);
        // Root the mmap so the borrowed pointer can't dangle if the caller keeps the tensor but drops the loader
        // reference (e.g. a helper that returns only GetAllTensors()). Explicit Dispose() still tears it down — the
        // documented "don't dispose the loader while using its tensors" contract is unchanged.
        tensor.SetKeepAlive(_handle);
        return tensor;
    }

    /// <summary>Returns all tensors as a dictionary of name → Tensor. All tensors borrow mmap memory.</summary>
    public Dictionary<string, Tensor> GetAllTensors()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        Dictionary<string, Tensor> tensors = new(Descriptors.Count);
        foreach (KeyValuePair<string, SafeTensorDescriptor> kvp in Descriptors)
        {
            tensors[kvp.Key] = GetTensor(kvp.Key);
        }
        return tensors;
    }

    private static Dictionary<string, SafeTensorDescriptor> ParseHeader(ReadOnlySpan<byte> headerJson, long dataBaseOffset, out Dictionary<string, string>? metadata)
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = [];
        metadata = null;
        using JsonDocument doc = JsonDocument.Parse(headerJson.ToArray());

        foreach (JsonProperty prop in doc.RootElement.EnumerateObject())
        {
            if (prop.Name == "__metadata__")
            {
                // The spec restricts __metadata__ to string values; ignore anything else rather than fail the load.
                metadata = [];
                foreach (JsonProperty entry in prop.Value.EnumerateObject())
                {
                    if (entry.Value.ValueKind == JsonValueKind.String)
                        metadata[entry.Name] = entry.Value.GetString()!;
                }
                continue;
            }

            JsonElement value = prop.Value;

            // Parse dtype
            string dtypeStr = value.GetProperty("dtype").GetString()!;
            DType dtype = ParseDType(dtypeStr);

            // Parse shape
            JsonElement shapeArray = value.GetProperty("shape");
            long[] dims = new long[shapeArray.GetArrayLength()];
            int idx = 0;
            foreach (JsonElement dim in shapeArray.EnumerateArray())
            {
                dims[idx++] = dim.GetInt64();
            }
            TensorShape shape = new TensorShape(dims.AsSpan());

            // Parse data offsets [start, end]
            JsonElement offsets = value.GetProperty("data_offsets");
            long start = offsets[0].GetInt64();
            long end = offsets[1].GetInt64();

            descriptors[prop.Name] = new SafeTensorDescriptor
            {
                Name = prop.Name,
                DType = dtype,
                Shape = shape,
                DataOffset = dataBaseOffset + start,
                ByteLength = end - start,
            };
        }

        return descriptors;
    }

    private static DType ParseDType(string dtype) => dtype switch
    {
        "F64" => DType.F64,
        "F32" => DType.F32,
        "F16" => DType.F16,
        "BF16" => DType.BF16,
        "F8_E4M3" => DType.F8E4M3,
        "F8_E5M2" => DType.F8E5M2,
        "I64" => DType.I64,
        "I32" => DType.I32,
        "I8" => DType.I8,
        "U8" => DType.U8,
        "BOOL" => DType.Bool,
        _ => throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"Unsupported safetensors dtype: {dtype}"),
    };

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        _handle?.Dispose();
        _handle = null;
    }
}

/// <summary>Describes a single tensor entry in a safetensors file header.</summary>
public sealed class SafeTensorDescriptor
{
    /// <summary>Tensor name from the header.</summary>
    public required string Name { get; init; }

    /// <summary>Data type of the tensor.</summary>
    public required DType DType { get; init; }

    /// <summary>Shape of the tensor.</summary>
    public required TensorShape Shape { get; init; }

    /// <summary>Byte offset from the start of the file to this tensor's data.</summary>
    public required long DataOffset { get; init; }

    /// <summary>Total byte size of this tensor's data.</summary>
    public required long ByteLength { get; init; }
}

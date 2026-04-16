using System.Text.Json;
using SharpInference.Core.Memory;
using SharpInference.Core.Tensors;

namespace SharpInference.ModelHandler.SafeTensors;

/// <summary>Loads tensors from safetensors files via memory-mapping. The file format is: 8-byte LE uint64 header length, then JSON header, then contiguous tensor data.</summary>
public sealed class SafeTensorsLoader : IDisposable
{
    private MmapHandle? _handle;
    private int _disposed;

    /// <summary>Tensor name → descriptor mapping from the parsed header.</summary>
    public IReadOnlyDictionary<string, SafeTensorDescriptor> Descriptors { get; private set; } = new Dictionary<string, SafeTensorDescriptor>();

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
            throw new SharpInference.Core.Exceptions.SharpInferenceException($"Safetensors file too small: {fileLength} bytes.");

        // Read 8-byte LE header length
        long headerLength = *(long*)basePtr;
        if (headerLength <= 0 || headerLength > fileLength - 8)
            throw new SharpInference.Core.Exceptions.SharpInferenceException($"Invalid safetensors header length: {headerLength}.");

        long dataOffset = 8 + headerLength;

        // Parse JSON header
        ReadOnlySpan<byte> headerJson = new ReadOnlySpan<byte>(basePtr + 8, (int)headerLength);
        Dictionary<string, SafeTensorDescriptor> descriptors = ParseHeader(headerJson, dataOffset);
        Descriptors = descriptors;
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
        return new Tensor(dataPtr, descriptor.Shape, descriptor.DType);
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

    private static Dictionary<string, SafeTensorDescriptor> ParseHeader(ReadOnlySpan<byte> headerJson, long dataBaseOffset)
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = [];
        using JsonDocument doc = JsonDocument.Parse(headerJson.ToArray());

        foreach (JsonProperty prop in doc.RootElement.EnumerateObject())
        {
            // Skip __metadata__ key
            if (prop.Name == "__metadata__")
                continue;

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
        "F32" => DType.F32,
        "F16" => DType.F16,
        "BF16" => DType.BF16,
        "I8" => DType.I8,
        "U8" => DType.U8,
        _ => throw new SharpInference.Core.Exceptions.SharpInferenceException($"Unsupported safetensors dtype: {dtype}"),
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

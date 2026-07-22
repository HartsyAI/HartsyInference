using HartsyInference.Core.Memory;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf;

/// <summary>Loads GGUF files via memory-mapping, parsing the header, metadata, and tensor descriptors. GGUF format: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8) + metadata + tensor_infos + padding + tensor_data.</summary>
public sealed class GgufLoader : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF" bytes 47 47 55 46 read as little-endian uint32
    private const int SupportedVersion = 3;

    private MmapHandle? _handle;
    private int _disposed;

    /// <summary>Parsed metadata from the GGUF header.</summary>
    public GgufMetadata Metadata { get; private set; } = new();

    /// <summary>Tensor name → descriptor mapping.</summary>
    public IReadOnlyDictionary<string, GgufTensorDescriptor> Descriptors { get; private set; } = new Dictionary<string, GgufTensorDescriptor>();

    /// <summary>File path of the loaded GGUF file.</summary>
    public string FilePath { get; private set; } = string.Empty;

    /// <summary>GGUF version number.</summary>
    public int Version { get; private set; }

    /// <summary>Loads a GGUF file, parsing the header, metadata, and tensor info.</summary>
    public unsafe void Load(string filePath)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        _handle?.Dispose();
        _handle = MmapHandle.OpenRead(filePath);
        FilePath = filePath;

        byte* ptr = _handle.BasePointer;
        long remaining = _handle.ByteLength;
        long offset = 0;

        // Magic
        if (remaining < 4)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException("GGUF file too small for magic number.");
        uint magic = *(uint*)(ptr + offset);
        if (magic != GgufMagic)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"Invalid GGUF magic: 0x{magic:X8}, expected 0x{GgufMagic:X8}.");
        offset += 4;

        // Version
        uint version = *(uint*)(ptr + offset);
        offset += 4;
        if (version < 2 || version > SupportedVersion)
            throw new HartsyInference.Core.Exceptions.UnsupportedModelException($"Unsupported GGUF version: {version}.", null, "GGUF");
        Version = (int)version;

        // Tensor count
        ulong tensorCount = *(ulong*)(ptr + offset);
        offset += 8;

        // Metadata KV count
        ulong metadataKvCount = *(ulong*)(ptr + offset);
        offset += 8;

        // Parse metadata
        GgufMetadata metadata = new GgufMetadata();
        for (ulong i = 0; i < metadataKvCount; i++)
        {
            (string key, object value, long bytesRead) = ReadMetadataKv(ptr + offset, remaining - offset);
            metadata.Add(key, value);
            offset += bytesRead;
        }
        Metadata = metadata;

        // Parse tensor infos
        Dictionary<string, GgufTensorDescriptor> descriptors = new((int)tensorCount);
        for (ulong i = 0; i < tensorCount; i++)
        {
            (GgufTensorDescriptor desc, long bytesRead) = ReadTensorInfo(ptr + offset, remaining - offset);
            descriptors[desc.Name] = desc;
            offset += bytesRead;
        }

        // Tensor data starts after alignment padding (typically 32 bytes)
        long alignment = (long)metadata.GetUInt64("general.alignment", 32);
        long dataStart = AlignUp(offset, alignment);

        // Adjust tensor offsets relative to file start
        foreach (GgufTensorDescriptor desc in descriptors.Values)
        {
            desc.AbsoluteOffset = dataStart + desc.RelativeOffset;
        }

        Descriptors = descriptors;
    }

    /// <summary>Returns a tensor backed by memory-mapped data for the given tensor name. For quantized tensors, call <see cref="GgufDequantizer"/> to convert to F16/F32.</summary>
    public unsafe Tensor GetTensor(string name)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        if (_handle is null)
            throw new InvalidOperationException("No GGUF file loaded. Call Load() first.");

        if (!Descriptors.TryGetValue(name, out GgufTensorDescriptor? desc))
            throw new KeyNotFoundException($"Tensor '{name}' not found in GGUF file '{FilePath}'.");

        void* dataPtr = _handle.PointerAt(desc.AbsoluteOffset);
        Tensor tensor = new Tensor(dataPtr, desc.Shape, desc.DType);
        // Root the mmap against premature GC/finalization while the borrowed pointer is in use (see SafeTensorsLoader).
        tensor.SetKeepAlive(_handle);
        return tensor;
    }

    private static unsafe (string Key, object Value, long BytesRead) ReadMetadataKv(byte* ptr, long remaining)
    {
        long offset = 0;

        // Key: string (length-prefixed)
        (string key, long keyBytes) = ReadGgufString(ptr + offset, remaining);
        offset += keyBytes;

        // Value type (uint32)
        uint valueType = *(uint*)(ptr + offset);
        offset += 4;

        // Value
        (object value, long valueBytes) = ReadMetadataValue(ptr + offset, remaining - offset, valueType);
        offset += valueBytes;

        return (key, value, offset);
    }

    private static unsafe (object Value, long BytesRead) ReadMetadataValue(byte* ptr, long remaining, uint valueType)
    {
        long offset = 0;

        switch (valueType)
        {
            case 0: // UINT8
                return (*(ptr + offset), 1);
            case 1: // INT8
                return ((sbyte)*(ptr + offset), 1);
            case 2: // UINT16
                return (*(ushort*)(ptr + offset), 2);
            case 3: // INT16
                return (*(short*)(ptr + offset), 2);
            case 4: // UINT32
                return (*(uint*)(ptr + offset), 4);
            case 5: // INT32
                return (*(int*)(ptr + offset), 4);
            case 6: // FLOAT32
                return (*(float*)(ptr + offset), 4);
            case 7: // BOOL
                return (*(ptr + offset) != 0, 1);
            case 8: // STRING
            {
                (string s, long bytes) = ReadGgufString(ptr + offset, remaining);
                return (s, bytes);
            }
            case 9: // ARRAY
            {
                uint elemType = *(uint*)(ptr + offset);
                offset += 4;
                ulong count = *(ulong*)(ptr + offset);
                offset += 8;

                // For string arrays, collect values
                if (elemType == 8)
                {
                    string[] arr = new string[count];
                    for (ulong i = 0; i < count; i++)
                    {
                        (string s, long bytes) = ReadGgufString(ptr + offset, remaining - offset);
                        arr[i] = s;
                        offset += bytes;
                    }
                    return (arr, offset);
                }

                // Integer arrays (UINT32/INT32) — needed for e.g. tokenizer.ggml.token_type. Collect as int[].
                if (elemType == 4 || elemType == 5)
                {
                    int[] ints = new int[count];
                    for (ulong i = 0; i < count; i++)
                    {
                        ints[i] = *(int*)(ptr + offset);
                        offset += 4;
                    }
                    return (ints, offset);
                }

                // Float32 arrays — needed for SentencePiece tokenizer.ggml.scores. Collect as float[].
                if (elemType == 6)
                {
                    float[] floats = new float[count];
                    for (ulong i = 0; i < count; i++)
                    {
                        floats[i] = *(float*)(ptr + offset);
                        offset += 4;
                    }
                    return (floats, offset);
                }

                // Bool arrays — needed for Gemma-4's per-layer attention.sliding_window_pattern (llama.cpp's
                // get_key_or_arr can store this as a real per-layer array, not just a broadcast scalar; silently
                // dropping it here previously made every layer read as "global", corrupting per-layer head-dim
                // sizing for architectures that actually rely on a real array).
                if (elemType == 7)
                {
                    bool[] bools = new bool[count];
                    for (ulong i = 0; i < count; i++)
                    {
                        bools[i] = *(ptr + offset) != 0;
                        offset += 1;
                    }
                    return (bools, offset);
                }

                // For other arrays, skip the data
                for (ulong i = 0; i < count; i++)
                {
                    (_, long bytes) = ReadMetadataValue(ptr + offset, remaining - offset, elemType);
                    offset += bytes;
                }
                return (Array.Empty<object>(), offset);
            }
            case 10: // UINT64
                return (*(ulong*)(ptr + offset), 8);
            case 11: // INT64
                return (*(long*)(ptr + offset), 8);
            case 12: // FLOAT64
                return (*(double*)(ptr + offset), 8);
            default:
                throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"Unknown GGUF metadata value type: {valueType}.");
        }
    }

    private static unsafe (GgufTensorDescriptor Descriptor, long BytesRead) ReadTensorInfo(byte* ptr, long remaining)
    {
        long offset = 0;

        // Name
        (string name, long nameBytes) = ReadGgufString(ptr + offset, remaining);
        offset += nameBytes;

        // Number of dimensions (uint32)
        uint nDims = *(uint*)(ptr + offset);
        offset += 4;

        // Dimension sizes (uint64 each)
        long[] dims = new long[nDims];
        for (int i = 0; i < (int)nDims; i++)
        {
            dims[i] = (long)(*(ulong*)(ptr + offset));
            offset += 8;
        }

        // Type (uint32 → GgufDType → our DType)
        uint ggufType = *(uint*)(ptr + offset);
        offset += 4;

        // Offset within the data section (uint64)
        ulong relativeOffset = *(ulong*)(ptr + offset);
        offset += 8;

        DType dtype = MapGgufType(ggufType);
        TensorShape shape = new TensorShape(dims.AsSpan());

        GgufTensorDescriptor desc = new GgufTensorDescriptor
        {
            Name = name,
            DType = dtype,
            Shape = shape,
            GgufTypeId = ggufType,
            RelativeOffset = (long)relativeOffset,
            AbsoluteOffset = 0, // set later after alignment
        };

        return (desc, offset);
    }

    private static unsafe (string Value, long BytesRead) ReadGgufString(byte* ptr, long remaining)
    {
        if (remaining < 8)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException("GGUF string: not enough bytes for length prefix.");

        ulong length = *(ulong*)ptr;
        if ((long)length > remaining - 8)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"GGUF string: length {length} exceeds remaining bytes {remaining - 8}.");

        string value = System.Text.Encoding.UTF8.GetString(ptr + 8, (int)length);
        return (value, 8 + (long)length);
    }

    /// <summary>Maps GGUF/ggml tensor type IDs to <see cref="DType"/>. IDs follow `enum ggml_type` in ggml.h. Variants like Q4_K_S/M/L share the same block layout (id=12) — the per-tensor mix policy is encoded in the file metadata, not the type ID.
    ///
    /// <para>Note: <see cref="DType"/> registration ≠ codec availability. A type ID can be mapped here while the corresponding codec in <see cref="Codecs.GgufCodecRegistry"/> is still pending. Caller will get an explicit "no codec registered" error from the registry rather than a silent corruption.</para></summary>
    private static DType MapGgufType(uint ggufType) => ggufType switch
    {
        0 => DType.F32,
        1 => DType.F16,
        2 => DType.Q4_0,
        3 => DType.Q4_1,
        6 => DType.Q5_0,
        7 => DType.Q5_1,
        8 => DType.Q8_0,
        9 => DType.Q8_1,
        10 => DType.Q2_K,
        11 => DType.Q3_K,
        12 => DType.Q4_K,
        13 => DType.Q5_K,
        14 => DType.Q6_K,
        15 => DType.Q8_K,
        16 => DType.IQ2_XXS,
        17 => DType.IQ2_XS,
        18 => DType.IQ3_XXS,
        19 => DType.IQ1_S,
        20 => DType.IQ4_NL,
        21 => DType.IQ3_S,
        22 => DType.IQ2_S,
        23 => DType.IQ4_XS,
        29 => DType.IQ1_M,
        30 => DType.BF16,
        31 => DType.TQ1_0,
        32 => DType.TQ2_0,
        39 => DType.MXFP4,
        _ => throw new HartsyInference.Core.Exceptions.HartsyInferenceException($"Unsupported GGUF tensor type: {ggufType}. See ggml.h enum ggml_type."),
    };

    private static long AlignUp(long value, long alignment)
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        _handle?.Dispose();
        _handle = null;
    }
}

/// <summary>Describes a single tensor entry in a GGUF file.</summary>
public sealed class GgufTensorDescriptor
{
    /// <summary>Tensor name.</summary>
    public required string Name { get; init; }

    /// <summary>HartsyInference data type (mapped from GGUF type ID).</summary>
    public required DType DType { get; init; }

    /// <summary>Shape of the tensor.</summary>
    public required TensorShape Shape { get; init; }

    /// <summary>Original GGUF type ID for dequantization lookup.</summary>
    public required uint GgufTypeId { get; init; }

    /// <summary>Offset relative to the start of the data section.</summary>
    public required long RelativeOffset { get; init; }

    /// <summary>Absolute offset from file start (set after alignment calculation).</summary>
    public long AbsoluteOffset { get; internal set; }
}

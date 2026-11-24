using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;

namespace HartsyInference.ModelAssets.Dcp;

/// <summary>The tensor table of a PyTorch Distributed Checkpoint, read from its <c>.metadata</c> pickle by the allow-list <see cref="DcpMetadataUnpickler"/>. Only unsharded tensors (one chunk covering the whole shape) are supported; bytes entries are skipped.</summary>
public sealed class DcpMetadata
{
    private DcpMetadata(IReadOnlyDictionary<string, DcpTensorEntry> tensors) => Tensors = tensors;

    /// <summary>Tensor key to its entry, in the order the metadata lists them.</summary>
    public IReadOnlyDictionary<string, DcpTensorEntry> Tensors { get; }

    /// <summary>Parses the bytes of a <c>.metadata</c> file.</summary>
    public static DcpMetadata Parse(byte[] pickle)
    {
        ArgumentNullException.ThrowIfNull(pickle);
        object? root = new DcpMetadataUnpickler(pickle).Run();
        if (root is not DcpObject { Name: "Metadata" } metadata)
            throw new InvalidDataException("The file is not a distributed-checkpoint metadata pickle (no Metadata root).");
        DcpDict states = metadata.Field<DcpDict>("state_dict_metadata")
            ?? throw new InvalidDataException("The metadata has no state_dict_metadata.");
        DcpDict storage = metadata.Field<DcpDict>("storage_data")
            ?? throw new InvalidDataException("The metadata has no storage_data; the checkpoint was not written by a file-system writer.");

        Dictionary<string, List<(long[]? Offsets, DcpObject Info)>> infoByKey = new(StringComparer.Ordinal);
        foreach (KeyValuePair<object?, object?> entry in storage.Entries)
        {
            if (entry.Key is not DcpObject index || entry.Value is not DcpObject info) continue;
            if (index.Fields.GetValueOrDefault("fqn") is not string fqn) continue;
            if (!infoByKey.TryGetValue(fqn, out List<(long[]?, DcpObject)>? list)) infoByKey[fqn] = list = [];
            list.Add((index.Fields.GetValueOrDefault("offset") as long[], info));
        }

        Dictionary<string, DcpTensorEntry> tensors = new(states.Entries.Count, StringComparer.Ordinal);
        foreach (KeyValuePair<object?, object?> entry in states.Entries)
        {
            if (entry.Key is not string key || entry.Value is not DcpObject { Name: "TensorStorageMetadata" } meta) continue;
            long[] shape = meta.Field<long[]>("size") ?? throw new InvalidDataException($"Tensor '{key}' has no size.");
            DType dtype = ReadDType(key, meta);
            List<object?> chunks = meta.Field<List<object?>>("chunks") ?? throw new InvalidDataException($"Tensor '{key}' has no chunks.");
            if (chunks.Count != 1 || chunks[0] is not DcpObject chunk
                || !Covers(chunk.Fields.GetValueOrDefault("offsets") as long[], chunk.Fields.GetValueOrDefault("sizes") as long[], shape))
                throw new NotSupportedException($"Tensor '{key}' is stored in {chunks.Count} chunk(s); only whole-tensor chunks are supported.");
            if (!infoByKey.TryGetValue(key, out List<(long[]? Offsets, DcpObject Info)>? candidates) || candidates.Count != 1)
                throw new InvalidDataException($"Tensor '{key}' has {candidates?.Count ?? 0} storage records, expected exactly one.");

            DcpObject storageInfo = candidates[0].Info;
            string path = storageInfo.Fields.GetValueOrDefault("relative_path") as string
                ?? throw new InvalidDataException($"Tensor '{key}' has no relative_path.");
            if (storageInfo.Fields.GetValueOrDefault("offset") is not long offset || storageInfo.Fields.GetValueOrDefault("length") is not long length
                || offset < 0 || length <= 0)
                throw new InvalidDataException($"Tensor '{key}' has an invalid byte range.");
            if (Path.IsPathRooted(path) || path.Contains("..", StringComparison.Ordinal))
                throw new InvalidDataException($"Tensor '{key}' names a storage file outside the checkpoint folder ('{path}').");
            tensors[key] = new DcpTensorEntry(key, dtype, shape, path, offset, length);
        }
        return new DcpMetadata(tensors);
    }

    private static bool Covers(long[]? offsets, long[]? sizes, long[] shape)
    {
        if (offsets is null || sizes is null || offsets.Length != shape.Length || sizes.Length != shape.Length) return false;
        for (int d = 0; d < shape.Length; d++)
            if (offsets[d] != 0 || sizes[d] != shape[d]) return false;
        return true;
    }

    private static DType ReadDType(string key, DcpObject meta)
    {
        DcpObject? properties = meta.Field<DcpObject>("properties");
        PickleGlobal? dtype = properties?.Fields.GetValueOrDefault("dtype") as PickleGlobal;
        return dtype?.Name switch
        {
            "float32" => DType.F32,
            "float16" => DType.F16,
            "bfloat16" => DType.BF16,
            "float64" => DType.F64,
            "int64" => DType.I64,
            "int32" => DType.I32,
            "int8" => DType.I8,
            "uint8" => DType.U8,
            "bool" => DType.Bool,
            _ => throw new NotSupportedException($"Tensor '{key}' has unsupported dtype '{dtype?.Name ?? "<missing>"}'."),
        };
    }
}

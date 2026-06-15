using System.IO.Compression;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelHandler.PyTorch;

/// <summary>Loads a PyTorch <c>.pt</c> / <c>.pth</c> checkpoint (the modern <c>torch.save</c> ZIP container) into a
/// <c>Dictionary&lt;string, Tensor&gt;</c>, mirroring <see cref="SafeTensors.SafeTensorsLoader"/>'s surface so any
/// model loader can consume <c>.pt</c>-only checkpoints (Hunyuan-GameCraft, Cosmos, …). Implements a
/// <b>safe-subset</b> pickle VM (no arbitrary code execution): it recognizes only the opcodes and the two torch
/// reduce functions (<c>torch._utils._rebuild_tensor_v2</c> + storage <c>persistent_id</c>) that a flat
/// <c>state_dict</c> emits. Tensors are materialized into owned native memory (one copy), so the loader can be
/// disposed immediately after <see cref="GetAllTensors"/>.
/// <para><b>Validation-gated:</b> exact storage-type → dtype coverage and any non-contiguous/shared-storage edge
/// cases are confirmed against a real checkpoint dump. Quantized storages are not supported (weights are
/// F32/F16/BF16/int).</para></summary>
public sealed class PytorchPickleLoader : IDisposable
{
    private readonly Dictionary<string, Tensor> _tensors = [];
    private int _disposed;

    /// <summary>Tensor name → metadata (shape/dtype) parsed from the pickle.</summary>
    public IReadOnlyDictionary<string, PytorchTensorDescriptor> Descriptors { get; private set; }
        = new Dictionary<string, PytorchTensorDescriptor>();

    /// <summary>Path of the loaded checkpoint.</summary>
    public string FilePath { get; private set; } = string.Empty;

    /// <summary>Opens the <c>.pt</c> ZIP, parses <c>data.pkl</c>, and materializes every tensor into owned memory.</summary>
    public unsafe void Load(string filePath)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        FilePath = filePath;

        using ZipArchive zip = ZipFile.OpenRead(filePath);
        ZipArchiveEntry pkl = FindEntry(zip, "/data.pkl", "data.pkl")
            ?? throw new InvalidOperationException($"'{filePath}' is not a torch ZIP checkpoint (no data.pkl).");

        byte[] pickle;
        using (Stream s = pkl.Open()) { using MemoryStream ms = new(); s.CopyTo(ms); pickle = ms.ToArray(); }

        object? root = new PickleMachine(pickle).Run();
        Dictionary<string, PickleTensor> metas = [];
        FlattenStateDict(root, metas);

        Dictionary<string, PytorchTensorDescriptor> descriptors = new(metas.Count);
        foreach ((string name, PickleTensor meta) in metas)
        {
            TensorShape shape = new(meta.Size);
            Tensor t = new(shape, meta.Storage.DType);
            ReadStorageInto(zip, meta, (byte*)t.DataPointer, shape.ElementCount * meta.Storage.DType.SizeInBytes);
            _tensors[name] = t;
            descriptors[name] = new PytorchTensorDescriptor { Name = name, DType = meta.Storage.DType, Shape = shape };
        }
        Descriptors = descriptors;
    }

    /// <summary>All tensors as name → owned Tensor. Valid until <see cref="Dispose"/>.</summary>
    public Dictionary<string, Tensor> GetAllTensors()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return new Dictionary<string, Tensor>(_tensors);
    }

    private static unsafe void ReadStorageInto(ZipArchive zip, PickleTensor meta, byte* dst, long byteCount)
    {
        ZipArchiveEntry entry = FindEntry(zip, "/data/" + meta.Storage.Key, "data/" + meta.Storage.Key)
            ?? throw new InvalidOperationException($"Missing storage blob 'data/{meta.Storage.Key}'.");
        long skip = meta.Offset * meta.Storage.DType.SizeInBytes;
        using Stream s = entry.Open();
        const int BufSize = 81920;
        byte* buf = stackalloc byte[BufSize];
        Span<byte> span = new(buf, BufSize);
        // Skip the storage_offset prefix (usually 0 for a flat state_dict).
        while (skip > 0)
        {
            int got = s.Read(span[..(int)Math.Min(skip, BufSize)]);
            if (got <= 0) throw new EndOfStreamException("Truncated storage blob (skip).");
            skip -= got;
        }
        long pos = 0, remaining = byteCount;
        while (remaining > 0)
        {
            int got = s.Read(span[..(int)Math.Min(remaining, BufSize)]);
            if (got <= 0) throw new EndOfStreamException("Truncated storage blob (data).");
            Buffer.MemoryCopy(buf, dst + pos, got, got);
            pos += got;
            remaining -= got;
        }
    }

    private static ZipArchiveEntry? FindEntry(ZipArchive zip, string suffix, string exact)
    {
        foreach (ZipArchiveEntry e in zip.Entries)
            if (e.FullName == exact || e.FullName.EndsWith(suffix, StringComparison.Ordinal))
                return e;
        return null;
    }

    /// <summary>Collects every <see cref="PickleTensor"/> in the unpickled object, descending one wrapper level
    /// (e.g. a <c>{"module": {...}}</c> or <c>{"state_dict": {...}}</c> envelope) if the top level holds no tensors.</summary>
    private static void FlattenStateDict(object? root, Dictionary<string, PickleTensor> outMap)
    {
        if (root is not Dictionary<string, object?> dict)
            throw new InvalidOperationException("Checkpoint root is not a dict/state_dict.");

        bool anyTop = false;
        foreach ((string k, object? v) in dict)
            if (v is PickleTensor t) { outMap[k] = t; anyTop = true; }

        if (anyTop) return;
        // No tensors at the top — descend into the first dict-valued wrapper.
        foreach ((_, object? v) in dict)
            if (v is Dictionary<string, object?> inner)
            {
                foreach ((string ik, object? iv) in inner)
                    if (iv is PickleTensor it) outMap[ik] = it;
                if (outMap.Count > 0) return;
            }
        if (outMap.Count == 0) throw new InvalidOperationException("No tensors found in checkpoint.");
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in _tensors.Values) t.Dispose();
        _tensors.Clear();
    }
}

/// <summary>Shape/dtype descriptor for a tensor in a <c>.pt</c> checkpoint.</summary>
public sealed class PytorchTensorDescriptor
{
    public required string Name { get; init; }
    public required DType DType { get; init; }
    public required TensorShape Shape { get; init; }
}

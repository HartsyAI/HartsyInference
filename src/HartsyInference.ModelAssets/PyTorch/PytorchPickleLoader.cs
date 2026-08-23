using System.IO.Compression;
using System.IO.MemoryMappedFiles;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.PyTorch;

/// <summary>Loads a PyTorch <c>.pt</c> / <c>.pth</c> checkpoint (the modern <c>torch.save</c> ZIP container) into a <c>Dictionary&lt;string, Tensor&gt;</c>, mirroring <see cref="SafeTensors.SafeTensorsLoader"/>'s surface so any model loader can consume <c>.pt</c>-only checkpoints (Hunyuan-GameCraft, Cosmos, …). Implements a <b>safe-subset</b> pickle VM (no arbitrary code execution): it recognizes only the opcodes and the two torch reduce functions (<c>torch._utils._rebuild_tensor_v2</c> + storage <c>persistent_id</c>) that a flat <c>state_dict</c> emits. Tensors are materialized into owned native memory (one copy), so the loader can be disposed immediately after <see cref="GetAllTensors"/>.
/// <para><b>Validation-gated:</b> exact storage-type → dtype coverage and any non-contiguous/shared-storage edge cases are confirmed against a real checkpoint dump. Quantized storages are not supported (weights are F32/F16/BF16/int).</para></summary>
public sealed class PytorchPickleLoader : IDisposable
{
    private readonly Dictionary<string, Tensor> _tensors = [];
    private int _disposed;

    /// <summary>Tensor name → metadata (shape/dtype) parsed from the pickle.</summary>
    public IReadOnlyDictionary<string, PytorchTensorDescriptor> Descriptors { get; private set; }
        = new Dictionary<string, PytorchTensorDescriptor>();

    /// <summary>Path of the loaded checkpoint.</summary>
    public string FilePath { get; private set; } = string.Empty;

    /// <summary>Opens the <c>.pt</c> ZIP, parses <c>data.pkl</c>, and materializes every tensor into owned memory.
    /// <para>When <paramref name="recursiveFlatten"/> is true, a multi-module checkpoint — a dict whose values are themselves state-dicts (e.g. Kokoro's <c>{bert:{…}, predictor:{…}, decoder:{…}}</c>) — is flattened into fully-qualified dotted keys (<c>bert.embeddings.weight</c>, …) covering EVERY module. The default (false) keeps the legacy behavior: load top-level tensors, else descend one wrapper envelope.</para></summary>
    public void Load(string filePath, bool recursiveFlatten = false)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        FilePath = filePath;

        // Sniff the container: "PK\x03\x04" = modern torch.save ZIP; 0x80 = a raw pickle stream (legacy pre-2019
        // torch.save). AudioGen/MusicGen ship their T5 text encoder in the legacy format.
        Span<byte> magic = stackalloc byte[2];
        using (FileStream fs = File.OpenRead(filePath))
            if (fs.Read(magic) < 2) throw new InvalidOperationException($"'{filePath}' is too small to be a torch checkpoint.");

        if (magic[0] == 0x50 && magic[1] == 0x4b) { LoadZip(filePath, recursiveFlatten); return; }        // "PK"
        if (magic[0] == 0x80) { LoadLegacy(filePath, recursiveFlatten); return; }                          // pickle PROTO
        throw new InvalidOperationException($"'{filePath}' is not a recognized torch checkpoint (magic 0x{magic[0]:x2}{magic[1]:x2}).");
    }

    /// <summary>Modern <c>torch.save</c> path: a ZIP whose <c>data.pkl</c> holds the pickle and whose <c>data/&lt;key&gt;</c> entries hold each storage's raw bytes.</summary>
    private unsafe void LoadZip(string filePath, bool recursiveFlatten)
    {
        using ZipArchive zip = ZipFile.OpenRead(filePath);
        ZipArchiveEntry pkl = FindEntry(zip, "/data.pkl", "data.pkl")
            ?? throw new InvalidOperationException($"'{filePath}' is not a torch ZIP checkpoint (no data.pkl).");

        byte[] pickle;
        using (Stream s = pkl.Open()) { using MemoryStream ms = new(); s.CopyTo(ms); pickle = ms.ToArray(); }

        object? root = new PickleMachine(pickle).Run();
        Dictionary<string, PickleTensor> metas = [];
        if (recursiveFlatten)
        {
            FlattenStateDictRecursive(root, "", metas);
            if (metas.Count == 0) throw new InvalidOperationException("No tensors found in checkpoint.");
        }
        else
        {
            FlattenStateDict(root, metas);
        }

        Dictionary<string, PytorchTensorDescriptor> descriptors = new(metas.Count);
        foreach ((string name, PickleTensor meta) in metas)
        {
            TensorShape shape = new(meta.Size);
            Tensor t = new(shape, meta.Storage.DType);
            ReadStorageInto(zip, meta, (byte*)t.DataPointer, shape.ElementCount * meta.Storage.DType.SizeInBytes);
            MakeRowMajor(t, meta.Size, meta.Stride, meta.Storage.DType);
            _tensors[name] = t;
            descriptors[name] = new PytorchTensorDescriptor { Name = name, DType = meta.Storage.DType, Shape = shape };
        }
        Descriptors = descriptors;
    }

    /// <summary>Legacy <c>torch.save</c> path (<c>_legacy_save</c>): five back-to-back pickle streams (MAGIC_NUMBER, PROTOCOL_VERSION, sys_info, the state_dict, then the storage-key list) followed by each storage's raw bytes in key-list order, prefixed by an <c>int64</c> element count. No ZIP, no per-storage alignment. The whole file is read into memory (legacy checkpoints predate the &gt;2 GB era; .NET caps a single read at ~2 GB, which this format never exceeds in practice).</summary>
    private unsafe void LoadLegacy(string filePath, bool recursiveFlatten)
    {
        // Memory-map instead of File.ReadAllBytes: a legacy T5 encoder (e.g. t5-large's 2.9 GB pytorch_model.bin)
        // exceeds the 2 GB managed-array cap ReadAllBytes enforces. The mapped pointer has no such limit and the
        // storage-table walk below needs 64-bit offsets into the multi-GB tail anyway.
        long length = new FileInfo(filePath).Length;
        using MemoryMappedFile mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
        using MemoryMappedViewAccessor acc = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        byte* pAll = null;
        acc.SafeMemoryMappedViewHandle.AcquirePointer(ref pAll);
        try
        {
            LoadLegacyCore(pAll, length, recursiveFlatten);
        }
        finally
        {
            acc.SafeMemoryMappedViewHandle.ReleasePointer();
        }
    }

    private unsafe void LoadLegacyCore(byte* pAll, long length, bool recursiveFlatten)
    {
        // The five pickle streams (metadata + storage-key list) live at the very start and are small; copy just that
        // prefix into a managed buffer so the byte[]-based pickle VM can parse them. 256 MB is far beyond any real
        // checkpoint's metadata yet safely under the 2 GB managed-array cap.
        int headLen = (int)Math.Min(length, 256L * 1024 * 1024);
        byte[] head = new byte[headLen];
        fixed (byte* pHead = head) Buffer.MemoryCopy(pAll, pHead, headLen, headLen);

        // Five sequential pickle streams over the SAME buffer; each Run() stops at its '.' opcode and reports the
        // offset just past it, which seeds the next stream.
        int off = 0;
        off = RunPickle(head, off, out _);                                                  // 1: MAGIC_NUMBER
        off = RunPickle(head, off, out _);                                                  // 2: PROTOCOL_VERSION
        off = RunPickle(head, off, out _);                                                  // 3: sys_info
        off = RunPickle(head, off, out object? root);                                       // 4: state_dict
        off = RunPickle(head, off, out object? keysObj);                                    // 5: storage-key list
        if (off > headLen) throw new InvalidOperationException("Legacy checkpoint metadata exceeds the 256 MB header window.");

        Dictionary<string, PickleTensor> metas = [];
        if (recursiveFlatten)
        {
            FlattenStateDictRecursive(root, "", metas);
            if (metas.Count == 0) throw new InvalidOperationException("No tensors found in checkpoint.");
        }
        else
        {
            FlattenStateDict(root, metas);
        }

        if (keysObj is not List<object?> keyList)
            throw new InvalidOperationException("Legacy checkpoint is missing its storage-key list (stream 5).");

        // dtype/element-size per storage key comes from the tensors that reference it.
        Dictionary<string, DType> keyDType = [];
        foreach (PickleTensor mt in metas.Values) keyDType[mt.Storage.Key] = mt.Storage.DType;

        // Walk the appended storage region once, recording where each key's data bytes start. Each record is an
        // int64 element count followed by numel * element_size raw bytes; records are packed with no padding.
        Dictionary<string, long> keyDataOffset = new(keyList.Count);
        long p = off;
        foreach (object? ko in keyList)
        {
            string key = ko as string ?? throw new InvalidOperationException("Legacy storage key is not a string.");
            if (p + 8 > length) throw new EndOfStreamException("Truncated legacy storage table.");
            long numel = *(long*)(pAll + p);
            p += 8;
            keyDataOffset[key] = p;
            if (!keyDType.TryGetValue(key, out DType dt))
                throw new InvalidOperationException($"Legacy storage '{key}' is unreferenced (unknown element size); cannot advance.");
            p += numel * dt.SizeInBytes;
            if (p > length) throw new EndOfStreamException($"Legacy storage '{key}' runs past end of file.");
        }

        Dictionary<string, PytorchTensorDescriptor> descriptors = new(metas.Count);
        foreach ((string name, PickleTensor meta) in metas)
        {
            TensorShape shape = new(meta.Size);
            Tensor t = new(shape, meta.Storage.DType);
            long byteCount = shape.ElementCount * meta.Storage.DType.SizeInBytes;
            long src = keyDataOffset[meta.Storage.Key] + meta.Offset * meta.Storage.DType.SizeInBytes; // storage_offset
            if (src + byteCount > length) throw new EndOfStreamException($"Tensor '{name}' storage out of range.");
            Buffer.MemoryCopy(pAll + src, (byte*)t.DataPointer, byteCount, byteCount);
            MakeRowMajor(t, meta.Size, meta.Stride, meta.Storage.DType);
            _tensors[name] = t;
            descriptors[name] = new PytorchTensorDescriptor { Name = name, DType = meta.Storage.DType, Shape = shape };
        }
        Descriptors = descriptors;
    }

    /// <summary>Runs one pickle stream starting at <paramref name="start"/>, yields its unpickled root, and returns the offset just past its STOP opcode (i.e. the start of the next stream).</summary>
    private static int RunPickle(byte[] buf, int start, out object? result)
    {
        PickleMachine m = new(buf, start);
        result = m.Run();
        return m.Position;
    }

    /// <summary>All tensors as name → owned Tensor. Valid until <see cref="Dispose"/>.</summary>
    public Dictionary<string, Tensor> GetAllTensors()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return new Dictionary<string, Tensor>(_tensors);
    }

    /// <summary>Reorders a just-materialized tensor from its torch STORAGE order into row-major (C-contiguous) order when the tensor's <paramref name="stride"/> is non-contiguous (e.g. a transposed <c>.t()</c> view, which <c>torch.save</c> preserves as size + non-row-major stride over shared storage). The raw copy above wrote the storage bytes as-is; for a contiguous stride that is already correct (no-op fast path), but for a view it yields the transpose — which silently mis-loads e.g. bert-base-uncased Linear weights. Gathers each row-major element from its strided source position. Only fires for the rare non-contiguous case.</summary>
    private static unsafe void MakeRowMajor(Tensor t, long[] size, long[] stride, DType dt)
    {
        int nd = size.Length;
        if (nd == 0 || stride is null || stride.Length != nd) return;
        long[] contiguous = new long[nd];
        long acc = 1;
        for (int i = nd - 1; i >= 0; i--) { contiguous[i] = acc; acc *= size[i]; }
        bool isContiguous = true;
        for (int i = 0; i < nd; i++) if (stride[i] != contiguous[i]) { isContiguous = false; break; }
        if (isContiguous) return;
        long n = acc;
        int es = dt.SizeInBytes;
        byte* data = (byte*)t.DataPointer;
        byte* tmp = (byte*)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)(n * es));
        try
        {
            Buffer.MemoryCopy(data, tmp, n * es, n * es);           // tmp = storage-order bytes we just read
            long[] idx = new long[nd];
            for (long r = 0; r < n; r++)
            {
                long srcElem = 0;
                for (int k = 0; k < nd; k++) srcElem += idx[k] * stride[k];
                Buffer.MemoryCopy(tmp + srcElem * es, data + r * es, es, es);
                for (int k = nd - 1; k >= 0; k--) { if (++idx[k] < size[k]) break; idx[k] = 0; }
            }
        }
        finally { System.Runtime.InteropServices.NativeMemory.Free(tmp); }
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

    /// <summary>Collects every <see cref="PickleTensor"/> in the unpickled object, descending one wrapper level (e.g. a <c>{"module": {...}}</c> or <c>{"state_dict": {...}}</c> envelope) if the top level holds no tensors.</summary>
    private static void FlattenStateDict(object? root, Dictionary<string, PickleTensor> outMap)
    {
        // torch.save(tensor) writes the tensor as the root object, with no dict wrapper (ACE-Step's silence_latent.pt).
        if (root is PickleTensor bare)
        {
            outMap["data"] = bare;
            return;
        }
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

    /// <summary>Recursively flattens nested dicts into fully-qualified dotted keys, visiting every module (unlike <see cref="FlattenStateDict"/>, which descends only one wrapper and drops the prefix). Non-tensor, non-dict leaves (version ints, metadata) are ignored.</summary>
    private static void FlattenStateDictRecursive(object? node, string prefix, Dictionary<string, PickleTensor> outMap)
    {
        if (node is PickleTensor t)
        {
            if (prefix.Length > 0)
            {
                outMap[prefix] = t;
            }
            return;
        }
        if (node is Dictionary<string, object?> dict)
        {
            foreach ((string k, object? v) in dict)
            {
                FlattenStateDictRecursive(v, prefix.Length == 0 ? k : $"{prefix}.{k}", outMap);
            }
        }
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

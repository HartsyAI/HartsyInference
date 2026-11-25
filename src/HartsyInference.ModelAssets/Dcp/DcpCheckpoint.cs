using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using Microsoft.Win32.SafeHandles;

namespace HartsyInference.ModelAssets.Dcp;

/// <summary>Reads a PyTorch Distributed Checkpoint folder without torch: <c>.metadata</c> gives each tensor's byte range, and each range of a <c>.distcp</c> file is a standalone <c>torch.save</c> ZIP read by <see cref="PytorchPickleLoader"/>.
/// Nothing in either file is executed. Tensors come back one at a time so a multi-gigabyte checkpoint never has to fit in memory.</summary>
public sealed class DcpCheckpoint : IDisposable
{
    public const string MetadataFileName = ".metadata";

    private readonly string _folder;
    private readonly Dictionary<string, SafeFileHandle> _handles = new(StringComparer.Ordinal);
    private readonly object _gate = new();
    private int _disposed;

    private DcpCheckpoint(string folder, DcpMetadata metadata)
    {
        _folder = folder;
        Metadata = metadata;
    }

    public DcpMetadata Metadata { get; }

    /// <summary>Tensor key to entry.</summary>
    public IReadOnlyDictionary<string, DcpTensorEntry> Entries => Metadata.Tensors;

    /// <summary>Opens the checkpoint folder (the one holding <c>.metadata</c> and the <c>.distcp</c> files).</summary>
    public static DcpCheckpoint Open(string folder)
    {
        ArgumentException.ThrowIfNullOrEmpty(folder);
        string metadataPath = Path.Combine(folder, MetadataFileName);
        if (!File.Exists(metadataPath))
            throw new FileNotFoundException($"'{folder}' is not a distributed checkpoint folder: no {MetadataFileName} file.", metadataPath);
        return new DcpCheckpoint(folder, DcpMetadata.Parse(File.ReadAllBytes(metadataPath)));
    }

    /// <summary>The key prefixes of the form <c>root.family.</c> (for example <c>models.generator.</c>) that the tensors share, with their tensor counts.</summary>
    public IReadOnlyDictionary<string, int> FamilyPrefixes()
    {
        Dictionary<string, int> counts = new(StringComparer.Ordinal);
        foreach (string key in Entries.Keys)
        {
            int first = key.IndexOf('.');
            int second = first < 0 ? -1 : key.IndexOf('.', first + 1);
            if (second < 0) continue;
            string prefix = key[..(second + 1)];
            counts[prefix] = counts.GetValueOrDefault(prefix) + 1;
        }
        return counts;
    }

    /// <summary>Reads one tensor into new memory the caller owns, converted to <paramref name="convertTo"/> when given.</summary>
    public Tensor ReadTensor(string key, DType? convertTo = null)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!Entries.TryGetValue(key, out DcpTensorEntry entry))
            throw new KeyNotFoundException($"The distributed checkpoint has no tensor '{key}'.");

        SafeFileHandle handle = HandleFor(entry.RelativePath);
        if (entry.Offset + entry.Length > RandomAccess.GetLength(handle))
            throw new EndOfStreamException($"Tensor '{key}' runs past the end of '{entry.RelativePath}'.");

        using SliceReadStream slice = new(handle, entry.Offset, entry.Length);
        using PytorchPickleLoader loader = new();
        loader.OpenLazy(slice, key);
        if (loader.Descriptors.Count != 1)
            throw new InvalidDataException($"The record of tensor '{key}' holds {loader.Descriptors.Count} tensors, expected one.");
        PytorchTensorDescriptor descriptor = loader.Descriptors.Values.First();
        if (descriptor.DType != entry.DType || !ShapeMatches(descriptor.Shape, entry.Shape))
            throw new InvalidDataException(
                $"Tensor '{key}' is {descriptor.DType} {descriptor.Shape} in its record but {entry.DType} [{string.Join(',', entry.Shape)}] in the metadata.");
        return loader.ReadTensor(descriptor.Name, convertTo);
    }

    private static bool ShapeMatches(TensorShape shape, long[] expected)
    {
        if (shape.Rank != expected.Length) return false;
        for (int d = 0; d < expected.Length; d++)
            if (shape[d] != expected[d]) return false;
        return true;
    }

    private SafeFileHandle HandleFor(string relativePath)
    {
        lock (_gate)
        {
            if (_handles.TryGetValue(relativePath, out SafeFileHandle? existing)) return existing;
            string full = Path.GetFullPath(Path.Combine(_folder, relativePath));
            if (!full.StartsWith(Path.GetFullPath(_folder), StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"Storage file '{relativePath}' is outside the checkpoint folder.");
            SafeFileHandle handle = File.OpenHandle(full, FileMode.Open, FileAccess.Read, FileShare.Read, FileOptions.RandomAccess);
            _handles[relativePath] = handle;
            return handle;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        lock (_gate)
        {
            foreach (SafeFileHandle handle in _handles.Values) handle.Dispose();
            _handles.Clear();
        }
    }
}

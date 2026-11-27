using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.PyTorch;

/// <summary>Repacks a pickle checkpoint into safetensors one tensor at a time, so the checkpoint's size does not bound host memory, and checks the result against its source.
/// Only the safe-subset pickle reader parses the source: nothing in it is executed. Names, their order, shapes, dtypes and bytes are carried over unchanged.</summary>
public static class StreamingPickleRepack
{
    /// <summary>The most differences <see cref="Verify"/> lists.</summary>
    public const int MaxReportedDifferences = 20;

    private const int CompareChunkBytes = 1 << 28;

    /// <summary>Writes every tensor of <paramref name="sourcePath"/> to <paramref name="outputPath"/>, in the source's key order, through a staging file.</summary>
    /// <param name="recursiveFlatten">Passed to the pickle reader: true for a root that is a dict of state dicts (a Lightning <c>.ckpt</c>).</param>
    /// <param name="metadata">Optional string map for the output header's <c>__metadata__</c>.</param>
    public static PickleRepackSummary Repack(string sourcePath, string outputPath, bool recursiveFlatten = false, IReadOnlyDictionary<string, string>? metadata = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(sourcePath);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        string? directory = Path.GetDirectoryName(Path.GetFullPath(outputPath));
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }
        if (!PytorchPickleLoader.IsZipCheckpoint(sourcePath))
        {
            return RepackWhole(sourcePath, outputPath, recursiveFlatten, metadata);
        }
        using PytorchPickleLoader loader = new();
        loader.OpenLazy(sourcePath, recursiveFlatten);
        if (loader.Descriptors.Count == 0)
        {
            throw new InvalidDataException($"Checkpoint '{sourcePath}' holds no tensors.");
        }
        List<SafeTensorsStreamWriter.Entry> entries = [];
        long bytes = 0;
        foreach (PytorchTensorDescriptor descriptor in loader.Descriptors.Values)
        {
            string name = descriptor.Name;
            entries.Add(new SafeTensorsStreamWriter.Entry(name, descriptor.DType, descriptor.Shape, () => (loader.ReadTensor(name), true)));
            bytes += descriptor.DType.ComputeByteCount(descriptor.Shape.ElementCount);
        }
        SafeTensorsStreamWriter.Write(outputPath, entries, metadata);
        return new PickleRepackSummary(entries.Count, bytes, true, CountStorageRereads(loader.Descriptors.Values));
    }

    /// <summary>Compares every tensor of <paramref name="sourcePath"/> with the same-named tensor of <paramref name="repackedPath"/>. A name, order, shape, dtype or byte difference is a finding, not an exception.
    /// By default the source is read tensor by tensor, as <see cref="Repack"/> reads it, so a fault in that reader would show on both sides; <paramref name="wholeSource"/> reads it through the whole-file path
    /// (<see cref="PytorchPickleLoader.Load"/>, its own byte-reading code, at the cost of holding the tensors in memory), which a reader fault on the lazy path cannot hide behind.</summary>
    public static PickleRepackVerification Verify(string sourcePath, string repackedPath, bool recursiveFlatten = false, bool wholeSource = false)
    {
        ArgumentException.ThrowIfNullOrEmpty(sourcePath);
        ArgumentException.ThrowIfNullOrEmpty(repackedPath);
        List<string> differences = [];
        using PytorchPickleLoader pickle = new();
        Dictionary<string, Tensor>? whole = null;
        if (wholeSource || !PytorchPickleLoader.IsZipCheckpoint(sourcePath))
        {
            pickle.Load(sourcePath, recursiveFlatten);
            whole = pickle.GetAllTensors();
        }
        else
        {
            pickle.OpenLazy(sourcePath, recursiveFlatten);
        }
        List<string> names = [.. whole is null ? pickle.Descriptors.Keys : whole.Keys];
        using SafeTensorsLoader repacked = new();
        repacked.Load(repackedPath);
        List<string> repackedNames = [.. repacked.Descriptors.Keys];
        if (!names.SequenceEqual(repackedNames, StringComparer.Ordinal))
        {
            Report(differences, $"the tensor names or their order differ ({names.Count} in the source, {repackedNames.Count} in the repack)");
        }
        int compared = 0;
        foreach (string name in names)
        {
            if (!repacked.Descriptors.TryGetValue(name, out SafeTensorDescriptor? target))
            {
                Report(differences, $"'{name}' is missing from the repack");
                continue;
            }
            DType dtype = whole is null ? pickle.Descriptors[name].DType : whole[name].DType;
            TensorShape shape = whole is null ? pickle.Descriptors[name].Shape : whole[name].Shape;
            if (dtype != target.DType || !shape.Equals(target.Shape))
            {
                Report(differences, $"'{name}' is {dtype.Name} {shape} in the source and {target.DType.Name} {target.Shape} in the repack");
                continue;
            }
            Tensor source = whole is null ? pickle.ReadTensor(name) : whole[name];
            try
            {
                if (!SameBytes(source, repacked.GetTensor(name), dtype.ComputeByteCount(shape.ElementCount)))
                {
                    Report(differences, $"'{name}' differs in value");
                }
            }
            finally
            {
                if (whole is null)
                {
                    source.Dispose();
                }
            }
            compared++;
        }
        return new PickleRepackVerification(differences.Count == 0, compared, differences);
    }

    private static PickleRepackSummary RepackWhole(string sourcePath, string outputPath, bool recursiveFlatten, IReadOnlyDictionary<string, string>? metadata)
    {
        using PytorchPickleLoader loader = new();
        loader.Load(sourcePath, recursiveFlatten);
        Dictionary<string, Tensor> tensors = loader.GetAllTensors();
        if (tensors.Count == 0)
        {
            throw new InvalidDataException($"Checkpoint '{sourcePath}' holds no tensors.");
        }
        SafeTensorsWriter.Save(outputPath, tensors, metadata);
        long bytes = tensors.Values.Sum(t => t.DType.ComputeByteCount(t.ElementCount));
        return new PickleRepackSummary(tensors.Count, bytes, false, 0);
    }

    // Mirrors the lazy reader's cursor, which keeps one storage entry open: a read costs a second pass when its entry was opened before and the cursor is on another entry or already past the tensor.
    private static int CountStorageRereads(IEnumerable<PytorchTensorDescriptor> descriptors)
    {
        int rereads = 0;
        HashSet<string> opened = new(StringComparer.Ordinal);
        string? key = null;
        long position = 0;
        foreach (PytorchTensorDescriptor descriptor in descriptors)
        {
            long start = descriptor.StorageByteOffset;
            if (key != descriptor.StorageKey || position > start)
            {
                if (!opened.Add(descriptor.StorageKey) || key == descriptor.StorageKey)
                {
                    rereads++;
                }
                key = descriptor.StorageKey;
            }
            position = start + descriptor.DType.ComputeByteCount(descriptor.Shape.ElementCount);
        }
        return rereads;
    }

    private static unsafe bool SameBytes(Tensor left, Tensor right, long byteCount)
    {
        byte* a = (byte*)left.DataPointer;
        byte* b = (byte*)right.DataPointer;
        for (long offset = 0; offset < byteCount; offset += CompareChunkBytes)
        {
            int length = (int)Math.Min(CompareChunkBytes, byteCount - offset);
            if (!new ReadOnlySpan<byte>(a + offset, length).SequenceEqual(new ReadOnlySpan<byte>(b + offset, length)))
            {
                return false;
            }
        }
        return true;
    }

    private static void Report(List<string> differences, string message)
    {
        if (differences.Count < MaxReportedDifferences)
        {
            differences.Add(message);
        }
    }
}

using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.IO;
using HartsyInference.Core.Memory;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.SafeTensors;

/// <summary>A multi-file safetensors checkpoint opened header-first, so the full inventory is cross-checked before any shard is mapped.</summary>
/// <remarks>Tensors from <see cref="GetTensor"/> borrow their shard's mapping and are valid until the set is disposed.</remarks>
public sealed class ShardedSafeTensorSet : IDisposable
{
    private const string IndexFileName = "model.safetensors.index.json";
    private const int ExamplesPerProblem = 5;

    private readonly ShardSetOptions _options;
    private int _disposed;

    private ShardedSafeTensorSet(string source, IReadOnlyList<SafeTensorShard> shards,
        IReadOnlyDictionary<string, TensorLocation> inventory, long totalBytes, long? declaredTotalSize, ShardSetOptions options)
    {
        Source = source;
        Shards = shards;
        Inventory = inventory;
        TotalTensorBytes = totalBytes;
        DeclaredTotalSize = declaredTotalSize;
        _options = options;
    }

    /// <summary>The directory or file list this set was opened from, for diagnostics.</summary>
    public string Source { get; }

    /// <summary>Every shard, in index order (sorted by file name for <see cref="OpenIndex"/>).</summary>
    public IReadOnlyList<SafeTensorShard> Shards { get; }

    /// <summary>Tensor name to location, complete across all shards.</summary>
    public IReadOnlyDictionary<string, TensorLocation> Inventory { get; }

    /// <summary>Sum of every tensor's byte length.</summary>
    public long TotalTensorBytes { get; }

    /// <summary>The index's <c>metadata.total_size</c>, or null when there is no index or it declares none.</summary>
    public long? DeclaredTotalSize { get; }

    /// <summary>How many shards are currently memory-mapped.</summary>
    public int MappedShardCount => Shards.Count(static shard => shard.IsMapped);

    /// <summary>Opens the shards listed by <c>model.safetensors.index.json</c> in <paramref name="directory"/>.</summary>
    public static ShardedSafeTensorSet OpenIndex(string directory, ShardSetOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        ShardSetOptions resolved = options ?? new ShardSetOptions();
        string indexPath = Path.Combine(directory, IndexFileName);
        if (!File.Exists(indexPath))
            throw new FileNotFoundException($"'{directory}' has no {IndexFileName}; pass the shard files to OpenFiles instead.", indexPath);

        (Dictionary<string, string> weightMap, long? totalSize) = ReadIndex(indexPath);
        List<string> fileNames = weightMap.Values.Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToList();
        List<string> missing = fileNames.Where(name => !File.Exists(Path.Combine(directory, name))).ToList();
        if (missing.Count > 0)
        {
            throw new HartsyInferenceException(
                $"{IndexFileName} in '{directory}' lists {fileNames.Count} shard files but {missing.Count} are missing on disk: "
                + $"{Examples(missing)}. Re-run the download to fetch them.");
        }

        return Build(directory, fileNames.Select(name => Path.Combine(directory, name)).ToList(), weightMap, totalSize, resolved);
    }

    /// <summary>Opens explicit shard files, with no index to cross-check against (odd names such as MLX's).</summary>
    public static ShardedSafeTensorSet OpenFiles(IReadOnlyList<string> files, ShardSetOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        if (files.Count == 0)
            throw new ArgumentException("A shard set needs at least one file.", nameof(files));
        List<string> paths = files.Select(Path.GetFullPath).ToList();
        HashSet<string> distinct = new HashSet<string>(paths, StringComparer.Ordinal);
        if (distinct.Count != paths.Count)
            throw new ArgumentException("The same shard file is listed twice.", nameof(files));
        List<string> missing = paths.Where(path => !File.Exists(path)).ToList();
        if (missing.Count > 0)
            throw new HartsyInferenceException($"{missing.Count} shard file(s) are missing on disk: {Examples(missing)}.");

        return Build($"{paths.Count} files starting at '{paths[0]}'", paths, weightMap: null, declaredTotal: null,
            options ?? new ShardSetOptions());
    }

    /// <summary>Returns the tensor as a borrowed view of its shard's mapping, mapping the shard on first use.</summary>
    public unsafe Tensor GetTensor(string name)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!Inventory.TryGetValue(name, out TensorLocation? location))
            throw new KeyNotFoundException($"Tensor '{name}' is not in the shard set from {Source}.");
        SafeTensorShard shard = location.Shard;
        if (shard.PreadOnly)
        {
            throw new InvalidOperationException(
                $"Tensor '{name}' lives in {shard.FileName}, which is pread-only and never mapped; read its bytes "
                + "through GetByteSource(shard).");
        }
        SafeTensorDTypes.ThrowIfNotMaterialisable(location.DType, name, shard.Path);

        MmapHandle handle = shard.Map(_options.AdviseRandom);
        Tensor tensor = new Tensor(handle.PointerAt(location.FileOffset), location.Shape, location.DType);
        tensor.SetKeepAlive(handle);
        return tensor;
    }

    /// <summary>Returns the shard's positional-read source, opening it on first use. The set owns it; do not dispose it.</summary>
    public IWeightByteSource GetByteSource(SafeTensorShard shard)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(shard);
        if (shard.Index < 0 || shard.Index >= Shards.Count || !ReferenceEquals(Shards[shard.Index], shard))
            throw new ArgumentException($"Shard '{shard.FileName}' does not belong to this set.", nameof(shard));
        return shard.Source();
    }

    private static ShardedSafeTensorSet Build(string source, List<string> paths, Dictionary<string, string>? weightMap,
        long? declaredTotal, ShardSetOptions options)
    {
        HashSet<string> preadNames = new HashSet<string>(options.PreadOnlyShards, StringComparer.Ordinal);
        HashSet<string> knownNames = paths.Select(Path.GetFileName).OfType<string>().ToHashSet(StringComparer.Ordinal);
        List<string> unknownPread = preadNames.Where(name => !knownNames.Contains(name)).Order(StringComparer.Ordinal).ToList();
        if (unknownPread.Count > 0)
        {
            throw new ArgumentException(
                $"PreadOnlyShards names {Examples(unknownPread)}, which {(unknownPread.Count == 1 ? "is" : "are")} not in the shard set.");
        }

        List<SafeTensorShard> shards = new List<SafeTensorShard>(paths.Count);
        for (int i = 0; i < paths.Count; i++)
        {
            SafeTensorHeader header = SafeTensorHeaderReader.Read(paths[i], options.MaxHeaderBytes);
            shards.Add(new SafeTensorShard(i, paths[i], header, preadNames.Contains(Path.GetFileName(paths[i]))));
        }

        Dictionary<string, TensorLocation> inventory = new Dictionary<string, TensorLocation>(StringComparer.Ordinal);
        List<string> problems = [];
        List<string> duplicates = [];
        long totalBytes = 0;
        foreach (SafeTensorShard shard in shards)
        {
            foreach (KeyValuePair<string, SafeTensorDescriptor> entry in shard.Header.Tensors)
            {
                SafeTensorDescriptor descriptor = entry.Value;
                TensorLocation location = new TensorLocation(shard, descriptor.DType, descriptor.Shape,
                    descriptor.DataOffset, descriptor.ByteLength);
                if (inventory.TryGetValue(entry.Key, out TensorLocation? first))
                    duplicates.Add($"'{entry.Key}' in {first.Shard.FileName} and {shard.FileName}");
                else
                    inventory.Add(entry.Key, location);
                totalBytes += descriptor.ByteLength;
            }
        }
        if (duplicates.Count > 0)
            problems.Add($"{duplicates.Count} key(s) appear in more than one shard: {Examples(duplicates)}");

        if (weightMap is not null)
            CheckAgainstIndex(shards, inventory, weightMap, duplicates.Count, problems);
        CheckTotals(totalBytes, declaredTotal, options, weightMap is not null, problems);

        if (problems.Count > 0)
        {
            throw new HartsyInferenceException(
                $"Shard set from {source} is inconsistent:\n - " + string.Join("\n - ", problems));
        }
        return new ShardedSafeTensorSet(source, shards, inventory, totalBytes, declaredTotal, options);
    }

    private static void CheckAgainstIndex(List<SafeTensorShard> shards, Dictionary<string, TensorLocation> inventory,
        Dictionary<string, string> weightMap, int duplicateCount, List<string> problems)
    {
        List<string> notInIndex = [];
        foreach (KeyValuePair<string, TensorLocation> entry in inventory)
        {
            if (!weightMap.ContainsKey(entry.Key))
                notInIndex.Add($"'{entry.Key}' ({entry.Value.Shard.FileName})");
        }
        if (notInIndex.Count > 0)
            problems.Add($"{notInIndex.Count} key(s) are in shard headers but not in the index: {Examples(notInIndex)}");

        List<string> notInHeader = [];
        List<string> wrongShard = [];
        foreach (KeyValuePair<string, string> entry in weightMap)
        {
            if (!inventory.TryGetValue(entry.Key, out TensorLocation? location))
                notInHeader.Add($"'{entry.Key}' ({entry.Value})");
            else if (duplicateCount == 0 && !string.Equals(location.Shard.FileName, entry.Value, StringComparison.Ordinal))
                wrongShard.Add($"'{entry.Key}' is indexed to {entry.Value} but stored in {location.Shard.FileName}");
        }
        if (notInHeader.Count > 0)
            problems.Add($"{notInHeader.Count} key(s) are in the index but not in the header of their shard: {Examples(notInHeader)}");
        if (wrongShard.Count > 0)
            problems.Add($"{wrongShard.Count} key(s) sit in a different shard than the index says: {Examples(wrongShard)}");
    }

    private static void CheckTotals(long totalBytes, long? declaredTotal, ShardSetOptions options, bool hasIndex, List<string> problems)
    {
        if (options.RequireTotalSizeMatch && declaredTotal is { } declared && declared != totalBytes)
        {
            problems.Add($"tensor bytes sum to {totalBytes} but the index metadata.total_size is {declared} "
                + $"(difference {totalBytes - declared}); a shard is missing, short or from another revision");
        }
        if (options.ExpectedTotalBytes is { } expected && expected != totalBytes)
        {
            problems.Add($"tensor bytes sum to {totalBytes} but {expected} were expected for this checkpoint "
                + $"(difference {totalBytes - expected}){(hasIndex ? string.Empty : "; no index to cross-check against")}");
        }
    }

    private static (Dictionary<string, string> WeightMap, long? TotalSize) ReadIndex(string indexPath)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(indexPath));
        JsonElement root = document.RootElement;
        if (root.ValueKind != JsonValueKind.Object
            || !root.TryGetProperty("weight_map", out JsonElement map) || map.ValueKind != JsonValueKind.Object)
        {
            throw new HartsyInferenceException($"'{indexPath}' has no weight_map object; it is not a safetensors index.");
        }

        Dictionary<string, string> weightMap = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (JsonProperty entry in map.EnumerateObject())
        {
            string? file = entry.Value.ValueKind == JsonValueKind.String ? entry.Value.GetString() : null;
            if (string.IsNullOrEmpty(file) || file != Path.GetFileName(file))
            {
                throw new HartsyInferenceException(
                    $"'{indexPath}' maps '{entry.Name}' to '{entry.Value}', which is not a plain file name beside the index.");
            }
            if (!weightMap.TryAdd(entry.Name, file))
                throw new HartsyInferenceException($"'{indexPath}' lists key '{entry.Name}' twice in weight_map.");
        }

        long? totalSize = null;
        if (root.TryGetProperty("metadata", out JsonElement metadata) && metadata.ValueKind == JsonValueKind.Object
            && metadata.TryGetProperty("total_size", out JsonElement size))
        {
            if (size.ValueKind != JsonValueKind.Number || !size.TryGetInt64(out long parsed) || parsed < 0)
                throw new HartsyInferenceException($"'{indexPath}' has a metadata.total_size that is not a non-negative integer.");
            totalSize = parsed;
        }
        return (weightMap, totalSize);
    }

    private static string Examples(IReadOnlyList<string> items)
    {
        string shown = string.Join(", ", items.Take(ExamplesPerProblem));
        return items.Count > ExamplesPerProblem ? $"{shown}, and {items.Count - ExamplesPerProblem} more" : shown;
    }

    /// <summary>Unmaps every shard and closes every pread handle. Borrowed tensors must not be used afterwards.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (SafeTensorShard shard in Shards)
            shard.Release();
    }
}

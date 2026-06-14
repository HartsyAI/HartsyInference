using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelHandler.SafeTensors;

/// <summary>Loads multi-file sharded safetensors models. Reads each shard sequentially and builds a unified tensor index across all files.</summary>
public sealed class SafeTensorsShardLoader : IDisposable
{
    private readonly List<SafeTensorsLoader> _shards = [];
    private readonly Dictionary<string, (SafeTensorsLoader Loader, string TensorName)> _tensorIndex = [];
    private int _disposed;

    /// <summary>All tensor names available across all shards.</summary>
    public IReadOnlyCollection<string> TensorNames => _tensorIndex.Keys;

    /// <summary>Number of shards loaded.</summary>
    public int ShardCount => _shards.Count;

    /// <summary>Loads all safetensors shards from a directory. Shard files are expected to follow the pattern model-00001-of-00005.safetensors or similar naming.</summary>
    public void LoadDirectory(string directoryPath)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        string[] files = Directory.GetFiles(directoryPath, "*.safetensors");
        Array.Sort(files, StringComparer.OrdinalIgnoreCase);

        if (files.Length == 0)
            throw new FileNotFoundException($"No safetensors files found in {directoryPath}.");

        foreach (string file in files)
        {
            LoadShard(file);
        }
    }

    /// <summary>Loads a single shard file and merges its tensor index.</summary>
    public void LoadShard(string filePath)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(filePath);
        _shards.Add(loader);

        foreach (KeyValuePair<string, SafeTensorDescriptor> kvp in loader.Descriptors)
        {
            _tensorIndex[kvp.Key] = (loader, kvp.Key);
        }
    }

    /// <summary>Returns a tensor by name, resolved across all loaded shards.</summary>
    public Tensor GetTensor(string name)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        if (!_tensorIndex.TryGetValue(name, out (SafeTensorsLoader Loader, string TensorName) entry))
            throw new KeyNotFoundException($"Tensor '{name}' not found in any loaded shard.");

        return entry.Loader.GetTensor(entry.TensorName);
    }

    /// <summary>Returns all tensors across all shards as a unified dictionary.</summary>
    public Dictionary<string, Tensor> GetAllTensors()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        Dictionary<string, Tensor> tensors = new(_tensorIndex.Count);
        foreach (KeyValuePair<string, (SafeTensorsLoader Loader, string TensorName)> kvp in _tensorIndex)
        {
            tensors[kvp.Key] = kvp.Value.Loader.GetTensor(kvp.Value.TensorName);
        }
        return tensors;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        foreach (SafeTensorsLoader shard in _shards)
        {
            shard.Dispose();
        }
        _shards.Clear();
        _tensorIndex.Clear();
    }
}

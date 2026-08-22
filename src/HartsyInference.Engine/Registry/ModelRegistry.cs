using System.Collections.Concurrent;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Models;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.HuggingFace;
using HartsyInference.ModelAssets.Registry;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Registry;

/// <summary>In-memory cache of loaded model data. Tracks which models are currently loaded and provides lookup by model ID.</summary>
public sealed class ModelRegistry : IDisposable
{
    private readonly ConcurrentDictionary<string, LoadedModel> _loaded = new(StringComparer.OrdinalIgnoreCase);

    // safetensors tensors borrow mmap memory, so the loader that produced a model's weights must stay
    // alive until the model is unloaded. Tracked here and disposed in Unload/Dispose.
    private readonly ConcurrentDictionary<string, List<IDisposable>> _loaders = new(StringComparer.OrdinalIgnoreCase);
    private int _disposed;

    /// <summary>Number of currently loaded models.</summary>
    public int LoadedCount => _loaded.Count;

    /// <summary>All currently loaded model IDs.</summary>
    public IReadOnlyCollection<string> LoadedModelIds => _loaded.Keys.ToArray();

    /// <summary>Registers a loaded model with its tensor weights.</summary>
    public void Register(ModelInfo info, IReadOnlyDictionary<string, Tensor> weights)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        LoadedModel model = new LoadedModel
        {
            Info = info,
            Weights = weights,
            LoadedAt = DateTime.UtcNow,
        };

        _loaded[info.Id] = model;
        Logs.Info($"Registered model '{info.Name}' ({info.Id}) with {weights.Count} tensors.");
    }

    /// <summary>Gets a loaded model by ID, or null if not loaded.</summary>
    public LoadedModel? Get(string modelId)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        if (_loaded.TryGetValue(modelId, out LoadedModel? model))
        {
            model.Info.LastAccessed = DateTime.UtcNow;
            return model;
        }

        return null;
    }

    /// <summary>Checks if a model is currently loaded.</summary>
    public bool IsLoaded(string modelId)
    {
        return _loaded.ContainsKey(modelId);
    }

    /// <summary>Unloads a model, disposing any owned tensors.</summary>
    public bool Unload(string modelId)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        if (_loaded.TryRemove(modelId, out LoadedModel? model))
        {
            DisposeModelWeights(model);
            DisposeLoaders(modelId);
            Logs.Info($"Unloaded model '{model.Info.Name}' ({modelId}).");
            return true;
        }

        return false;
    }

    /// <summary>Loads a model from a local path or HuggingFace repo id, registers it, and returns it.
    /// If the id is already loaded, the cached instance is returned immediately. Local paths (a
    /// safetensors file, a sharded checkpoint directory, or a diffusers-layout directory) are loaded
    /// directly; otherwise the id is treated as a HuggingFace repo and the preferred safetensors
    /// variant is downloaded into <paramref name="cache"/> (default: <c>~/.hartsyinference/models</c>)
    /// before loading.
    ///
    /// <para>The returned weights borrow memory-mapped storage; the underlying loaders are retained by
    /// this registry and released on <see cref="Unload"/> / <see cref="Dispose"/>.</para></summary>
    /// <param name="modelIdOrPath">Local file/directory path, or a HuggingFace repo id (e.g. "stabilityai/sdxl").</param>
    /// <param name="cache">Disk cache for downloaded models. A default store is used when null.</param>
    /// <param name="hub">HuggingFace client for downloads. A default client (honoring HF_TOKEN) is created and disposed when null.</param>
    /// <param name="downloadProgress">Reports download progress 0..1 for the HuggingFace path.</param>
    public async Task<LoadedModel> LoadAsync(
        string modelIdOrPath,
        ModelCacheStore? cache = null,
        HuggingFaceClient? hub = null,
        IProgress<double>? downloadProgress = null,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        if (_loaded.TryGetValue(modelIdOrPath, out LoadedModel? existing))
        {
            existing.Info.LastAccessed = DateTime.UtcNow;
            return existing;
        }

        // Local path → resolve layout and load directly.
        if (File.Exists(modelIdOrPath) || Directory.Exists(modelIdOrPath))
        {
            ModelLayout layout = ModelLayoutResolver.Resolve(modelIdOrPath);
            return LoadFromLayout(modelIdOrPath, modelIdOrPath, layout, huggingFaceRepo: null);
        }

        // Otherwise treat as a HuggingFace repo id and download the preferred variant.
        cache ??= new ModelCacheStore();
        bool ownsHub = hub is null;
        hub ??= new HuggingFaceClient();
        try
        {
            string localFile = await EnsureDownloadedAsync(modelIdOrPath, cache, hub, downloadProgress, ct)
                .ConfigureAwait(false);
            ModelLayout layout = ModelLayoutResolver.Resolve(localFile);
            return LoadFromLayout(modelIdOrPath, localFile, layout, huggingFaceRepo: modelIdOrPath);
        }
        finally
        {
            if (ownsHub) hub.Dispose();
        }
    }

    /// <summary>Downloads the preferred safetensors variant of <paramref name="repoId"/> into the cache (skipping if already present) and returns the local file path.</summary>
    private static async Task<string> EnsureDownloadedAsync(
        string repoId, ModelCacheStore cache, HuggingFaceClient hub, IProgress<double>? progress, CancellationToken ct)
    {
        ModelInfo? cached = cache.Get(repoId);
        if (cached is not null && File.Exists(cached.LocalPath))
        {
            Logs.Info($"Model '{repoId}' already cached at {cached.LocalPath}.");
            return cached.LocalPath;
        }

        IReadOnlyList<HuggingFaceFileInfo> files = await hub.ListModelFilesAsync(repoId, ct).ConfigureAwait(false);
        HuggingFaceFileInfo? chosen = ChooseSafeTensorsFile(files);
        if (chosen is null)
        {
            throw new InvalidOperationException(
                $"No .safetensors file found in HuggingFace repo '{repoId}'. Only single-file safetensors auto-download is supported.");
        }

        string destination = cache.GetCachePath(repoId, Path.GetFileName(chosen.FileName));
        await hub.DownloadFileAsync(repoId, chosen.FileName, destination, progress, sha256: null, ct).ConfigureAwait(false);

        cache.Add(new ModelInfo
        {
            Id = repoId,
            Name = repoId,
            Format = ModelFormat.SafeTensors,
            LocalPath = destination,
            FileSize = new FileInfo(destination).Length,
            HuggingFaceRepo = repoId,
        });

        return destination;
    }

    /// <summary>Picks the preferred single safetensors file: fp16 variant first, then the largest.</summary>
    private static HuggingFaceFileInfo? ChooseSafeTensorsFile(IReadOnlyList<HuggingFaceFileInfo> files)
    {
        List<HuggingFaceFileInfo> safetensors = files
            .Where(f => f.FileName.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
            .ToList();
        if (safetensors.Count == 0) return null;

        HuggingFaceFileInfo? fp16 = safetensors
            .Where(f => f.FileName.Contains("fp16", StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(f => f.Size)
            .FirstOrDefault();

        return fp16 ?? safetensors.OrderByDescending(f => f.Size).First();
    }

    /// <summary>Loads weights from every safetensors file in the layout, retains the loaders, and registers the model.</summary>
    private LoadedModel LoadFromLayout(string id, string displayPath, ModelLayout layout, string? huggingFaceRepo)
    {
        Dictionary<string, Tensor> weights = new();
        List<IDisposable> loaders = new();

        foreach (string file in layout.SafeTensorsFiles)
        {
            SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(file);
            loaders.Add(loader);
            foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
            {
                weights[kvp.Key] = kvp.Value;
            }
        }

        ModelArchitecture arch = ModelArchitectureDetector.DetectFromFile(layout.RepresentativeFile);

        ModelInfo info = new ModelInfo
        {
            Id = id,
            Name = Path.GetFileNameWithoutExtension(displayPath),
            Format = ModelFormat.SafeTensors,
            Architecture = arch == ModelArchitecture.Unknown ? null : arch.ToString(),
            LocalPath = layout.RepresentativeFile,
            FileSize = File.Exists(layout.RepresentativeFile) ? new FileInfo(layout.RepresentativeFile).Length : 0,
            HuggingFaceRepo = huggingFaceRepo,
        };

        Register(info, weights);
        _loaders[id] = loaders;
        return _loaded[id];
    }

    private void DisposeLoaders(string modelId)
    {
        if (_loaders.TryRemove(modelId, out List<IDisposable>? loaders))
        {
            foreach (IDisposable loader in loaders) loader.Dispose();
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        foreach (KeyValuePair<string, LoadedModel> kvp in _loaded)
        {
            DisposeModelWeights(kvp.Value);
        }

        foreach (KeyValuePair<string, List<IDisposable>> kvp in _loaders)
        {
            foreach (IDisposable loader in kvp.Value) loader.Dispose();
        }

        _loaded.Clear();
        _loaders.Clear();
    }

    private static void DisposeModelWeights(LoadedModel model)
    {
        foreach (KeyValuePair<string, Tensor> kvp in model.Weights)
        {
            if (kvp.Value.OwnsMemory)
            {
                kvp.Value.Dispose();
            }
        }
    }
}

/// <summary>A model that is currently loaded in memory with its tensor weights.</summary>
public sealed class LoadedModel
{
    public required ModelInfo Info { get; init; }

    /// <summary>Tensor name → weight data mapping.</summary>
    public required IReadOnlyDictionary<string, Tensor> Weights { get; init; }

    /// <summary>When this model was loaded.</summary>
    public required DateTime LoadedAt { get; init; }
}

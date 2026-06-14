using System.Collections.Concurrent;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelHandler.Registry;

/// <summary>In-memory cache of loaded model data. Tracks which models are currently loaded and provides lookup by model ID.</summary>
public sealed class ModelRegistry : IDisposable
{
    private readonly ConcurrentDictionary<string, LoadedModel> _loaded = new(StringComparer.OrdinalIgnoreCase);
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
            Logs.Info($"Unloaded model '{model.Info.Name}' ({modelId}).");
            return true;
        }

        return false;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        foreach (KeyValuePair<string, LoadedModel> kvp in _loaded)
        {
            DisposeModelWeights(kvp.Value);
        }

        _loaded.Clear();
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
    /// <summary>Model metadata.</summary>
    public required ModelInfo Info { get; init; }

    /// <summary>Tensor name → weight data mapping.</summary>
    public required IReadOnlyDictionary<string, Tensor> Weights { get; init; }

    /// <summary>When this model was loaded.</summary>
    public required DateTime LoadedAt { get; init; }
}

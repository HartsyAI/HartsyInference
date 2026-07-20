using System.Text.Json;
using HartsyInference.Audio.Cache;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Audio;

/// <summary>Loads a HuggingFace checkpoint into one merged tensor map plus the loaders whose mmapped buffers the
/// tensors reference: single-file safetensors, then the sharded index, then a PyTorch pickle.</summary>
internal static class AudioCheckpoints
{
    /// <summary>Resolves and loads <paramref name="repo"/>'s weights, downloading them on first use.</summary>
    internal static async Task<(IReadOnlyDictionary<string, Tensor> Dict, IDisposable[] Loaders)> LoadAsync(string repo, CancellationToken cancel)
    {
        try
        {
            string path = await AudioModelCache.GetAsync(repo, "model.safetensors", ct: cancel).ConfigureAwait(false);
            SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(path);
            return (loader.GetAllTensors(), [loader]);
        }
        catch (FileNotFoundException ex)
        {
            Logs.Debug($"[Audio] '{repo}' has no single-file model.safetensors ({ex.Message}); trying the shard index.");
        }

        try
        {
            string indexPath = await AudioModelCache.GetAsync(repo, "model.safetensors.index.json", ct: cancel).ConfigureAwait(false);
            HashSet<string> shards = ReadShardNames(indexPath);
            Dictionary<string, Tensor> merged = new Dictionary<string, Tensor>(StringComparer.Ordinal);
            List<IDisposable> loaders = [];
            foreach (string shard in shards)
            {
                string shardPath = await AudioModelCache.GetAsync(repo, shard, ct: cancel).ConfigureAwait(false);
                SafeTensorsLoader shardLoader = new SafeTensorsLoader();
                shardLoader.Load(shardPath);
                loaders.Add(shardLoader);
                foreach (KeyValuePair<string, Tensor> entry in shardLoader.GetAllTensors())
                {
                    merged[entry.Key] = entry.Value;
                }
            }
            return (merged, [.. loaders]);
        }
        catch (FileNotFoundException ex)
        {
            Logs.Debug($"[Audio] '{repo}' has no safetensors shard index ({ex.Message}); falling back to pytorch_model.bin.");
        }

        string binPath = await AudioModelCache.GetAsync(repo, "pytorch_model.bin", ct: cancel).ConfigureAwait(false);
        PytorchPickleLoader pickle = new PytorchPickleLoader();
        pickle.Load(binPath);
        return (pickle.GetAllTensors(), [pickle]);
    }

    /// <summary>Loads a local checkpoint by extension: safetensors (mmap) or a PyTorch pickle.</summary>
    internal static (IReadOnlyDictionary<string, Tensor> Tensors, IDisposable Loader) LoadFile(string path)
    {
        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            SafeTensorsLoader safeTensors = new SafeTensorsLoader();
            safeTensors.Load(path);
            return (safeTensors.GetAllTensors(), safeTensors);
        }
        PytorchPickleLoader pickle = new PytorchPickleLoader();
        pickle.Load(path);
        return (pickle.GetAllTensors(), pickle);
    }

    private static HashSet<string> ReadShardNames(string indexPath)
    {
        HashSet<string> shards = new HashSet<string>(StringComparer.Ordinal);
        using FileStream stream = File.OpenRead(indexPath);
        using JsonDocument document = JsonDocument.Parse(stream);
        if (!document.RootElement.TryGetProperty("weight_map", out JsonElement weightMap))
        {
            throw new InvalidDataException($"Safetensors index '{indexPath}' has no weight_map.");
        }
        foreach (JsonProperty entry in weightMap.EnumerateObject())
        {
            if (entry.Value.GetString() is { Length: > 0 } shard)
            {
                shards.Add(shard);
            }
        }
        return shards;
    }
}

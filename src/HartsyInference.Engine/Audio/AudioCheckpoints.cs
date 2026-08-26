using System.Text.Json;
using HartsyInference.Audio.Cache;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Audio;

/// <summary>Loads a HuggingFace checkpoint into one merged tensor map plus the loaders whose mmapped buffers the tensors reference: single-file safetensors, then the sharded index, then a PyTorch pickle.</summary>
internal static class AudioCheckpoints
{
    /// <summary>Resolves and loads <paramref name="repo"/>'s weights, downloading them on first use.</summary>
    internal static async Task<(IReadOnlyDictionary<string, Tensor> Dict, IDisposable[] Loaders)> LoadAsync(string repo, string category, CancellationToken cancel)
    {
        try
        {
            string path = await AudioModelCache.GetAsync(repo, "model.safetensors", category, ct: cancel).ConfigureAwait(false);
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
            string indexPath = await AudioModelCache.GetAsync(repo, "model.safetensors.index.json", category, ct: cancel).ConfigureAwait(false);
            HashSet<string> shards = ReadShardNames(indexPath);
            Dictionary<string, Tensor> merged = new Dictionary<string, Tensor>(StringComparer.Ordinal);
            List<IDisposable> loaders = [];
            foreach (string shard in shards)
            {
                string shardPath = await AudioModelCache.GetAsync(repo, shard, category, ct: cancel).ConfigureAwait(false);
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

        string binPath = await AudioModelCache.GetAsync(repo, "pytorch_model.bin", category, ct: cancel).ConfigureAwait(false);
        PytorchPickleLoader pickle = new PytorchPickleLoader();
        pickle.Load(binPath);
        return (pickle.GetAllTensors(), [pickle]);
    }

    /// <summary>Names the files <see cref="LoadAsync"/> would read, without loading any tensors — for
    /// prefetching a checkpoint ahead of first use.
    ///
    /// <para>Resolution mirrors <see cref="LoadAsync"/>'s: single-file safetensors, else the shard index, else
    /// the pickle. A sharded repo cannot state its own file list without the index, so that one small file is
    /// downloaded here to read the shard names out of it.</para></summary>
    internal static async Task<IReadOnlyList<AudioModelFile>> ResolveCheckpointFilesAsync(string repo, string category,
        CancellationToken cancel)
    {
        // Probed, not downloaded: fetching the weights here to find out whether they exist would land the
        // primary artifact before its companions, which is the one ordering this whole path guarantees.
        if (await AudioModelCache.ExistsAsync(repo, "model.safetensors", category, ct: cancel).ConfigureAwait(false))
        {
            return [new AudioModelFile("model.safetensors")];
        }
        if (await AudioModelCache.ExistsAsync(repo, "model.safetensors.index.json", category, ct: cancel).ConfigureAwait(false))
        {
            string indexPath = await AudioModelCache.GetAsync(repo, "model.safetensors.index.json", category, ct: cancel).ConfigureAwait(false);
            List<AudioModelFile> files = [new AudioModelFile("model.safetensors.index.json")];
            foreach (string shard in ReadShardNames(indexPath))
            {
                files.Add(new AudioModelFile(shard));
            }
            return files;
        }
        Logs.Debug($"[Audio] '{repo}' has neither a single-file nor a sharded safetensors layout; assuming a pickle.");
        return [new AudioModelFile("pytorch_model.bin")];
    }

    /// <summary>Loads one SUBFOLDER of a repo, fetching only that subfolder's files. Multi-component checkpoints (diffusers-style: <c>transformer/</c>, <c>vocoder/</c>, …) have no weights at the repo root, and pulling the whole repo would drag in every sibling component — for MiniMax Music 3 that is tens of gigabytes of formats the engine never reads.</summary>
    internal static async Task<(IReadOnlyDictionary<string, Tensor> Dict, IDisposable[] Loaders)> LoadSubfolderAsync(
        string repo, string subfolder, string category, CancellationToken cancel)
    {
        // diffusers components ship diffusion_pytorch_model.*; transformers ones ship model.*.
        string[] singleFiles = ["diffusion_pytorch_model.safetensors", "model.safetensors"];
        foreach (string name in singleFiles)
        {
            try
            {
                string path = await AudioModelCache.GetAsync(repo, $"{subfolder}/{name}", category, ct: cancel).ConfigureAwait(false);
                SafeTensorsLoader loader = new SafeTensorsLoader();
                loader.Load(path);
                return (loader.GetAllTensors(), [loader]);
            }
            catch (FileNotFoundException ex)
            {
                Logs.Debug($"[Audio] '{repo}/{subfolder}' has no {name} ({ex.Message}); trying the next layout.");
            }
        }

        foreach (string name in singleFiles)
        {
            string indexName = $"{name}.index.json";
            string indexPath;
            try
            {
                indexPath = await AudioModelCache.GetAsync(repo, $"{subfolder}/{indexName}", category, ct: cancel).ConfigureAwait(false);
            }
            catch (FileNotFoundException ex)
            {
                Logs.Debug($"[Audio] '{repo}/{subfolder}' has no {indexName} ({ex.Message}); trying the next layout.");
                continue;
            }
            Dictionary<string, Tensor> merged = new Dictionary<string, Tensor>(StringComparer.Ordinal);
            List<IDisposable> loaders = [];
            foreach (string shard in ReadShardNames(indexPath))
            {
                string shardPath = await AudioModelCache.GetAsync(repo, $"{subfolder}/{shard}", category, ct: cancel).ConfigureAwait(false);
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
        throw new FileNotFoundException($"'{repo}/{subfolder}' has no safetensors weights (single-file or sharded).");
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

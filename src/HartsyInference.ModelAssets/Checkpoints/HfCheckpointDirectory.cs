using System.Text.Json;
using HartsyInference.Core.IO;
using HartsyInference.ModelAssets.Quant;

namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>Recognises a Hugging Face style checkpoint directory (config plus safetensors weights) without opening any weight file.</summary>
public static class HfCheckpointDirectory
{
    /// <summary>File name of the model config.</summary>
    public const string ConfigFileName = "config.json";

    /// <summary>File name of the safetensors shard index.</summary>
    public const string IndexFileName = "model.safetensors.index.json";

    /// <summary>File name of the fast-tokenizer definition.</summary>
    public const string TokenizerFileName = "tokenizer.json";

    /// <summary>Describes <paramref name="directory"/>, or returns null when it is not such a checkpoint.</summary>
    /// <remarks>Requires a parseable <c>config.json</c> with a <c>model_type</c> and either a shard index or at least one <c>.safetensors</c> file, so a folder that only carries a GGUF plus a stray config is not claimed.</remarks>
    public static HfCheckpointInfo? TryProbe(string? directory)
    {
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory))
            return null;
        string root = Path.GetFullPath(directory);
        string configPath = Path.Combine(root, ConfigFileName);
        if (!File.Exists(configPath))
            return null;

        string? modelType;
        QuantFlavor? flavor;
        try
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(configPath));
            JsonElement config = document.RootElement;
            if (config.ValueKind != JsonValueKind.Object
                || !config.TryGetProperty("model_type", out JsonElement type)
                || type.ValueKind != JsonValueKind.String)
                return null;
            modelType = type.GetString();
            flavor = HfQuantFlavorDetector.FromConfig(config);
        }
        catch (Exception exception) when (exception is JsonException or IOException or UnauthorizedAccessException)
        {
            return null;
        }
        if (string.IsNullOrEmpty(modelType))
            return null;

        string indexPath = Path.Combine(root, IndexFileName);
        bool hasIndex = File.Exists(indexPath);
        if (!hasIndex && !Directory.EnumerateFiles(root, "*.safetensors").Any())
            return null;
        string tokenizerPath = Path.Combine(root, TokenizerFileName);
        return new HfCheckpointInfo(root, configPath, hasIndex ? indexPath : null,
            File.Exists(tokenizerPath) ? tokenizerPath : null, modelType, flavor);
    }

    /// <summary>The safetensors files that make up the checkpoint in <paramref name="directory"/>: every shard a
    /// <c>*.safetensors.index.json</c> names (transformers' <c>model.</c> or diffusers'
    /// <c>diffusion_pytorch_model.</c> index; a canonical index wins over a variant one such as <c>.fp16.</c>), else
    /// every <c>.safetensors</c> file, in name order.</summary>
    /// <exception cref="FileNotFoundException">The directory holds no weights, or an index names a missing shard.</exception>
    public static IReadOnlyList<string> WeightFiles(string directory)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        string root = Path.GetFullPath(directory);
        string? index = Indexes(root).FirstOrDefault();
        if (index is null)
        {
            string[] files = [.. Directory.EnumerateFiles(root, "*.safetensors").Order(StringComparer.Ordinal)];
            return files.Length > 0 ? files : throw new FileNotFoundException($"'{root}' holds no .safetensors weights.");
        }
        return Shards(index, ReadWeightMap(index, root));
    }

    /// <summary>The files holding the checkpoint <paramref name="path"/> names: a folder's <see cref="WeightFiles"/>, and
    /// for a file, the whole shard set of the index in its folder that names it (so picking any one shard of a diffusers
    /// component loads all of them), else the file alone.</summary>
    public static IReadOnlyList<string> CheckpointFiles(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (Directory.Exists(path))
            return WeightFiles(path);
        string file = Path.GetFullPath(path);
        return IndexNaming(file) is { } found ? Shards(found.Index, found.Map) : [file];
    }

    /// <summary>Every tensor name the shard index of the checkpoint holding <paramref name="file"/> maps, or null when
    /// no index in its folder names it. Lets a caller see the whole key set of a sharded checkpoint without opening a
    /// shard: the signature of a model whose distinctive tensors land in different shards.</summary>
    public static IReadOnlyCollection<string>? IndexedTensorNames(string file)
    {
        ArgumentException.ThrowIfNullOrEmpty(file);
        return IndexNaming(Path.GetFullPath(file))?.Map.Keys;
    }

    /// <summary>The root of a diffusers pipeline folder around <paramref name="path"/>: the path itself when it is a
    /// folder holding one of <paramref name="components"/>, else the nearest of up to three ancestors that does (a file
    /// inside <c>transformer/</c> resolves to the folder above it). Null when none does.</summary>
    public static string? FindPipelineRoot(string path, params string[] components)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        ArgumentNullException.ThrowIfNull(components);
        string? directory = Directory.Exists(path) ? Path.GetFullPath(path) : Path.GetDirectoryName(Path.GetFullPath(path));
        for (int depth = 0; depth < 4 && directory is not null; depth++)
        {
            foreach (string component in components)
            {
                if (Directory.Exists(Path.Combine(directory, component)))
                    return directory;
            }
            directory = Path.GetDirectoryName(directory);
        }
        return null;
    }

    /// <summary>The folder's shard indexes, canonical ones (<c>{stem}.safetensors.index.json</c> with a dot-free stem)
    /// before variants such as <c>diffusion_pytorch_model.fp16.safetensors.index.json</c>, each group in name order.</summary>
    private static IEnumerable<string> Indexes(string root)
    {
        const string suffix = ".safetensors.index.json";
        return Directory.EnumerateFiles(root, "*" + suffix)
            .OrderBy(index => Path.GetFileName(index)[..^suffix.Length].Contains('.'))
            .ThenBy(index => index, StringComparer.Ordinal);
    }

    /// <summary>The index in <paramref name="file"/>'s folder whose weight map names it, with that map; null for none.
    /// The file is matched the way the filesystem matches names, so a path typed in another case still finds its set.</summary>
    private static (string Index, Dictionary<string, string> Map)? IndexNaming(string file)
    {
        string? folder = Path.GetDirectoryName(file);
        if (folder is null)
            return null;
        foreach (string index in Indexes(folder))
        {
            Dictionary<string, string> map = ReadWeightMap(index, folder);
            if (map.Values.Contains(file, CaseInsensitivePath.PathComparer))
                return (index, map);
        }
        return null;
    }

    /// <summary>The distinct shards <paramref name="map"/> names, in name order, refusing one that is not on disk.</summary>
    private static string[] Shards(string index, Dictionary<string, string> map)
    {
        string[] shards = [.. map.Values.Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal)];
        foreach (string shard in shards)
        {
            if (!File.Exists(shard))
                throw new FileNotFoundException($"'{Path.GetFileName(index)}' names a shard that is not on disk.", shard);
        }
        return shards;
    }

    /// <summary>The index's <c>weight_map</c> as tensor name → absolute shard path, refusing a shard outside its folder.</summary>
    private static Dictionary<string, string> ReadWeightMap(string index, string root)
    {
        string folder = Path.GetFullPath(root);
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(index));
        Dictionary<string, string> map = new(StringComparer.Ordinal);
        foreach (JsonProperty entry in document.RootElement.GetProperty("weight_map").EnumerateObject())
        {
            string name = entry.Value.GetString() ?? throw new InvalidDataException($"'{index}' maps '{entry.Name}' to no file.");
            string shard = Path.GetFullPath(Path.Combine(folder, name));
            if (!shard.StartsWith(folder + Path.DirectorySeparatorChar, StringComparison.Ordinal))
                throw new InvalidDataException($"'{index}' names a shard outside its folder: '{name}'.");
            map[entry.Name] = shard;
        }
        return map;
    }
}

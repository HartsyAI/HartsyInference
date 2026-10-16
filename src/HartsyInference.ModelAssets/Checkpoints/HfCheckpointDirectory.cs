using System.Text.Json;
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
    /// <c>diffusion_pytorch_model.</c> index), else every <c>.safetensors</c> file, in name order.</summary>
    /// <exception cref="FileNotFoundException">The directory holds no weights, or an index names a missing shard.</exception>
    public static IReadOnlyList<string> WeightFiles(string directory)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        string root = Path.GetFullPath(directory);
        string? index = Directory.EnumerateFiles(root, "*.safetensors.index.json").Order(StringComparer.Ordinal).FirstOrDefault();
        if (index is null)
        {
            string[] files = [.. Directory.EnumerateFiles(root, "*.safetensors").Order(StringComparer.Ordinal)];
            return files.Length > 0 ? files : throw new FileNotFoundException($"'{root}' holds no .safetensors weights.");
        }
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(index));
        SortedSet<string> shards = new(StringComparer.Ordinal);
        foreach (JsonProperty entry in document.RootElement.GetProperty("weight_map").EnumerateObject())
        {
            string name = entry.Value.GetString() ?? throw new InvalidDataException($"'{index}' maps '{entry.Name}' to no file.");
            string path = Path.GetFullPath(Path.Combine(root, name));
            if (!path.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal))
                throw new InvalidDataException($"'{index}' names a shard outside its folder: '{name}'.");
            shards.Add(path);
        }
        foreach (string shard in shards)
        {
            if (!File.Exists(shard))
                throw new FileNotFoundException($"'{Path.GetFileName(index)}' names a shard that is not on disk.", shard);
        }
        return [.. shards];
    }
}

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
}

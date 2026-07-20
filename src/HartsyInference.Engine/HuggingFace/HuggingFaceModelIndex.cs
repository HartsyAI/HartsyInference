using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Models;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Registry;
using HartsyInference.ModelAssets.Registry;

namespace HartsyInference.Engine.HuggingFace;

/// <summary>Parses HuggingFace model card metadata and config.json to determine model architecture, weight format, and build registry entries.</summary>
public sealed class HuggingFaceModelIndex
{
    /// <summary>Reads config.json from the given repository and determines the model architecture.</summary>
    public async Task<string?> ParseModelCardAsync(
        string repoId,
        HuggingFaceClient client,
        CancellationToken ct)
    {
        Logs.Debug($"HuggingFaceModelIndex: parsing model card for \"{repoId}\"");

        IReadOnlyList<HuggingFaceFileInfo> files = await client
            .ListModelFilesAsync(repoId, ct)
            .ConfigureAwait(false);

        bool hasConfig = false;
        foreach (HuggingFaceFileInfo file in files)
        {
            if (string.Equals(file.FileName, "config.json", StringComparison.OrdinalIgnoreCase))
            {
                hasConfig = true;
                break;
            }
        }

        if (!hasConfig)
        {
            Logs.Warning($"HuggingFaceModelIndex: no config.json found in \"{repoId}\"");
            return null;
        }

        string tempDir = Path.Combine(Path.GetTempPath(), "hartsyinference", "config_cache");
        string tempFile = Path.Combine(tempDir, $"{repoId.Replace('/', '_')}_config.json");

        await client.DownloadFileAsync(repoId, "config.json", tempFile, null, sha256: null, ct).ConfigureAwait(false);

        try
        {
            byte[] configBytes = await File.ReadAllBytesAsync(tempFile, ct).ConfigureAwait(false);
            JsonDocument doc = JsonDocument.Parse(configBytes);
            string? architecture = DetectArchitecture(doc.RootElement);

            Logs.Info($"HuggingFaceModelIndex: detected architecture \"{architecture ?? "(unknown)"}\" for \"{repoId}\"");
            return architecture;
        }
        finally
        {
            TryDeleteFile(tempFile);
        }
    }

    /// <summary>Inspects a parsed config.json to determine the model architecture identifier.</summary>
    public string? DetectArchitecture(JsonElement configJson)
    {
        // Check _class_name field (common in diffusers configs)
        if (configJson.TryGetProperty("_class_name", out JsonElement className))
        {
            string? classNameStr = className.GetString();
            if (classNameStr is not null)
            {
                string lower = classNameStr.ToLowerInvariant();

                if (lower.Contains("stableddiffusion3") || lower.Contains("sd3"))
                {
                    return "sd3";
                }

                if (lower.Contains("stableddiffusionxl") || lower.Contains("sdxl"))
                {
                    return "sdxl";
                }

                if (lower.Contains("stableddiffusion") || lower.Contains("stablediffusion"))
                {
                    return "stable-diffusion";
                }

                if (lower.Contains("flux"))
                {
                    return "flux";
                }
            }
        }

        // Check model_type field (common in transformers configs)
        if (configJson.TryGetProperty("model_type", out JsonElement modelType))
        {
            string? modelTypeStr = modelType.GetString();
            if (modelTypeStr is not null)
            {
                string lower = modelTypeStr.ToLowerInvariant();

                if (lower.Contains("whisper"))
                {
                    return "whisper";
                }

                if (lower.Contains("kokoro"))
                {
                    return "kokoro";
                }
            }
        }

        // Check architectures array (transformers standard)
        if (configJson.TryGetProperty("architectures", out JsonElement architectures)
            && architectures.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement arch in architectures.EnumerateArray())
            {
                string? archStr = arch.GetString();
                if (archStr is null)
                {
                    continue;
                }

                string lower = archStr.ToLowerInvariant();

                if (lower.Contains("whisper"))
                {
                    return "whisper";
                }

                if (lower.Contains("yolo"))
                {
                    return "yolo";
                }

                if (lower.Contains("kokoro"))
                {
                    return "kokoro";
                }
            }
        }

        // Check for YOLO-specific fields
        if (configJson.TryGetProperty("backbone", out JsonElement _)
            && configJson.TryGetProperty("nc", out JsonElement _))
        {
            return "yolo";
        }

        // Check _diffusers_version as a fallback hint for diffusion models
        if (configJson.TryGetProperty("_diffusers_version", out JsonElement _))
        {
            // Has diffusers metadata but no specific class matched — generic diffusion
            if (configJson.TryGetProperty("sample_size", out JsonElement _))
            {
                return "stable-diffusion";
            }
        }

        Logs.Debug("HuggingFaceModelIndex: could not determine architecture from config.json");
        return null;
    }

    /// <summary>Determines the model weight format from the file listing, preferring SafeTensors over GGUF.</summary>
    public ModelFormat DetectWeightFormat(IReadOnlyList<HuggingFaceFileInfo> files)
    {
        bool hasSafeTensors = false;
        bool hasGguf = false;
        bool hasOnnx = false;

        foreach (HuggingFaceFileInfo file in files)
        {
            string name = file.FileName;

            if (name.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
            {
                hasSafeTensors = true;
            }
            else if (name.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            {
                hasGguf = true;
            }
            else if (name.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            {
                hasOnnx = true;
            }
        }

        if (hasSafeTensors)
        {
            Logs.Debug("HuggingFaceModelIndex: detected SafeTensors format");
            return ModelFormat.SafeTensors;
        }

        if (hasGguf)
        {
            Logs.Debug("HuggingFaceModelIndex: detected GGUF format");
            return ModelFormat.Gguf;
        }

        if (hasOnnx)
        {
            Logs.Debug("HuggingFaceModelIndex: detected ONNX format");
            return ModelFormat.Onnx;
        }

        Logs.Warning("HuggingFaceModelIndex: no recognized weight format found, defaulting to SafeTensors");
        return ModelFormat.SafeTensors;
    }

    /// <summary>Builds a <see cref="ModelInfo"/> registry entry from HuggingFace metadata, file listing, and a local path.</summary>
    public ModelInfo BuildModelInfo(
        string repoId,
        HuggingFaceModelInfo hubInfo,
        IReadOnlyList<HuggingFaceFileInfo> files,
        string localPath)
    {
        ModelFormat format = DetectWeightFormat(files);
        string? architecture = DetectArchitectureFromTags(hubInfo);

        long totalSize = 0L;
        foreach (HuggingFaceFileInfo file in files)
        {
            totalSize += file.Size;
        }

        DType weightDType = InferDType(files, format);

        ModelInfo info = new ModelInfo
        {
            Id = repoId,
            Name = hubInfo.ModelId,
            Format = format,
            Architecture = architecture,
            WeightDType = weightDType,
            LocalPath = localPath,
            FileSize = totalSize,
            HuggingFaceRepo = repoId,
            LastAccessed = DateTime.UtcNow,
        };

        Logs.Info($"HuggingFaceModelIndex: built ModelInfo for \"{repoId}\" (format={format}, arch=\"{architecture ?? "(unknown)"}\")");
        return info;
    }

    private static string? DetectArchitectureFromTags(HuggingFaceModelInfo hubInfo)
    {
        foreach (string tag in hubInfo.Tags)
        {
            string lower = tag.ToLowerInvariant();

            if (lower.Contains("stable-diffusion-xl") || lower.Contains("sdxl"))
            {
                return "sdxl";
            }

            if (lower.Contains("stable-diffusion-3") || lower.Contains("sd3"))
            {
                return "sd3";
            }

            if (lower.Contains("stable-diffusion"))
            {
                return "stable-diffusion";
            }

            if (lower.Contains("flux"))
            {
                return "flux";
            }

            if (lower.Contains("whisper"))
            {
                return "whisper";
            }

            if (lower.Contains("kokoro"))
            {
                return "kokoro";
            }

            if (lower.Contains("yolo"))
            {
                return "yolo";
            }
        }

        return null;
    }

    private static DType InferDType(IReadOnlyList<HuggingFaceFileInfo> files, ModelFormat format)
    {
        if (format == ModelFormat.Gguf)
        {
            // GGUF files are typically quantized; assume Q4_K as a common default
            foreach (HuggingFaceFileInfo file in files)
            {
                string lower = file.FileName.ToLowerInvariant();
                if (lower.Contains("q8"))
                {
                    return DType.Q8_0;
                }

                if (lower.Contains("q4"))
                {
                    return DType.Q4_K;
                }
            }

            return DType.Q4_K;
        }

        // For safetensors, check if fp16 variant exists
        foreach (HuggingFaceFileInfo file in files)
        {
            if (file.FileName.Contains("fp16", StringComparison.OrdinalIgnoreCase))
            {
                return DType.F16;
            }
        }

        return DType.F16;
    }

    private static void TryDeleteFile(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (Exception ex)
        {
            Logs.Verbose($"Best-effort delete of {path} failed: {ex.Message}");
        }
    }
}

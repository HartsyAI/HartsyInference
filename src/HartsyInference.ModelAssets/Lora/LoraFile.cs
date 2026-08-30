using System.Globalization;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.Lora.Mappers;
using HartsyInference.ModelAssets.MiniMaxH3;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora;

/// <summary>A loaded LoRA safetensors file with its layers parsed into a model-format-agnostic representation. The file's mmap-backed data is owned by this instance — disposing this object invalidates every layer's down/up tensors.</summary>
public sealed class LoraFile : IDisposable
{
    private SafeTensorsLoader? _loader;
    private MiniMaxH3PddTrunkConversion? _miniMaxH3Conversion;
    private int _disposed;

    /// <summary>Path of the loaded safetensors file.</summary>
    public required string FilePath { get; init; }

    /// <summary>Detected LoRA naming format. The rules live in <see cref="LoraFormatDetector.Detect"/>.</summary>
    public required LoraFormat Format { get; init; }

    /// <summary>Parsed LoRA layers, each pairing a canonical target weight key with its down/up matrices.</summary>
    public required IReadOnlyList<LoraLayer> Layers { get; init; }

    /// <summary>Full-weight <c>.diff</c>/<c>.diff_b</c> deltas (Comfy-style Wan repacks); empty for formats without them.</summary>
    public IReadOnlyList<LoraFullWeightDiff> FullWeightDiffs { get; init; } = [];

    /// <summary>Optional per-file training metadata extracted from the safetensors __metadata__ dictionary (e.g. ss_network_module, ss_network_alpha). Null when the file has no metadata block.</summary>
    public IReadOnlyDictionary<string, string>? Metadata { get; init; }

    /// <summary>Opens a LoRA safetensors file, detects its format, and parses every layer into the canonical representation.</summary>
    public static LoraFile Load(string filePath)
    {
        SafeTensorsLoader loader = new();
        MiniMaxH3PddTrunkConversion? miniMaxH3Conversion = null;
        try
        {
            loader.Load(filePath);
            LoraFormat format = LoraFormatDetector.Detect(loader.Descriptors);
            if (format == LoraFormat.Unknown)
            {
                IEnumerable<string> sample = loader.Descriptors.Keys.Take(5);
                throw new HartsyInferenceException(
                    $"Could not detect LoRA format for '{filePath}'. Sample keys: {string.Join(", ", sample)}");
            }

            IReadOnlyList<LoraFullWeightDiff> fullWeightDiffs = [];
            IReadOnlyList<LoraLayer> layers;
            if (format == LoraFormat.DiffusersMiniMaxH3)
            {
                float alpha = ResolveMiniMaxH3Alpha(loader);
                miniMaxH3Conversion = MiniMaxH3PddKeyConverter.Convert(
                    loader.GetAllTensors(), new HashSet<string>(StringComparer.Ordinal), alpha,
                    requireMainAdaln: false);
                layers = miniMaxH3Conversion.Layers;
                fullWeightDiffs = miniMaxH3Conversion.FullWeightDiffs;
            }
            else
            {
                layers = format switch
                {
                    LoraFormat.KohyaSd15 or LoraFormat.KohyaSdxl => KohyaSdMapper.ParseLayers(loader, format),
                    LoraFormat.KohyaFlux => KohyaFluxMapper.ParseLayers(loader),
                    LoraFormat.AiToolkitFlux => AiToolkitFluxMapper.ParseLayers(loader),
                    LoraFormat.DiffusersFlux => DiffusersFluxMapper.ParseLayers(loader),
                    LoraFormat.DiffusersBareDit => DiffusersFluxMapper.ParseLayers(loader, bareRoots: true),
                    LoraFormat.ComfyBflDit => KohyaFluxMapper.ParseLayers(loader, dottedBflRoots: true),
                    LoraFormat.KohyaWan or LoraFormat.DiffusersWan => WanLoraMapper.ParseLayers(loader, format, out fullWeightDiffs),
                    _ => throw new NotSupportedException($"LoRA format {format} parsing not implemented."),
                };
            }

            string diffNote = fullWeightDiffs.Count > 0 ? $", fullWeightDiffs={fullWeightDiffs.Count}" : "";
            Logs.Info($"Loaded LoRA '{Path.GetFileName(filePath)}' (format={format}, layers={layers.Count}{diffNote}).");

            LoraFile file = new()
            {
                FilePath = filePath,
                Format = format,
                Layers = layers,
                FullWeightDiffs = fullWeightDiffs,
                Metadata = loader.Metadata is null
                    ? null
                    : new Dictionary<string, string>(loader.Metadata, StringComparer.Ordinal),
            };
            file._loader = loader;
            file._miniMaxH3Conversion = miniMaxH3Conversion;
            return file;
        }
        catch
        {
            miniMaxH3Conversion?.Dispose();
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Reads the published LightX alpha metadata, falling back to the first adapter rank as PEFT specifies.</summary>
    private static float ResolveMiniMaxH3Alpha(SafeTensorsLoader loader)
    {
        string[] metadataKeys = ["alpha", "network_alpha", "ss_network_alpha"];
        if (loader.Metadata is not null)
        {
            foreach (string key in metadataKeys)
            {
                if (!loader.Metadata.TryGetValue(key, out string? value)) continue;
                if (float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float alpha)
                    && alpha > 0.0f && float.IsFinite(alpha))
                {
                    return alpha;
                }
                throw new HartsyInferenceException(
                    $"MiniMax-H3 Diffusers LoRA metadata '{key}' must be a finite positive number; got '{value}'.");
            }
        }

        foreach ((string key, SafeTensorDescriptor descriptor) in loader.Descriptors)
        {
            if (!key.EndsWith(".lora_A.default.weight", StringComparison.Ordinal)
                && !key.EndsWith(".lora_A.weight", StringComparison.Ordinal))
            {
                continue;
            }
            if (descriptor.Shape.Rank == 2 && descriptor.Shape[0] > 0)
            {
                return descriptor.Shape[0];
            }
        }
        throw new HartsyInferenceException(
            "MiniMax-H3 Diffusers LoRA has no usable alpha metadata or rank-two lora_A matrix.");
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _miniMaxH3Conversion?.Dispose();
        _miniMaxH3Conversion = null;
        _loader?.Dispose();
        _loader = null;
    }
}

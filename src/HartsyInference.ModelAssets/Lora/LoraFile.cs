using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.Lora.Mappers;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora;

/// <summary>A loaded LoRA safetensors file with its layers parsed into a model-format-agnostic representation. The file's mmap-backed data is owned by this instance — disposing this object invalidates every layer's down/up tensors.</summary>
public sealed class LoraFile : IDisposable
{
    private SafeTensorsLoader? _loader;
    private int _disposed;

    /// <summary>Path of the loaded safetensors file.</summary>
    public required string FilePath { get; init; }

    /// <summary>Detected LoRA naming format. The rules live in <see cref="LoraFormatDetector.Detect"/>.</summary>
    public required LoraFormat Format { get; init; }

    /// <summary>Parsed LoRA layers, each pairing a canonical target weight key with its down/up matrices.</summary>
    public required IReadOnlyList<LoraLayer> Layers { get; init; }

    /// <summary>Optional per-file training metadata extracted from the safetensors __metadata__ dictionary (e.g. ss_network_module, ss_network_alpha). Null when the file has no metadata block.</summary>
    public IReadOnlyDictionary<string, string>? Metadata { get; init; }

    /// <summary>Opens a LoRA safetensors file, detects its format, and parses every layer into the canonical representation.</summary>
    public static LoraFile Load(string filePath)
    {
        SafeTensorsLoader loader = new();
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

            IReadOnlyList<LoraLayer> layers = format switch
            {
                LoraFormat.KohyaSd15 or LoraFormat.KohyaSdxl => KohyaSdMapper.ParseLayers(loader, format),
                LoraFormat.KohyaFlux => KohyaFluxMapper.ParseLayers(loader),
                LoraFormat.AiToolkitFlux => AiToolkitFluxMapper.ParseLayers(loader),
                LoraFormat.DiffusersFlux => DiffusersFluxMapper.ParseLayers(loader),
                LoraFormat.DiffusersBareDit => DiffusersFluxMapper.ParseLayers(loader, bareRoots: true),
                LoraFormat.ComfyBflDit => KohyaFluxMapper.ParseLayers(loader, dottedBflRoots: true),
                LoraFormat.KohyaWan or LoraFormat.DiffusersWan => WanLoraMapper.ParseLayers(loader, format),
                _ => throw new NotSupportedException($"LoRA format {format} parsing not implemented."),
            };

            Logs.Info($"Loaded LoRA '{Path.GetFileName(filePath)}' (format={format}, layers={layers.Count}).");

            LoraFile file = new()
            {
                FilePath = filePath,
                Format = format,
                Layers = layers,
                Metadata = null,
            };
            file._loader = loader;
            return file;
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _loader?.Dispose();
        _loader = null;
    }
}

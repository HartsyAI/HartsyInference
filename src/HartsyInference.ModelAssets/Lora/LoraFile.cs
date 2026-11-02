using System.Globalization;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora.Mappers;
using HartsyInference.ModelAssets.MiniMaxH3;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora;

/// <summary>A loaded LoRA safetensors file with its layers parsed into a model-format-agnostic representation. The file's mmap-backed data is owned by this instance — disposing this object invalidates every layer's down/up tensors.</summary>
public sealed class LoraFile : IDisposable
{
    private SafeTensorsLoader? _loader;
    private MiniMaxH3PddTrunkConversion? _miniMaxH3Conversion;
    private List<Tensor> _ownedTensors = [];
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
        // Tensors a mapper had to allocate (a reordered matrix) rather than borrow from the mapping; owned with the file.
        List<Tensor> ownedTensors = [];
        try
        {
            loader.Load(filePath);
            RefuseUnsupportedForms(loader, filePath);
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
            else if (format == LoraFormat.Fibo2)
            {
                layers = Fibo2LoraMapper.ParseLayers(loader, out fullWeightDiffs);
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
                    LoraFormat.ComfyZImageDit => ZImageLoraMapper.ParseLayers(loader),
                    LoraFormat.KohyaWan or LoraFormat.DiffusersWan => WanLoraMapper.ParseLayers(loader, format, out fullWeightDiffs),
                    LoraFormat.Flux3 => Flux3LoraMapper.ParseLayers(loader, ownedTensors),
                    LoraFormat.MossImage => MossImageLoraMapper.ParseLayers(loader),
                    LoraFormat.Anima => AnimaLoraMapper.ParseLayers(loader),
                    LoraFormat.Krea2 => Krea2LoraMapper.ParseLayers(loader),
                    LoraFormat.KohyaGeneric => KohyaGenericMapper.ParseLayers(loader),
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
            file._ownedTensors = ownedTensors;
            return file;
        }
        catch
        {
            foreach (Tensor tensor in ownedTensors) tensor.Dispose();
            miniMaxH3Conversion?.Dispose();
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Refuses a file that carries a LoRA form this engine cannot apply, naming its first such key, and warns once about
    /// adapter tensors under a suffix the engine does not recognize. Runs before format detection and before any mapper, so no
    /// mapper can drop one of these forms silently.</summary>
    private static void RefuseUnsupportedForms(SafeTensorsLoader loader, string filePath)
    {
        string fileName = Path.GetFileName(filePath);
        int refused = 0, unrecognized = 0;
        string? firstRefused = null, firstRefusedReason = null, firstUnrecognized = null;
        foreach (string key in loader.Descriptors.Keys)
        {
            string? reason = LoraRoleSuffix.UnsupportedFormReason(key);
            if (reason is not null)
            {
                refused++;
                firstRefused ??= key;
                firstRefusedReason ??= reason;
            }
            else if (LoraRoleSuffix.IsUnrecognizedAdapterKey(key))
            {
                unrecognized++;
                firstUnrecognized ??= key;
            }
        }
        if (refused > 0)
        {
            throw new NotSupportedException(
                $"LoRA '{fileName}' carries {refused} tensor(s) in a form this engine does not apply; the first is "
                + $"'{firstRefused}', which is {firstRefusedReason}. Applying the rest without it would give a wrong result, "
                + "so the file is refused. Use a LoRA saved in a supported form.");
        }
        if (unrecognized > 0)
        {
            Logs.Warning($"LoRA '{fileName}' has {unrecognized} adapter tensor(s) with a suffix this engine does not recognize; "
                + $"they are not applied (first: '{firstUnrecognized}').");
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
        foreach (Tensor tensor in _ownedTensors) tensor.Dispose();
        _ownedTensors = [];
        _loader?.Dispose();
        _loader = null;
    }
}

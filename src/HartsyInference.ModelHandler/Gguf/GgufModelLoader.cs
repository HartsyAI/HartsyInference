using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.Gguf.Codecs;

namespace HartsyInference.ModelHandler.Gguf;

/// <summary>One entry point for loading any GGUF file in this codebase. Parses the file, detects the architecture, applies the per-architecture key remap, and returns a tensor dict in the format the model's existing <c>*CheckpointConverter.Convert</c> already expects.
///
/// <para><b>Tensor lifecycle</b>: tensors stay in their on-disk quantized dtype by default — caller is responsible for dequantizing via <see cref="GgufCodecRegistry"/> or by calling <see cref="LoadDequantized"/> instead. The mmap stays open for the lifetime of the returned <see cref="LoadedGgufModel"/>; disposing it invalidates every tensor reference.</para></summary>
public sealed class GgufModelLoader : IDisposable
{
    private GgufLoader? _loader;
    private int _disposed;

    public required GgufLoader UnderlyingLoader { get; init; }
    public required IGgufKeyMapper Mapper { get; init; }
    public required IReadOnlyDictionary<string, Tensor> Weights { get; init; }
    public required string Architecture { get; init; }
    public GgufMetadata Metadata => UnderlyingLoader.Metadata;

    /// <summary>Result wrapper exposed by <see cref="Load"/>.</summary>
    public sealed class LoadedGgufModel : IDisposable
    {
        private GgufLoader? _loader;
        private int _disposed;

        public required IGgufKeyMapper Mapper { get; init; }
        public required IReadOnlyDictionary<string, Tensor> Weights { get; init; }
        public required string Architecture { get; init; }
        public required GgufMetadata Metadata { get; init; }

        internal void AttachLoader(GgufLoader loader) => _loader = loader;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            _loader?.Dispose();
            _loader = null;
        }
    }

    /// <summary>Opens the GGUF, detects architecture, and produces the remapped tensor dict in (still-quantized) lazy form. Caller dequantizes per-tensor via <see cref="GgufCodecRegistry"/> when needed.</summary>
    public static LoadedGgufModel Load(string path)
    {
        GgufLoader loader = new();
        try
        {
            loader.Load(path);

            string architecture = ResolveArchitecture(loader);
            IGgufKeyMapper mapper = ResolveMapper(loader, architecture);

            Dictionary<string, Tensor> remapped = new(StringComparer.Ordinal);
            int dropped = 0;
            foreach (KeyValuePair<string, GgufTensorDescriptor> kv in loader.Descriptors)
            {
                string? targetKey = mapper.MapKey(kv.Key);
                if (targetKey is null) { dropped++; continue; }
                if (remapped.ContainsKey(targetKey))
                    throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                        $"GGUF key remap collision: GGUF '{kv.Key}' → '{targetKey}' (already present from a prior key).");
                remapped[targetKey] = loader.GetTensor(kv.Key);
            }

            Logs.Info($"GGUF loaded: arch={mapper.Architecture}, tensors={remapped.Count} (dropped {dropped} metadata-only), file={Path.GetFileName(path)}.");

            LoadedGgufModel result = new()
            {
                Mapper = mapper,
                Weights = remapped,
                Architecture = mapper.Architecture,
                Metadata = loader.Metadata,
            };
            result.AttachLoader(loader);
            return result;
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Convenience wrapper that loads + immediately dequantizes every quantized tensor to <paramref name="targetDtype"/>. Use this when the rest of the pipeline can't yet handle quantized inputs (current default — GPU dequant on the fly is Phase D).</summary>
    public static (Dictionary<string, Tensor> weights, LoadedGgufModel handle) LoadDequantized(string path, DType targetDtype)
    {
        if (targetDtype != DType.F32 && targetDtype != DType.F16)
            throw new ArgumentException("LoadDequantized supports F32 or F16 only.", nameof(targetDtype));

        LoadedGgufModel raw = Load(path);
        try
        {
            Dictionary<string, Tensor> output = new(StringComparer.Ordinal);
            int quantized = 0, passthrough = 0;
            foreach (KeyValuePair<string, Tensor> kv in raw.Weights)
            {
                Tensor src = kv.Value;
                if (src.DType.IsQuantized)
                {
                    output[kv.Key] = GgufDequantizer.Dequantize(src, targetDtype);
                    quantized++;
                }
                else if (src.DType != targetDtype && src.DType.IsFloatingPoint)
                {
                    output[kv.Key] = src.CastTo(targetDtype);
                    passthrough++;
                }
                else
                {
                    output[kv.Key] = src.CastTo(src.DType);
                    passthrough++;
                }
            }
            Logs.Info($"GGUF dequantized: {quantized} quantized + {passthrough} passthrough → {targetDtype}.");
            return (output, raw);
        }
        catch
        {
            raw.Dispose();
            throw;
        }
    }

    private static string ResolveArchitecture(GgufLoader loader)
    {
        if (loader.Metadata.ContainsKey("general.architecture"))
        {
            string arch = loader.Metadata.GetString("general.architecture") ?? "";
            if (!string.IsNullOrEmpty(arch)) return arch.ToLowerInvariant();
        }
        return string.Empty;
    }

    private static IGgufKeyMapper ResolveMapper(GgufLoader loader, string architecture)
    {
        if (!string.IsNullOrEmpty(architecture))
        {
            IGgufKeyMapper? hit = GgufKeyMapperRegistry.GetByArchitecture(architecture);
            if (hit is not null)
            {
                Logs.Info($"GGUF: matched architecture '{architecture}' → {hit.GetType().Name}.");
                return hit;
            }
            Logs.Warning($"GGUF: declared architecture '{architecture}' has no registered mapper; falling back to key-heuristic detection.");
        }
        IGgufKeyMapper detected = GgufKeyMapperRegistry.DetectByKeys(loader.Descriptors.Keys.ToList());
        Logs.Info($"GGUF: heuristic-detected architecture → {detected.Architecture} ({detected.GetType().Name}).");
        return detected;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _loader?.Dispose();
        _loader = null;
    }
}

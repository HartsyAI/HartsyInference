using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf.Codecs;

namespace HartsyInference.ModelAssets.Gguf;

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
        /// <summary>The GGUF's own <c>general.architecture</c> (e.g. "qwen2", "gemma3"), or the resolved mapper's name when the file declared none. This is the real model architecture, not the mapper that handles it (one mapper, e.g. the llama-family mapper, serves several architectures).</summary>
        public required string Architecture { get; init; }
        /// <summary>The display name of the <see cref="IGgufKeyMapper"/> that remapped this file's tensor keys. Often differs from <see cref="Architecture"/> (e.g. arch "qwen2" handled by the "llama"-family mapper).</summary>
        public required string MapperName { get; init; }
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
            bool declaredArchRegistered = string.IsNullOrEmpty(architecture) || GgufKeyMapperRegistry.GetByArchitecture(architecture) is not null;
            IGgufKeyMapper mapper = ResolveMapper(loader, architecture);

            Dictionary<string, Tensor> remapped = new(StringComparer.Ordinal);
            int dropped = 0;
            try
            {
                foreach (KeyValuePair<string, GgufTensorDescriptor> kv in loader.Descriptors)
                {
                    string? targetKey = mapper.MapKey(kv.Key);
                    if (targetKey is null) { dropped++; continue; }
                    if (remapped.ContainsKey(targetKey))
                        throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                            $"GGUF key remap collision: GGUF '{kv.Key}' → '{targetKey}' (already present from a prior key).");
                    remapped[targetKey] = loader.GetTensor(kv.Key);
                }
            }
            catch (Exception ex) when (!declaredArchRegistered)
            {
                // A declared-but-unregistered architecture falls through to heuristic key-matching (see
                // ResolveMapper), which can pick a structurally incompatible mapper — wrong tensor counts/shapes
                // then surface as an opaque ArgumentOutOfRangeException/IndexOutOfRangeException deep in
                // GgufLoader.GetTensor. Translate that into a clear "not supported" error instead of a confusing crash.
                string msg = $"GGUF declares architecture '{architecture}' which has no registered key mapper in this " +
                    $"engine. Heuristic key-matching selected '{mapper.Architecture}' ({mapper.GetType().Name}), but " +
                    $"that mapping is structurally incompatible with this file: {ex.Message}";
                Logs.Error(msg, ex);
                throw new HartsyInference.Core.Exceptions.UnsupportedModelException(msg, architecture, "gguf");
            }

            string realArch = string.IsNullOrEmpty(architecture) ? mapper.Architecture : architecture;
            string mapperNote = string.Equals(realArch, mapper.Architecture, StringComparison.OrdinalIgnoreCase)
                ? "" : $" (mapper={mapper.Architecture})";
            Logs.Info($"GGUF loaded: arch={realArch}{mapperNote}, tensors={remapped.Count} (dropped {dropped} metadata-only), file={Path.GetFileName(path)}.");

            LoadedGgufModel result = new()
            {
                Mapper = mapper,
                Weights = remapped,
                Architecture = realArch,
                MapperName = mapper.Architecture,
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

    /// <summary>Relabels every rank-2 tensor's shape from GGUF's <c>[in, out]</c> (ggml <c>ne</c>) order to the <c>[out, in]</c> order the rest of the engine assumes for a matrix weight (matmul reads <c>N=Shape[0]</c>, <c>K=Shape[1]</c>; embeddings/heads are <c>[vocab, hidden]</c>). The underlying data is already row-major <c>[out, in]</c> — identical to an HF safetensors weight — so this is a pure metadata swap (a <see cref="Tensor.Reshape"/> that keeps borrowing the GGUF mmap, valid for quantized dtypes too since it touches no bytes). Diffusion GGUF converters must run their input through this before mapping keys, exactly as <c>GgufLanguageModel</c> does for LLM weights; skipping it leaves every Linear transposed and the first matmul derives a degenerate <c>M=0</c>.</summary>
    public static Dictionary<string, Tensor> RelabelRank2ToPyTorchOrder(IReadOnlyDictionary<string, Tensor> ggufWeights)
    {
        Dictionary<string, Tensor> relabeled = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> kv in ggufWeights)
        {
            Tensor t = kv.Value;
            if (t.Shape.Rank == 2)
                t = t.Reshape(new TensorShape((int)t.Shape[1], (int)t.Shape[0]));
            relabeled[kv.Key] = t;
        }
        return relabeled;
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

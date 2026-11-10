using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Models.MossVl;

/// <summary>How a MOSS-VL text decoder is loaded.</summary>
public sealed record MossVlLoadOptions
{
    /// <summary>The backend that will run the decoder. Projections keep their stored precision where it can run them:
    /// BF16 stays BF16 on a BF16 backend and becomes F16 elsewhere, and GGUF quants stay packed where the backend has a kernel
    /// (the rest widen to F16). CPU gets F32. Null keeps the host-only load: every tensor F32 except, with
    /// <see cref="KeepQuantized"/>, the packed GGUF projections.</summary>
    public IBackend? Backend { get; init; }

    /// <summary>Only when <see cref="Backend"/> is null: keeps GGUF block projections and <c>lm_head</c> packed.</summary>
    public bool KeepQuantized { get; init; }

    /// <summary>Routes packed projections through <c>QuantizedMatMul</c> instead of a cached widened copy (CUDA and Vulkan).</summary>
    public bool LowVramQuant { get; init; }

    /// <summary>Loads only layers <c>[0, LayerLimit)</c> (for layer-level parity runs); null loads the whole stack.</summary>
    public int? LayerLimit { get; init; }

    /// <summary>A GGUF block type the checkpoint's plain float projections are re-quantized to (see
    /// <see cref="DenseTextWeightPolicy.ParsePlainPrecision"/>), where the backend keeps it packed; null keeps them as stored.
    /// Quantized projections (NF4, FP8, GPTQ, …) follow <see cref="DenseTextWeightPolicy"/> either way.</summary>
    public DType? PlainPrecision { get; init; }

    /// <summary>A GGUF block type quantized projections (NF4, FP8, GPTQ, …) load as, instead of the width-matched default
    /// (<see cref="DenseTextWeightPolicy"/>); null uses the default.</summary>
    public DType? QuantizedPrecision { get; init; }

    /// <summary>Where a checkpoint folder's converted weights are cached; null uses <c>_gguf-cache</c> inside the folder when it is
    /// writable and <c>~/.cache/hartsyinference/quant-cache</c> otherwise (see <see cref="GgufQuantCache.DefaultPath"/>).</summary>
    public string? QuantCachePath { get; init; }
}

/// <summary>MOSS-VL's text model: a <see cref="GenericTransformer"/> over the checkpoint's <c>model.language_model.*</c> weights and
/// <c>lm_head</c>, plus the learned vision separator that the vision sequence ends each image with.</summary>
/// <remarks>Two weight sources load the same model: the safetensors checkpoint (<see cref="Open(string, MossVlConfig, MossVlLoadOptions)"/>)
/// and a MOSS-VL GGUF (<see cref="OpenGguf(string, MossVlConfig, MossVlLoadOptions)"/>). Given a backend, projections keep their stored
/// precision where that backend can run them (see <see cref="MossVlLoadOptions.Backend"/>); norms, gates, the embedding (a host gather)
/// and the separator are always F32. Converted copies are owned here and released on <see cref="Dispose"/>. Positions come from
/// <see cref="MossVlPositions"/>; vision features come from the vision tower.</remarks>
public sealed class MossVlTextDecoder : IDisposable
{
    private const string Prefix = "model.language_model";
    private const string SeparatorKey = "model.separator_token";
    private const string LmHeadKey = "lm_head.weight";

    private readonly GenericTransformer _model;
    private readonly IDisposable _weightOwner;
    private readonly List<IDisposable> _converted;
    private readonly Tensor _separator;
    private int _disposed;

    private MossVlTextDecoder(TransformerConfig text, GenericTransformer model, Tensor separator, IDisposable weightOwner,
        List<IDisposable> converted)
    {
        TextConfig = text;
        _model = model;
        _separator = separator;
        _weightOwner = weightOwner;
        _converted = converted;
    }

    /// <summary>The decoder's configuration. After a layer-limited open, <see cref="TransformerConfig.NumLayers"/> is the limit.</summary>
    public TransformerConfig TextConfig { get; }

    /// <summary>The text model: embedding, decoder layers, final norm and <c>lm_head</c>. Drive it with <see cref="GenericTransformer.ForwardEmbeds"/>
    /// and <see cref="GenericTransformer.ProjectLogits"/>.</summary>
    /// <exception cref="ObjectDisposedException">The decoder is disposed.</exception>
    public GenericTransformer Model
    {
        get
        {
            ThrowIfDisposed();
            return _model;
        }
    }

    /// <summary>Opens the text model of a MOSS-VL safetensors checkpoint directory on the host, every tensor widened to F32.</summary>
    /// <param name="checkpointDirectory">A directory holding <c>config.json</c> and the safetensors shards.</param>
    /// <param name="config">The parsed <c>config.json</c>.</param>
    /// <param name="layerLimit">Loads only layers <c>[0, layerLimit)</c> (for layer-level parity runs); null loads the whole stack.</param>
    /// <exception cref="InvalidDataException">The checkpoint lacks the text embedding, <c>lm_head</c> or separator.</exception>
    public static MossVlTextDecoder Open(string checkpointDirectory, MossVlConfig config, int? layerLimit = null) =>
        Open(checkpointDirectory, config, new MossVlLoadOptions { LayerLimit = layerLimit });

    /// <summary>Opens the text model of a MOSS-VL safetensors checkpoint directory for <see cref="MossVlLoadOptions.Backend"/>.</summary>
    /// <exception cref="InvalidDataException">The checkpoint lacks the text embedding, <c>lm_head</c> or separator.</exception>
    public static MossVlTextDecoder Open(string checkpointDirectory, MossVlConfig config, MossVlLoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ValidateLayerLimit(config, options.LayerLimit);   // before any shard is mapped, so a bad limit leaks nothing
        if (HfQuantCheckpoint.DeclaresQuantization(checkpointDirectory) || options.PlainPrecision is not null)
            return OpenConverted(checkpointDirectory, config, options);
        CheckpointSource source = CheckpointSource.OpenShards(HfCheckpointDirectory.WeightFiles(checkpointDirectory));
        return Build(source.Weights, source, config, options, ggufLayout: false, checkpointDirectory);
    }

    /// <summary>Bumped when the converted-weight cache's layout or policy changes.</summary>
    internal const int CacheVersion = 1;

    /// <summary>A quantized checkpoint folder (bitsandbytes NF4, compressed-tensors FP8, GPTQ, …) or a plain one with
    /// <see cref="MossVlLoadOptions.PlainPrecision"/>: every text weight is read through <see cref="HfQuantCheckpoint"/> at the dtype
    /// <see cref="DenseTextWeightPolicy"/> picks for the backend, with the converted ones cached once on disk.</summary>
    private static MossVlTextDecoder OpenConverted(string directory, MossVlConfig config, MossVlLoadOptions options)
    {
        int layers = options.LayerLimit ?? config.Text.NumLayers;
        bool Include(string key) =>
            key == SeparatorKey || key == LmHeadKey || (key.StartsWith(Prefix + ".", StringComparison.Ordinal) && !IsAboveLimit(key, layers));
        DType Target(HfQuantWeight w) => DenseTextWeightPolicy.Target(w,
            projection: w.Shape.Rank == 2 && (IsBlockProjection(w.Name) || w.Name == LmHeadKey), options.Backend, options.PlainPrecision,
            options.QuantizedPrecision);

        string plain = options.PlainPrecision?.Name.ToLowerInvariant() ?? "native";
        string quantized = options.QuantizedPrecision?.Name.ToLowerInvariant() ?? "auto";
        // The dtype a projection gets depends on which block types the backend keeps packed, so two backends sharing a folder
        // each keep their own cache instead of rebuilding it for one another.
        string backendKey = BackendKey(options.Backend);
        string cachePath = options.QuantCachePath ?? GgufQuantCache.DefaultPath(directory,
            $"moss-vl-text-{backendKey}-{plain}-{quantized}-{layers}l-v{CacheVersion}.gguf");
        Dictionary<string, object> identity = new(StringComparer.Ordinal)
        {
            ["hartsy.moss_vl.cache_version"] = CacheVersion,
            ["hartsy.moss_vl.backend"] = backendKey,
            ["hartsy.moss_vl.plain_precision"] = plain,
            ["hartsy.moss_vl.quantized_precision"] = quantized,
        };
        HfQuantCheckpoint checkpoint = HfQuantCheckpoint.Open(directory);
        IReadOnlyDictionary<string, Tensor> weights;
        IDisposable owner;
        try
        {
            (weights, owner) = HfQuantWeightCache.Load(checkpoint, Target, cachePath, "moss_vl", "[MOSS-VL]", identity, Include,
                GgufQuantCache.ParallelProduceAhead);
        }
        catch
        {
            checkpoint.Dispose();
            throw;
        }
        return Build(weights, owner, config, options, ggufLayout: false, directory);
    }

    /// <summary>Names the backend class for the converted-weight cache: the backend's name plus which block types it keeps packed
    /// (what <see cref="DenseTextWeightPolicy"/> chooses between); "host" for none or a CPU backend.</summary>
    private static string BackendKey(IBackend? backend)
    {
        if (backend is null or CpuBackend) return "host";
        string name = new([.. backend.Capabilities.Name.ToLowerInvariant().Select(static c => char.IsAsciiLetterOrDigit(c) ? c : '-')]);
        DType[] types = [DType.Q4_0, DType.Q4_K, DType.Q5_K, DType.Q6_K, DType.Q8_0];
        return $"{name}-{string.Concat(types.Select(t => backend.SupportsResidentQuant(t) ? '1' : '0'))}";
    }

    /// <summary>Opens the text model of a MOSS-VL GGUF file (the <c>moss_vl</c> architecture, mapped by <see cref="MossVlKeyMapper"/>).</summary>
    /// <param name="ggufPath">The GGUF file, text blocks plus any vision blocks it carries (the vision blocks are not read here).</param>
    /// <param name="config">The text configuration, from the checkpoint's <c>config.json</c>; the GGUF supplies the weights only.</param>
    /// <param name="layerLimit">Loads only layers <c>[0, layerLimit)</c>; null loads the whole stack.</param>
    /// <param name="keepQuantized">True keeps the block projections and <c>lm_head</c> in their GGUF quant. Use it only on a backend
    /// that reads those formats (CUDA, Vulkan for Q4_K/Q6_K). False widens every tensor to F32, which the CPU backend requires.</param>
    /// <param name="lowVramQuant">True routes quantized projections through <c>QuantizedMatMul</c>.</param>
    public static MossVlTextDecoder OpenGguf(string ggufPath, MossVlConfig config, int? layerLimit = null, bool keepQuantized = false,
        bool lowVramQuant = false) =>
        OpenGguf(ggufPath, config, new MossVlLoadOptions { LayerLimit = layerLimit, KeepQuantized = keepQuantized, LowVramQuant = lowVramQuant });

    /// <summary>Opens the text model of a MOSS-VL GGUF file for <see cref="MossVlLoadOptions.Backend"/>.</summary>
    public static MossVlTextDecoder OpenGguf(string ggufPath, MossVlConfig config, MossVlLoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ValidateLayerLimit(config, options.LayerLimit);
        GgufModelLoader.LoadedGgufModel handle = GgufModelLoader.Load(ggufPath);
        try
        {
            return Build(handle.Weights, handle, config, options, ggufLayout: true, ggufPath);
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }

    /// <summary>Appends the learned vision separator to one image's merged tokens: <paramref name="merged"/> is <c>[N, hidden]</c> F32,
    /// the result <c>[1, N + 1, hidden]</c> (caller owns). The separator follows the grid, as in the modeling code's convert-to-batch.</summary>
    public Tensor AppendSeparator(Tensor merged) => AppendSeparators([merged]);

    /// <summary>Appends the separator after each image's merged tokens and concatenates the blocks in prompt order: each
    /// <paramref name="merged"/> entry is <c>[N_k, hidden]</c> F32, the result <c>[1, Σ(N_k + 1), hidden]</c> (caller owns).</summary>
    public unsafe Tensor AppendSeparators(IReadOnlyList<Tensor> merged)
    {
        ThrowIfDisposed();
        int h = TextConfig.HiddenSize;
        long total = 0;
        foreach (Tensor m in merged)
        {
            if (m.DType != DType.F32 || m.Shape.Rank != 2 || m.Shape[1] != h)
                throw new ArgumentException($"Merged vision tokens must be F32 [N, {h}], got {m.Shape} {m.DType}.", nameof(merged));
            total += m.Shape[0] + 1;
        }
        Tensor output = new(new TensorShape(1, total, h), DType.F32);
        float* dst = (float*)output.DataPointer;
        long at = 0;
        foreach (Tensor m in merged)
        {
            long n = m.Shape[0];
            long bytes = n * h * sizeof(float);
            Buffer.MemoryCopy((void*)m.DataPointer, dst + at * h, bytes, bytes);
            at += n;
            Buffer.MemoryCopy((void*)_separator.DataPointer, dst + at * h, h * sizeof(float), h * sizeof(float));
            at++;
        }
        return output;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _model.Dispose();
        foreach (IDisposable t in _converted) t.Dispose();
        _weightOwner.Dispose();
    }

    /// <summary>The shared load: keeps the text keys for the layer limit, lays GGUF tensors out as <c>[out, in]</c>, picks each
    /// tensor's storage for the backend, and builds the transformer. <paramref name="weightOwner"/> is disposed with the model (or on
    /// failure).</summary>
    private static MossVlTextDecoder Build(IReadOnlyDictionary<string, Tensor> weights, IDisposable weightOwner, MossVlConfig config,
        MossVlLoadOptions options, bool ggufLayout, string origin)
    {
        ValidateLayerLimit(config, options.LayerLimit);
        int layers = options.LayerLimit ?? config.Text.NumLayers;
        TransformerConfig text = layers == config.Text.NumLayers
            ? config.Text
            : config.Text with { NumLayers = layers, CrossAttnLayers = config.Text.CrossAttnLayers.Where(i => i < layers).ToHashSet() };
        if (options.LowVramQuant) text = text with { LowVramQuant = true };
        IBackend? backend = options.Backend;
        bool device = backend is not null and not CpuBackend;

        List<IDisposable> converted = [];
        GenericTransformer? model = null;
        try
        {
            Dictionary<string, Tensor> w = new(StringComparer.Ordinal);
            foreach (KeyValuePair<string, Tensor> kv in weights)
            {
                string key = kv.Key;
                if (key != SeparatorKey && key != LmHeadKey && !(key.StartsWith(Prefix + ".", StringComparison.Ordinal) && !IsAboveLimit(key, layers)))
                    continue;
                Tensor t = kv.Value;
                if (ggufLayout && t.Shape.Rank == 2)
                    t = t.Reshape(new TensorShape((int)t.Shape[1], (int)t.Shape[0]));
                Tensor stored = Store(key, t, backend, device, options.KeepQuantized);
                if (!ReferenceEquals(stored, t)) converted.Add(stored);
                w[key] = stored;
            }
            if (!w.ContainsKey($"{Prefix}.embed_tokens.weight") || !w.ContainsKey(LmHeadKey) || !w.ContainsKey(SeparatorKey))
                throw new InvalidDataException($"'{origin}' holds no MOSS-VL text embedding, lm_head or separator.");
            // Packed weights the device has no kernel for are widened to F16 here, at load, rather than failing mid-GEMM.
            if (device) converted.Add(QuantizedWeightPolicy.PrepareForBackend(w, backend!, DType.F16));

            model = new GenericTransformer(text);
            model.LoadWeights(w, Prefix, LmHeadKey);
            return new MossVlTextDecoder(text, model, w[SeparatorKey], weightOwner, converted);
        }
        catch
        {
            model?.Dispose();
            foreach (IDisposable t in converted) t.Dispose();
            weightOwner.Dispose();
            throw;
        }
    }

    /// <summary>The storage one tensor keeps. Rank-2 block projections and <c>lm_head</c> keep their precision where the backend runs
    /// it; everything else (embedding, norms, gates, separator) is F32.</summary>
    internal static Tensor Store(string key, Tensor t, IBackend? backend, bool device, bool keepQuantized)
    {
        bool projection = t.Shape.Rank == 2 && (IsBlockProjection(key) || key == LmHeadKey);
        if (!projection)
            return t.DType.IsQuantized ? GgufDequantizer.Dequantize(t, DType.F32) : TensorCasts.EnsureF32(t);
        if (t.DType.IsQuantized)
        {
            bool keep = backend is null ? keepQuantized : device;
            return keep ? t : GgufDequantizer.Dequantize(t, DType.F32);
        }
        return backend is null ? TensorCasts.EnsureF32(t) : DenseWeightCasts.ForBackend(t, backend, key);
    }

    private static void ValidateLayerLimit(MossVlConfig config, int? layerLimit)
    {
        if (layerLimit is { } limit && (limit < 1 || limit > config.Text.NumLayers))
            throw new ArgumentOutOfRangeException(nameof(layerLimit), layerLimit, $"Layer limit must be in [1, {config.Text.NumLayers}].");
    }

    /// <summary>True for a <c>model.language_model.layers.N.(self_attn|cross_attn|mlp).*.weight</c> key.</summary>
    private static bool IsBlockProjection(string key) =>
        key.StartsWith(Prefix + ".layers.", StringComparison.Ordinal) && key.EndsWith(".weight", StringComparison.Ordinal)
        && (key.Contains(".self_attn.", StringComparison.Ordinal) || key.Contains(".cross_attn.", StringComparison.Ordinal)
            || key.Contains(".mlp.", StringComparison.Ordinal));

    /// <summary>True for a <c>model.language_model.layers.N.*</c> key with <c>N ≥ layers</c>.</summary>
    private static bool IsAboveLimit(string key, int layers)
    {
        const string LayerPrefix = Prefix + ".layers.";
        if (!key.StartsWith(LayerPrefix, StringComparison.Ordinal)) return false;
        int dot = key.IndexOf('.', LayerPrefix.Length);
        return int.TryParse(key.AsSpan(LayerPrefix.Length, dot - LayerPrefix.Length), out int index) && index >= layers;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(MossVlTextDecoder));
    }
}

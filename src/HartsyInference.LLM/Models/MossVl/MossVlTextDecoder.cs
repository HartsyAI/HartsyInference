using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Models.MossVl;

/// <summary>MOSS-VL's text model: a <see cref="GenericTransformer"/> over the checkpoint's <c>model.language_model.*</c> weights and
/// <c>lm_head</c>, plus the learned vision separator that the vision sequence ends each image with.</summary>
/// <remarks>Two weight sources load the same model: the safetensors checkpoint (<see cref="Open"/>, BF16 widened to F32) and a MOSS-VL
/// GGUF (<see cref="OpenGguf"/>, quantized projections kept packed on a quantized-capable backend, everything else widened to F32).
/// Converted copies are owned here and released on <see cref="Dispose"/>. Positions come from <see cref="MossVlPositions"/>; vision
/// features come from the vision tower.</remarks>
public sealed class MossVlTextDecoder : IDisposable
{
    private const string Prefix = "model.language_model";
    private const string SeparatorKey = "model.separator_token";
    private const string LmHeadKey = "lm_head.weight";

    private readonly GenericTransformer _model;
    private readonly IDisposable _weightOwner;
    private readonly List<Tensor> _converted;
    private readonly Tensor _separator;
    private int _disposed;

    private MossVlTextDecoder(TransformerConfig text, GenericTransformer model, Tensor separator, IDisposable weightOwner,
        List<Tensor> converted)
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

    /// <summary>Opens the text model of a MOSS-VL safetensors checkpoint directory.</summary>
    /// <param name="checkpointDirectory">A directory holding <c>config.json</c> and the safetensors shards.</param>
    /// <param name="config">The parsed <c>config.json</c>.</param>
    /// <param name="layerLimit">Loads only layers <c>[0, layerLimit)</c> (for layer-level parity runs); null loads the whole stack.</param>
    /// <exception cref="InvalidDataException">The checkpoint lacks the text embedding, <c>lm_head</c> or separator.</exception>
    public static MossVlTextDecoder Open(string checkpointDirectory, MossVlConfig config, int? layerLimit = null)
    {
        ValidateLayerLimit(config, layerLimit);   // before any shard is mapped, so a bad limit leaks nothing
        CheckpointSource source = CheckpointSource.OpenShards(HfCheckpointDirectory.WeightFiles(checkpointDirectory));
        return Build(source.Weights, source, config, layerLimit, ggufLayout: false, keepQuantized: false, lowVramQuant: false,
            checkpointDirectory);
    }

    /// <summary>Opens the text model of a MOSS-VL GGUF file (the <c>moss_vl</c> architecture, mapped by <see cref="MossVlKeyMapper"/>).</summary>
    /// <param name="ggufPath">The GGUF file, text blocks plus any vision blocks it carries (the vision blocks are not read here).</param>
    /// <param name="config">The text configuration, from the checkpoint's <c>config.json</c>; the GGUF supplies the weights only.</param>
    /// <param name="layerLimit">Loads only layers <c>[0, layerLimit)</c>; null loads the whole stack.</param>
    /// <param name="keepQuantized">True keeps the block projections and <c>lm_head</c> in their GGUF quant. Use it only on a backend
    /// that reads those formats (CUDA, Vulkan for Q4_K/Q6_K). False widens every tensor to F32, which the CPU backend requires.</param>
    /// <param name="lowVramQuant">True routes quantized projections through <c>QuantizedMatMul</c> (CUDA only). Leave false on Vulkan,
    /// where <c>Linear</c> runs the packed GEMV.</param>
    public static MossVlTextDecoder OpenGguf(string ggufPath, MossVlConfig config, int? layerLimit = null, bool keepQuantized = false,
        bool lowVramQuant = false)
    {
        GgufModelLoader.LoadedGgufModel handle = GgufModelLoader.Load(ggufPath);
        try
        {
            return Build(handle.Weights, handle, config, layerLimit, ggufLayout: true, keepQuantized, lowVramQuant, ggufPath);
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }

    /// <summary>Appends the learned vision separator to one image's merged tokens: <paramref name="merged"/> is <c>[N, hidden]</c> F32,
    /// the result <c>[1, N + 1, hidden]</c> (caller owns). The separator follows the grid, as in the modeling code's convert-to-batch.</summary>
    public unsafe Tensor AppendSeparator(Tensor merged)
    {
        ThrowIfDisposed();
        int n = (int)merged.Shape[0], h = TextConfig.HiddenSize;
        if (merged.DType != DType.F32 || merged.Shape.Rank != 2 || merged.Shape[1] != h)
            throw new ArgumentException($"Merged vision tokens must be F32 [N, {h}], got {merged.Shape} {merged.DType}.", nameof(merged));
        Tensor output = new(new TensorShape(1, n + 1, h), DType.F32);
        long bytes = (long)n * h * sizeof(float);
        float* dst = (float*)output.DataPointer;
        Buffer.MemoryCopy((void*)merged.DataPointer, dst, bytes, bytes);
        Buffer.MemoryCopy((void*)_separator.DataPointer, dst + (long)n * h, h * sizeof(float), h * sizeof(float));
        return output;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _model.Dispose();
        foreach (Tensor t in _converted) t.Dispose();
        _weightOwner.Dispose();
    }

    /// <summary>The shared load: keeps the text keys for the layer limit, lays GGUF tensors out as <c>[out, in]</c>, widens what the
    /// backend cannot read, and builds the transformer. <paramref name="weightOwner"/> is disposed with the model (or on failure).</summary>
    private static MossVlTextDecoder Build(IReadOnlyDictionary<string, Tensor> weights, IDisposable weightOwner, MossVlConfig config,
        int? layerLimit, bool ggufLayout, bool keepQuantized, bool lowVramQuant, string origin)
    {
        ValidateLayerLimit(config, layerLimit);
        int layers = layerLimit ?? config.Text.NumLayers;
        TransformerConfig text = layers == config.Text.NumLayers
            ? config.Text
            : config.Text with { NumLayers = layers, CrossAttnLayers = config.Text.CrossAttnLayers.Where(i => i < layers).ToHashSet() };
        if (lowVramQuant) text = text with { LowVramQuant = true };

        List<Tensor> converted = [];
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
                // Block projections and lm_head stay packed when the backend reads them; the embedding (host gather), norms,
                // gates and separator are always F32.
                bool packed = keepQuantized && t.DType.IsQuantized && (IsBlockProjection(key) || key == LmHeadKey);
                Tensor f32 = packed ? t : t.DType.IsQuantized ? GgufDequantizer.Dequantize(t, DType.F32) : TensorCasts.EnsureF32(t);
                if (!ReferenceEquals(f32, t)) converted.Add(f32);
                w[key] = f32;
            }
            if (!w.ContainsKey($"{Prefix}.embed_tokens.weight") || !w.ContainsKey(LmHeadKey) || !w.ContainsKey(SeparatorKey))
                throw new InvalidDataException($"'{origin}' holds no MOSS-VL text embedding, lm_head or separator.");

            model = new GenericTransformer(text);
            model.LoadWeights(w, Prefix, LmHeadKey);
            return new MossVlTextDecoder(text, model, w[SeparatorKey], weightOwner, converted);
        }
        catch
        {
            model?.Dispose();
            foreach (Tensor t in converted) t.Dispose();
            weightOwner.Dispose();
            throw;
        }
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

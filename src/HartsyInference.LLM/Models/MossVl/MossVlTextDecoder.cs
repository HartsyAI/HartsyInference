using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.LLM.Models.MossVl;

/// <summary>MOSS-VL's text model: a <see cref="GenericTransformer"/> over the checkpoint's <c>model.language_model.*</c> weights and
/// <c>lm_head</c>, plus the learned vision separator that the vision sequence ends each image with.</summary>
/// <remarks>The CPU backend computes in F32, so every tensor is widened once at load; the copies are owned here and released on
/// <see cref="Dispose"/>. Positions come from <see cref="MossVlPositions"/>; vision features come from the vision tower.</remarks>
public sealed class MossVlTextDecoder : IDisposable
{
    private const string Prefix = "model.language_model";
    private const string SeparatorKey = "model.separator_token";
    private const string LmHeadKey = "lm_head.weight";

    private readonly CheckpointSource _source;
    private readonly List<Tensor> _converted;
    private readonly Tensor _separator;
    private int _disposed;

    private MossVlTextDecoder(TransformerConfig text, GenericTransformer model, Tensor separator, CheckpointSource source,
        List<Tensor> converted)
    {
        TextConfig = text;
        Model = model;
        _separator = separator;
        _source = source;
        _converted = converted;
    }

    /// <summary>The decoder's configuration. After a layer-limited <see cref="Open"/>, <see cref="TransformerConfig.NumLayers"/> is the limit.</summary>
    public TransformerConfig TextConfig { get; }

    /// <summary>The text model: embedding, decoder layers, final norm and <c>lm_head</c>. Drive it with <see cref="GenericTransformer.ForwardEmbeds"/>
    /// and <see cref="GenericTransformer.ProjectLogits"/>.</summary>
    public GenericTransformer Model { get; }

    /// <summary>Opens the text model of a MOSS-VL checkpoint directory.</summary>
    /// <param name="checkpointDirectory">A directory holding <c>config.json</c> and the safetensors shards.</param>
    /// <param name="config">The parsed <c>config.json</c>.</param>
    /// <param name="layerLimit">Loads only layers <c>[0, layerLimit)</c> (for layer-level parity runs); null loads the whole stack.</param>
    /// <exception cref="InvalidDataException">The checkpoint lacks the text embedding, <c>lm_head</c> or separator.</exception>
    public static MossVlTextDecoder Open(string checkpointDirectory, MossVlConfig config, int? layerLimit = null)
    {
        int layers = layerLimit ?? config.Text.NumLayers;
        if (layers < 1 || layers > config.Text.NumLayers)
            throw new ArgumentOutOfRangeException(nameof(layerLimit), layerLimit, $"Layer limit must be in [1, {config.Text.NumLayers}].");
        TransformerConfig text = layers == config.Text.NumLayers
            ? config.Text
            : config.Text with { NumLayers = layers, CrossAttnLayers = config.Text.CrossAttnLayers.Where(i => i < layers).ToHashSet() };

        CheckpointSource source = CheckpointSource.OpenShards(HfCheckpointDirectory.WeightFiles(checkpointDirectory));
        List<Tensor> converted = [];
        GenericTransformer? model = null;
        try
        {
            Dictionary<string, Tensor> w = new(StringComparer.Ordinal);
            foreach ((string key, Tensor value) in source.Weights)
            {
                if (key != SeparatorKey && key != LmHeadKey && !(key.StartsWith(Prefix + ".", StringComparison.Ordinal) && !IsAboveLimit(key, layers)))
                    continue;
                Tensor f32 = TensorCasts.EnsureF32(value);
                if (!ReferenceEquals(f32, value)) converted.Add(f32);
                w[key] = f32;
            }
            if (!w.ContainsKey($"{Prefix}.embed_tokens.weight") || !w.ContainsKey(LmHeadKey) || !w.ContainsKey(SeparatorKey))
                throw new InvalidDataException($"'{checkpointDirectory}' holds no MOSS-VL text embedding, lm_head or separator.");

            model = new GenericTransformer(text);
            model.LoadWeights(w, Prefix, LmHeadKey);
            return new MossVlTextDecoder(text, model, w[SeparatorKey], source, converted);
        }
        catch
        {
            model?.Dispose();
            foreach (Tensor t in converted) t.Dispose();
            source.Dispose();
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
        Model.Dispose();
        foreach (Tensor t in _converted) t.Dispose();
        _source.Dispose();
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(MossVlTextDecoder));
    }

    /// <summary>True for a <c>model.language_model.layers.N.*</c> key with <c>N ≥ layers</c>.</summary>
    private static bool IsAboveLimit(string key, int layers)
    {
        const string LayerPrefix = Prefix + ".layers.";
        if (!key.StartsWith(LayerPrefix, StringComparison.Ordinal)) return false;
        int dot = key.IndexOf('.', LayerPrefix.Length);
        return int.TryParse(key.AsSpan(LayerPrefix.Length, dot - LayerPrefix.Length), out int index) && index >= layers;
    }
}

using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.LLM.Models.MossVl;

/// <summary>MOSS-VL's text decoder: the headless decoder body (self-attention and gated cross-attention layers), with the token
/// embedding and the learned vision separator that the outer model owns. Positions come from <see cref="MossVlPositions"/> and
/// the vision features from <see cref="MossVlVisionTower"/>.</summary>
/// <remarks>The CPU backend computes in F32, so every tensor is widened once at load; the copies are owned here and released on
/// <see cref="Dispose"/>. <c>lm_head</c> is not loaded yet: the logits projection belongs with generation.</remarks>
public sealed class MossVlTextDecoder : IDisposable
{
    private const string Prefix = "model.language_model";
    private const string SeparatorKey = "model.separator_token";

    private readonly CheckpointSource _source;
    private readonly List<Tensor> _converted;
    private readonly Tensor _embed;
    private readonly Tensor _separator;
    private int _disposed;

    private MossVlTextDecoder(TransformerConfig text, GenericTransformer body, Tensor embed, Tensor separator,
        CheckpointSource source, List<Tensor> converted)
    {
        TextConfig = text;
        Body = body;
        _embed = embed;
        _separator = separator;
        _source = source;
        _converted = converted;
    }

    /// <summary>The decoder's configuration. After a layer-limited <see cref="Open"/>, <see cref="TransformerConfig.NumLayers"/> is the limit.</summary>
    public TransformerConfig TextConfig { get; }

    /// <summary>The headless decoder body: the layers and the final norm. Drive it with <see cref="GenericTransformer.ForwardEmbeds"/>.</summary>
    public GenericTransformer Body { get; }

    /// <summary>The hidden width of the decoder.</summary>
    public int HiddenSize => TextConfig.HiddenSize;

    /// <summary>Opens the text decoder of a MOSS-VL checkpoint directory.</summary>
    /// <param name="checkpointDirectory">A directory holding <c>config.json</c> and the safetensors shards.</param>
    /// <param name="config">The parsed <c>config.json</c>.</param>
    /// <param name="layerLimit">Loads only layers <c>[0, layerLimit)</c> (for layer-level parity runs); null loads the whole stack.</param>
    /// <exception cref="InvalidDataException">The checkpoint holds no <c>model.language_model.*</c> weights, or lacks the embedding or separator.</exception>
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
        GenericTransformer? body = null;
        try
        {
            Dictionary<string, Tensor> w = new(StringComparer.Ordinal);
            foreach ((string key, Tensor value) in source.Weights)
            {
                if (key != SeparatorKey && !(key.StartsWith(Prefix + ".", StringComparison.Ordinal) && !IsAboveLimit(key, layers)))
                    continue;
                Tensor f32 = TensorCasts.EnsureF32(value);
                if (!ReferenceEquals(f32, value)) converted.Add(f32);
                w[key] = f32;
            }
            if (!w.ContainsKey($"{Prefix}.embed_tokens.weight") || !w.ContainsKey(SeparatorKey))
                throw new InvalidDataException($"'{checkpointDirectory}' holds no MOSS-VL text embedding or separator.");

            body = new GenericTransformer(text);
            body.LoadWeightsHeadless(w, Prefix);
            return new MossVlTextDecoder(text, body, w[$"{Prefix}.embed_tokens.weight"], w[SeparatorKey], source, converted);
        }
        catch
        {
            body?.Dispose();
            foreach (Tensor t in converted) t.Dispose();
            source.Dispose();
            throw;
        }
    }

    /// <summary>Gathers the embedding rows of <paramref name="tokenIds"/> into a fresh <c>[1, T, hidden]</c> F32 tensor (caller owns).</summary>
    public unsafe Tensor EmbedTokens(ReadOnlySpan<int> tokenIds)
    {
        ThrowIfDisposed();
        int h = HiddenSize;
        int vocab = (int)_embed.Shape[0];
        Tensor output = new(new TensorShape(1, tokenIds.Length, h), DType.F32);
        float* src = (float*)_embed.DataPointer;
        float* dst = (float*)output.DataPointer;
        for (int s = 0; s < tokenIds.Length; s++)
        {
            int id = tokenIds[s];
            if ((uint)id >= (uint)vocab)
            {
                output.Dispose();
                throw new ArgumentOutOfRangeException(nameof(tokenIds), id, $"Token {id} is outside the {vocab}-token vocabulary.");
            }
            Buffer.MemoryCopy(src + (long)id * h, dst + (long)s * h, h * sizeof(float), h * sizeof(float));
        }
        return output;
    }

    /// <summary>Appends the learned vision separator to one image's merged tokens: <paramref name="merged"/> is <c>[N, hidden]</c> F32,
    /// the result <c>[1, N + 1, hidden]</c> (caller owns). The separator follows the grid, as in the modeling code's convert-to-batch.</summary>
    public unsafe Tensor AppendSeparator(Tensor merged)
    {
        ThrowIfDisposed();
        int n = (int)merged.Shape[0], h = HiddenSize;
        if (merged.DType != DType.F32 || merged.Shape.Rank != 2 || merged.Shape[1] != h)
            throw new ArgumentException($"Merged vision tokens must be F32 [N, {h}], got {merged.Shape} {merged.DType}.", nameof(merged));
        Tensor output = new(new TensorShape(1, n + 1, h), DType.F32);
        long bytes = (long)n * h * sizeof(float);
        float* dst = (float*)output.DataPointer;
        Buffer.MemoryCopy((void*)merged.DataPointer, dst, bytes, bytes);
        Buffer.MemoryCopy((void*)_separator.DataPointer, dst + (long)n * h, h * sizeof(float), h * sizeof(float));
        return output;
    }

    /// <summary>A fresh KV cache for <see cref="TextConfig"/>'s layers, one sequence, up to <paramref name="maxSequenceLength"/> tokens.</summary>
    public FixedKvCache NewKvCache(int maxSequenceLength) =>
        new(TextConfig.NumLayers, 1, TextConfig.NumKvHeads, TextConfig.HeadDim, maxSequenceLength);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Body.Dispose();
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

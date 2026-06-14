using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>T5 v1.1 XXL encoder-only model for SD3 and Flux text conditioning. 24 encoder blocks with RMSNorm, self-attention with learned relative position bias, and GEGLU FFN. All linear layers are bias-free.</summary>
public sealed unsafe class T5TextEncoder : IDisposable
{
    private readonly T5TextEncoderConfig _config;
    private readonly T5Block[] _blocks;
    // T5 v1.1 mode (single shared bias): _positionBias is non-null, _perLayerPositionBias is null.
    // UMT5 mode (per-layer bias for Pile-T5-XL / AuraFlow): _perLayerPositionBias is non-null with one entry per block.
    private readonly T5RelativePositionBias? _positionBias;
    private readonly T5RelativePositionBias[]? _perLayerPositionBias;

    // Embedding table [vocabSize, dModel]
    private Tensor? _embedWeight;

    // Final RMSNorm
    private Tensor? _finalNormWeight;

    private int _disposed;

    public T5TextEncoder(T5TextEncoderConfig config)
    {
        _config = config;
        _blocks = new T5Block[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            _blocks[i] = new T5Block(config.DModel, config.NumHeads, config.DKv, config.LayerNormEpsilon);
        }
        if (config.UsePerLayerPositionBias)
        {
            _perLayerPositionBias = new T5RelativePositionBias[config.NumLayers];
            for (int i = 0; i < config.NumLayers; i++)
            {
                _perLayerPositionBias[i] = new T5RelativePositionBias(config.RelativeAttentionNumBuckets, config.RelativeAttentionMaxDistance, config.NumHeads);
            }
        }
        else
        {
            _positionBias = new T5RelativePositionBias(config.RelativeAttentionNumBuckets, config.RelativeAttentionMaxDistance, config.NumHeads);
        }
    }

    /// <summary>Loads all encoder weights from a tensor dictionary (HuggingFace safetensors format).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        // Token embeddings — may be under "shared.weight" or "encoder.embed_tokens.weight"
        // Auto-cast to F32 if needed (EmbeddingLookup uses float* directly)
        Tensor? rawEmbed = null;
        if (weights.TryGetValue("encoder.embed_tokens.weight", out Tensor? embedTokens))
        {
            rawEmbed = embedTokens;
        }
        else if (weights.TryGetValue("shared.weight", out Tensor? shared))
        {
            rawEmbed = shared;
        }
        else
        {
            throw new KeyNotFoundException("T5 embedding weights not found (expected 'encoder.embed_tokens.weight' or 'shared.weight')");
        }
        _embedWeight = rawEmbed.DType != DType.F32 ? rawEmbed.CastTo(DType.F32) : rawEmbed;

        // Relative position bias. T5 v1.1 stores it once on block 0 and shares across all layers.
        // UMT5 (Pile-T5-XL / AuraFlow) stores a separate learned bias per layer — we must load all
        // 24 of them, otherwise layers 1..N-1 silently inherit block 0's bias and the encoder
        // produces hidden states with the wrong positional structure (image high-quality, prompt ignored).
        if (_perLayerPositionBias is not null)
        {
            for (int i = 0; i < _config.NumLayers; i++)
            {
                Tensor perLayerBias = weights[$"encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight"];
                _perLayerPositionBias[i].LoadWeights(perLayerBias);
            }
        }
        else
        {
            Tensor biasTable = weights["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"];
            _positionBias!.LoadWeights(biasTable);
        }

        // Per-block weights
        for (int i = 0; i < _config.NumLayers; i++)
        {
            _blocks[i].LoadWeights(weights, $"encoder.block.{i}");
        }

        // Final layer norm — must be F32 (goes through backend.RmsNorm with F32 input)
        Tensor rawFinalNorm = weights["encoder.final_layer_norm.weight"];
        _finalNormWeight = rawFinalNorm.DType != DType.F32 ? rawFinalNorm.CastTo(DType.F32) : rawFinalNorm;

        Logs.Verbose($"T5 encoder loaded: {_config.NumLayers} layers, d_model={_config.DModel}, {_config.NumHeads} heads");
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embedWeight is not null) yield return _embedWeight;
        if (_finalNormWeight is not null) yield return _finalNormWeight;
        if (_perLayerPositionBias is not null)
        {
            for (int i = 0; i < _perLayerPositionBias.Length; i++)
                foreach (Tensor w in _perLayerPositionBias[i].EnumerateWeights()) yield return w;
        }
        else
        {
            foreach (Tensor w in _positionBias!.EnumerateWeights()) yield return w;
        }
        for (int i = 0; i < _blocks.Length; i++)
        {
            foreach (Tensor w in _blocks[i].EnumerateWeights()) yield return w;
        }
    }

    /// <summary>Encodes token IDs to contextualized embeddings. Returns [B, seqLen, dModel] tensor.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="tokenIds">Batch of token ID arrays, each of length seqLen.</param>
    /// <param name="attentionMasks">Optional batch of attention masks (1=attend, 0=pad). If null, all positions attend.</param>
    public Tensor Encode(IBackend backend, int[][] tokenIds, int[][]? attentionMasks = null) =>
        EncodeAtLayer(backend, tokenIds, _config.NumLayers, applyFinalNorm: true, attentionMasks);

    /// <summary>Runs the encoder for the first <paramref name="layerCount"/> blocks (1-indexed; pass <see cref="T5TextEncoderConfig.NumLayers"/> for the full encoder), optionally re-applying the final RMSNorm. Used by F-Lite, which conditions on <c>hidden_states[17]</c> with the final norm + dropout reapplied (dropout is a no-op in eval mode, so we skip it). Pass <paramref name="layerCount"/> = 17 to replicate F-Lite's <c>return_index = -8</c> on a 24-layer T5-XXL.</summary>
    public Tensor EncodeAtLayer(IBackend backend, int[][] tokenIds, int layerCount, bool applyFinalNorm, int[][]? attentionMasks = null)
    {
        ThrowIfDisposed();
        if (layerCount <= 0 || layerCount > _config.NumLayers)
            throw new ArgumentOutOfRangeException(nameof(layerCount), $"layerCount must be in [1, {_config.NumLayers}], got {layerCount}.");

        int batch = tokenIds.Length;
        int seqLen = tokenIds[0].Length;

        Tensor hidden = EmbeddingLookup(tokenIds, batch, seqLen);

        Tensor? maskTensor = attentionMasks is not null ? BuildMaskTensor(attentionMasks, batch, seqLen) : null;

        // T5 v1.1: compute bias once and reuse across layers (cheaper, identical to diffusers).
        // UMT5: each layer has its own learned bias — fetched per block.
        // Never dispose the returned bias tensors — T5RelativePositionBias.ComputeBias caches them
        // internally and the instance owns the lifetime. Disposing here corrupts the cache and the
        // next encode call (e.g. CFG negative prompt) hits an ObjectDisposedException.
        Tensor? sharedBias = _perLayerPositionBias is null ? _positionBias!.ComputeBias(seqLen) : null;

        for (int i = 0; i < layerCount; i++)
        {
            Tensor layerBias = sharedBias ?? _perLayerPositionBias![i].ComputeBias(seqLen);
            Tensor blockOutput = _blocks[i].Forward(backend, hidden, layerBias, maskTensor);
            hidden.Dispose();
            hidden = blockOutput;
        }

        maskTensor?.Dispose();

        if (!applyFinalNorm) return hidden;

        TensorShape outShape = new TensorShape(batch, seqLen, _config.DModel);
        Tensor output = new Tensor(outShape, DType.F32);
        backend.RmsNorm(output, hidden, _finalNormWeight!, _config.LayerNormEpsilon);
        hidden.Dispose();
        return output;
    }

    /// <summary>Looks up token embeddings from the embedding table.</summary>
    private Tensor EmbeddingLookup(int[][] tokenIds, int batch, int seqLen)
    {
        TensorShape shape = new TensorShape(batch, seqLen, _config.DModel);
        Tensor output = new Tensor(shape, DType.F32);

        float* embedPtr = (float*)_embedWeight!.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int tokenId = tokenIds[b][s];
                int srcOffset = tokenId * _config.DModel;
                int dstOffset = (b * seqLen + s) * _config.DModel;

                for (int d = 0; d < _config.DModel; d++)
                {
                    outPtr[dstOffset + d] = embedPtr[srcOffset + d];
                }
            }
        }

        return output;
    }

    /// <summary>Converts attention mask arrays to a [B, seqLen] float tensor.</summary>
    private static Tensor BuildMaskTensor(int[][] masks, int batch, int seqLen)
    {
        TensorShape shape = new TensorShape(batch, seqLen);
        Tensor tensor = new Tensor(shape, DType.F32);
        float* ptr = (float*)tensor.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                ptr[b * seqLen + s] = masks[b][s];
            }
        }

        return tensor;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    /// <summary>Disposes the encoder, freeing all weight tensors. Call this after computing embeddings to reclaim ~10GB for the MMDiT.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            // Note: weight tensors are owned by the tensor dictionary (mmap or loaded),
            // so we don't dispose them here — the caller manages their lifetime.
            // We just clear our references.
            _embedWeight = null;
            _finalNormWeight = null;
        }
    }
}

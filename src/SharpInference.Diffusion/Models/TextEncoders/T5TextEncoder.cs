using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.TextEncoders;

/// <summary>T5 v1.1 XXL encoder-only model for SD3 and Flux text conditioning. 24 encoder blocks with RMSNorm, self-attention with learned relative position bias, and GEGLU FFN. All linear layers are bias-free.</summary>
public sealed unsafe class T5TextEncoder : IDisposable
{
    private readonly T5TextEncoderConfig _config;
    private readonly T5Block[] _blocks;
    private readonly T5RelativePositionBias _positionBias;

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
        _positionBias = new T5RelativePositionBias(config.RelativeAttentionNumBuckets, config.RelativeAttentionMaxDistance, config.NumHeads);
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

        // Relative position bias (only on first block's attention layer)
        Tensor biasTable = weights["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"];
        _positionBias.LoadWeights(biasTable);

        // Per-block weights
        for (int i = 0; i < _config.NumLayers; i++)
        {
            _blocks[i].LoadWeights(weights, $"encoder.block.{i}");
        }

        // Final layer norm — must be F32 (goes through backend.RmsNorm with F32 input)
        Tensor rawFinalNorm = weights["encoder.final_layer_norm.weight"];
        _finalNormWeight = rawFinalNorm.DType != DType.F32 ? rawFinalNorm.CastTo(DType.F32) : rawFinalNorm;

        Logs.Info($"T5 encoder loaded: {_config.NumLayers} layers, d_model={_config.DModel}, {_config.NumHeads} heads");
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embedWeight is not null) yield return _embedWeight;
        if (_finalNormWeight is not null) yield return _finalNormWeight;
        foreach (Tensor w in _positionBias.EnumerateWeights()) yield return w;
        for (int i = 0; i < _blocks.Length; i++)
        {
            foreach (Tensor w in _blocks[i].EnumerateWeights()) yield return w;
        }
    }

    /// <summary>Encodes token IDs to contextualized embeddings. Returns [B, seqLen, dModel] tensor.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="tokenIds">Batch of token ID arrays, each of length seqLen.</param>
    /// <param name="attentionMasks">Optional batch of attention masks (1=attend, 0=pad). If null, all positions attend.</param>
    public Tensor Encode(IBackend backend, int[][] tokenIds, int[][]? attentionMasks = null)
    {
        ThrowIfDisposed();

        int batch = tokenIds.Length;
        int seqLen = tokenIds[0].Length;

        // 1. Embedding lookup
        Tensor hidden = EmbeddingLookup(tokenIds, batch, seqLen);

        // 2. Build attention mask tensor if provided
        Tensor? maskTensor = null;
        if (attentionMasks is not null)
        {
            maskTensor = BuildMaskTensor(attentionMasks, batch, seqLen);
        }

        // 3. Compute relative position bias once (shared across all layers)
        Tensor positionBias = _positionBias.ComputeBias(seqLen);

        // 4. Run through 24 encoder blocks
        for (int i = 0; i < _config.NumLayers; i++)
        {
            Tensor blockOutput = _blocks[i].Forward(backend, hidden, positionBias, maskTensor);
            hidden.Dispose();
            hidden = blockOutput;
        }

        maskTensor?.Dispose();

        // 5. Final RMSNorm
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

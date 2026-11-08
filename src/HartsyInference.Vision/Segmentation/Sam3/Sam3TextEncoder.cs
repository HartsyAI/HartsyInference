using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>SAM 3 text tower: token and position embeddings, a causal pre-norm CLIP transformer (24 blocks, width 1024, exact GELU),
/// <c>ln_final</c>, then the resizer Linear to the decoder width. Returns <see cref="Sam3TextFeatures"/> batch-first.
/// The checkpoint's <c>encoder.text_projection</c> is not loaded: upstream computes it and discards the result (G1).
/// Padding is derived from token id 0 (G2), and the causal mask is applied without a key-padding mask (G3).</summary>
public sealed class Sam3TextEncoder
{
    private readonly int _width;
    private readonly int _contextLength;
    private readonly int _vocabSize;
    private readonly int _dModel;
    private readonly float _eps;
    private readonly Tensor _tokenEmbedding;
    private readonly Tensor _positionEmbedding;
    private readonly Sam3TextTransformerLayer[] _layers;
    private readonly Tensor _lnFinalWeight;
    private readonly Tensor _lnFinalBias;
    private readonly Tensor _resizerWeight;
    private readonly Tensor _resizerBias;

    /// <summary>Loads the text tower. <paramref name="weights"/> are keyed relative to <c>detector.backbone.language_backbone.</c>
    /// (for example <c>encoder.token_embedding.weight</c> and <c>resizer.weight</c>); the dictionary keeps ownership of them.</summary>
    /// <exception cref="ArgumentException">The config's text geometry is inconsistent.</exception>
    /// <exception cref="InvalidDataException">A required weight is missing or has the wrong dtype or shape; the message names the key.</exception>
    public Sam3TextEncoder(Sam3Config config, IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        if (config.TextWidth <= 0 || config.TextHeads <= 0 || config.TextWidth % config.TextHeads != 0)
        {
            throw new ArgumentException($"Text width {config.TextWidth} must be a positive multiple of {config.TextHeads} heads.", nameof(config));
        }

        if (config.TextContextLength < 2 || config.TextLayers <= 0 || config.TextVocabSize <= 0 || config.DModel <= 0)
        {
            throw new ArgumentException("Text context (at least 2), layers, vocabulary and model width must be positive.", nameof(config));
        }

        _width = config.TextWidth;
        _contextLength = config.TextContextLength;
        _vocabSize = config.TextVocabSize;
        _dModel = config.DModel;
        _eps = config.LayerNormEps;

        _tokenEmbedding = Sam3WeightAccess.Require(weights, "encoder.token_embedding.weight", _vocabSize, _width);
        _positionEmbedding = Sam3WeightAccess.Require(weights, "encoder.positional_embedding", _contextLength, _width);

        _layers = new Sam3TextTransformerLayer[config.TextLayers];
        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i] = new Sam3TextTransformerLayer(config, weights, $"encoder.transformer.resblocks.{i}.");
        }

        _lnFinalWeight = Sam3WeightAccess.Require(weights, "encoder.ln_final.weight", _width);
        _lnFinalBias = Sam3WeightAccess.Require(weights, "encoder.ln_final.bias", _width);
        _resizerWeight = Sam3WeightAccess.Require(weights, "resizer.weight", _dModel, _width);
        _resizerBias = Sam3WeightAccess.Require(weights, "resizer.bias", _dModel);
    }

    /// <summary>Encodes one prompt's token ids. <paramref name="tokenIds"/> must be exactly the text context length, as produced by
    /// <see cref="Sam3Tokenizer.Encode"/>. The padding mask is <c>id == 0</c>; the causal mask is the only attention mask.</summary>
    /// <returns>Memory <c>[1, ContextLength, DModel]</c> F32 owned by the caller, and the padding mask.</returns>
    /// <exception cref="ArgumentException">The id count is not the context length, or an id is outside the vocabulary.</exception>
    public Sam3TextFeatures Forward(IBackend backend, int[] tokenIds)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(tokenIds);
        if (tokenIds.Length != _contextLength)
        {
            throw new ArgumentException($"Expected {_contextLength} token ids, got {tokenIds.Length}.", nameof(tokenIds));
        }

        bool[] paddingMask = new bool[_contextLength];
        for (int i = 0; i < tokenIds.Length; i++)
        {
            int id = tokenIds[i];
            if ((uint)id >= (uint)_vocabSize)
            {
                throw new ArgumentException($"Token id {id} at position {i} is outside the vocabulary of {_vocabSize}.", nameof(tokenIds));
            }

            paddingMask[i] = id == Sam3Tokenizer.PaddingId;
        }

        TensorShape seqShape = new TensorShape(1, _contextLength, _width);
        Tensor hidden = new Tensor(seqShape, DType.F32);
        Tensor causalMask = BuildCausalMask(_contextLength);
        try
        {
            EmbedTokens(hidden, tokenIds);
            foreach (Sam3TextTransformerLayer layer in _layers)
            {
                Tensor next = layer.Forward(backend, hidden, causalMask);
                hidden.Dispose();
                hidden = next;
            }

            Tensor normed = new Tensor(seqShape, DType.F32);
            backend.LayerNorm(normed, hidden, _lnFinalWeight, _lnFinalBias, _eps);
            hidden.Dispose();
            hidden = normed;

            Tensor memory = new Tensor(new TensorShape(1, _contextLength, _dModel), DType.F32);
            backend.Linear(memory, hidden, _resizerWeight, _resizerBias);
            return new Sam3TextFeatures(memory, paddingMask);
        }
        finally
        {
            hidden.Dispose();
            causalMask.Dispose();
        }
    }

    /// <summary>Token embedding lookup plus the learned position embedding, written into <paramref name="output"/> <c>[1, S, width]</c>.</summary>
    private void EmbedTokens(Tensor output, int[] tokenIds)
    {
        ReadOnlySpan<float> table = _tokenEmbedding.AsReadOnlySpan<float>();
        ReadOnlySpan<float> positions = _positionEmbedding.AsReadOnlySpan<float>();
        Span<float> destination = output.AsSpan<float>();
        for (int s = 0; s < tokenIds.Length; s++)
        {
            int row = tokenIds[s] * _width;
            int slot = s * _width;
            for (int h = 0; h < _width; h++)
            {
                destination[slot + h] = table[row + h] + positions[slot + h];
            }
        }
    }

    /// <summary>Additive causal mask <c>[S, S]</c>: 0 on and below the diagonal, negative infinity above it (upstream <c>triu(1)</c>).</summary>
    private static Tensor BuildCausalMask(int seqLen)
    {
        Tensor mask = new Tensor(new TensorShape(seqLen, seqLen), DType.F32);
        Span<float> values = mask.AsSpan<float>();
        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j < seqLen; j++)
            {
                values[i * seqLen + j] = j <= i ? 0f : float.NegativeInfinity;
            }
        }
        return mask;
    }
}

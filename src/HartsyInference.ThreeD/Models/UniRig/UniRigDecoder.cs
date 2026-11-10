using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>
/// The OPT decoder of the skeleton stage, hosted on <see cref="GenericTransformer"/>: learned positions (OPT's two-row
/// offset is dropped at load), LayerNorm with biases before attention and the ReLU feed-forward, a separate <c>lm_head</c>.
/// Conditioning tokens enter as input embeddings, so the model never sees token ids for them.
/// </summary>
public sealed class UniRigDecoder : IDisposable
{
    private const string Decoder = "transformer.model.decoder";
    private const int PositionOffset = 2;

    private readonly IBackend _backend;
    private readonly UniRigDecoderConfig _config;
    private readonly GenericTransformer _transformer;
    private readonly Tensor _embedTokens;
    private readonly List<Tensor> _owned = [];
    private FixedKvCache? _cache;
    private int _length;
    private bool _disposed;

    /// <param name="weights">Checkpoint tensors by module path; they must outlive this decoder.</param>
    public UniRigDecoder(IBackend backend, UniRigDecoderConfig config, IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        config.Validate();
        _backend = backend;
        _config = config;
        _transformer = new GenericTransformer(new TransformerConfig
        {
            HiddenSize = config.HiddenSize,
            NumLayers = config.NumLayers,
            NumHeads = config.NumHeads,
            NumKvHeads = config.NumHeads,
            HeadDim = config.HiddenSize / config.NumHeads,
            IntermediateSize = config.FfnSize,
            VocabSize = config.VocabSize,
            MaxPositionEmbeddings = config.MaxPositions,
            RmsNormEps = 1e-5f,
            AttentionBias = true,
            FfnBias = true,
            GatedFfn = false,
            Activation = ActivationKind.Relu,
            UseLayerNorm = true,
            AbsolutePositionEmbeddings = true,
            TieWordEmbeddings = false,
        });
        _embedTokens = UniRigCheckpoint.Get(weights, $"{Decoder}.embed_tokens.weight", config.VocabSize, config.HiddenSize);
        _transformer.LoadWeights(Remap(weights), "model", "lm_head.weight");
    }

    /// <summary>Tokens currently held in the key/value cache.</summary>
    public int Length => _length;

    /// <summary>Starts a sequence from conditioning embeddings plus start tokens and returns the logits after the last row (<c>[VocabSize]</c>).</summary>
    /// <param name="condition">Conditioning rows <c>[1, T, HiddenSize]</c>, already projected to the decoder width.</param>
    /// <param name="startTokens">Token ids appended after the conditioning rows (BOS, class).</param>
    public float[] Begin(Tensor condition, ReadOnlySpan<int> startTokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (condition.Shape.Rank != 3 || condition.Shape[0] != 1 || condition.Shape[2] != _config.HiddenSize)
        {
            throw new ArgumentException($"UniRig decoder condition must be [1, T, {_config.HiddenSize}]; got {condition.Shape}.");
        }

        int rows = (int)condition.Shape[1] + startTokens.Length;
        if (rows >= _config.MaxPositions)
        {
            throw new HartsyInferenceException($"UniRig prompt of {rows} rows leaves no room under {_config.MaxPositions} positions.");
        }

        _cache?.Dispose();
        _cache = new FixedKvCache(_config.NumLayers, 1, _config.NumHeads, _config.HiddenSize / _config.NumHeads, _config.MaxPositions);
        using Tensor embeds = new(new TensorShape(1, rows, _config.HiddenSize), DType.F32);
        Span<float> target = embeds.AsSpan<float>();
        condition.AsSpan<float>().CopyTo(target);
        int hidden = _config.HiddenSize;
        int conditionRows = (int)condition.Shape[1];
        ReadOnlySpan<float> table = _embedTokens.AsSpan<float>();
        for (int i = 0; i < startTokens.Length; i++)
        {
            table.Slice(startTokens[i] * hidden, hidden).CopyTo(target.Slice((conditionRows + i) * hidden, hidden));
        }

        using Tensor hiddenStates = _transformer.ForwardEmbeds(_backend, embeds, rows, 0, _cache);
        _length = rows;
        return LastLogits(hiddenStates, rows);
    }

    /// <summary>Feeds one generated token and returns the next logits.</summary>
    public float[] Step(int token)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_cache is null)
        {
            throw new InvalidOperationException("Call Begin before Step.");
        }

        if (_length >= _config.MaxPositions)
        {
            throw new HartsyInferenceException($"UniRig sequence reached the {_config.MaxPositions}-position limit.");
        }

        ReadOnlySpan<int> ids = [token];
        using Tensor hiddenStates = _transformer.Forward(_backend, ids, _length, _cache);
        _length++;
        return LastLogits(hiddenStates, 1);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _cache?.Dispose();
        _transformer.Dispose();
        foreach (Tensor tensor in _owned)
        {
            tensor.Dispose();
        }
    }

    private float[] LastLogits(Tensor hiddenStates, int rows)
    {
        using Tensor logits = _transformer.ProjectLogits(_backend, hiddenStates, rows);
        return logits.AsSpan<float>().Slice((rows - 1) * _config.VocabSize, _config.VocabSize).ToArray();
    }

    private Dictionary<string, Tensor> Remap(IReadOnlyDictionary<string, Tensor> weights)
    {
        int h = _config.HiddenSize;
        Dictionary<string, Tensor> map = new(StringComparer.Ordinal)
        {
            ["model.embed_tokens.weight"] = _embedTokens,
            ["model.embed_positions.weight"] = DropPositionOffset(UniRigCheckpoint.Get(weights, $"{Decoder}.embed_positions.weight",
                _config.MaxPositions + PositionOffset, h)),
            ["model.norm.weight"] = UniRigCheckpoint.Get(weights, $"{Decoder}.final_layer_norm.weight", h),
            ["model.norm.bias"] = UniRigCheckpoint.Get(weights, $"{Decoder}.final_layer_norm.bias", h),
            ["lm_head.weight"] = UniRigCheckpoint.Get(weights, "transformer.lm_head.weight", _config.VocabSize, h),
        };
        for (int i = 0; i < _config.NumLayers; i++)
        {
            string src = $"{Decoder}.layers.{i}";
            string dst = $"model.layers.{i}";
            foreach (string projection in new[] { "q_proj", "k_proj", "v_proj" })
            {
                map[$"{dst}.self_attn.{projection}.weight"] = UniRigCheckpoint.Get(weights, $"{src}.self_attn.{projection}.weight", h, h);
                map[$"{dst}.self_attn.{projection}.bias"] = UniRigCheckpoint.Get(weights, $"{src}.self_attn.{projection}.bias", h);
            }

            map[$"{dst}.self_attn.o_proj.weight"] = UniRigCheckpoint.Get(weights, $"{src}.self_attn.out_proj.weight", h, h);
            map[$"{dst}.self_attn.o_proj.bias"] = UniRigCheckpoint.Get(weights, $"{src}.self_attn.out_proj.bias", h);
            map[$"{dst}.input_layernorm.weight"] = UniRigCheckpoint.Get(weights, $"{src}.self_attn_layer_norm.weight", h);
            map[$"{dst}.input_layernorm.bias"] = UniRigCheckpoint.Get(weights, $"{src}.self_attn_layer_norm.bias", h);
            map[$"{dst}.post_attention_layernorm.weight"] = UniRigCheckpoint.Get(weights, $"{src}.final_layer_norm.weight", h);
            map[$"{dst}.post_attention_layernorm.bias"] = UniRigCheckpoint.Get(weights, $"{src}.final_layer_norm.bias", h);
            map[$"{dst}.mlp.up_proj.weight"] = UniRigCheckpoint.Get(weights, $"{src}.fc1.weight", _config.FfnSize, h);
            map[$"{dst}.mlp.up_proj.bias"] = UniRigCheckpoint.Get(weights, $"{src}.fc1.bias", _config.FfnSize);
            map[$"{dst}.mlp.down_proj.weight"] = UniRigCheckpoint.Get(weights, $"{src}.fc2.weight", h, _config.FfnSize);
            map[$"{dst}.mlp.down_proj.bias"] = UniRigCheckpoint.Get(weights, $"{src}.fc2.bias", h);
        }

        return map;
    }

    private Tensor DropPositionOffset(Tensor positions)
    {
        long width = positions.Shape[1];
        Tensor shifted = new(new TensorShape(positions.Shape[0] - PositionOffset, width), DType.F32);
        positions.AsSpan<float>()[(int)(PositionOffset * width)..].CopyTo(shifted.AsSpan<float>());
        _owned.Add(shifted);
        return shifted;
    }
}

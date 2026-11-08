using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Packed-projection multi-head attention over batch-first tensors (equal to upstream sequence-first at batch 1).</summary>
public sealed class Sam3MultiheadAttention
{
    private readonly int _embedDim;
    private readonly int _heads;
    private readonly int _headDim;
    private Tensor? _inWeight;
    private Tensor? _inBias;
    private Tensor? _outWeight;
    private Tensor? _outBias;

    /// <summary>Creates an attention block of width <paramref name="embedDim"/> split across <paramref name="heads"/> heads.</summary>
    public Sam3MultiheadAttention(int embedDim, int heads)
    {
        if (embedDim <= 0 || heads <= 0 || embedDim % heads != 0)
            throw new ArgumentException("Attention width must be a positive multiple of the head count.");
        _embedDim = embedDim;
        _heads = heads;
        _headDim = embedDim / heads;
    }

    /// <summary>Loads <c>in_proj_weight/bias</c> (query, key and value rows stacked) and <c>out_proj.weight/bias</c> under <paramref name="prefix"/>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(prefix);
        int e = _embedDim;
        _inWeight = Sam3WeightAccess.Require(weights, prefix + "in_proj_weight", 3 * e, e);
        _inBias = Sam3WeightAccess.Require(weights, prefix + "in_proj_bias", 3 * e);
        _outWeight = Sam3WeightAccess.Require(weights, prefix + "out_proj.weight", e, e);
        _outBias = Sam3WeightAccess.Require(weights, prefix + "out_proj.bias", e);
    }

    /// <summary>Attends <paramref name="query"/> over <paramref name="key"/> and <paramref name="value"/> (each <c>[1, S, E]</c>)
    /// with an optional additive bias <c>[1, heads, Sq, Skv]</c>; returns a new <c>[1, Sq, E]</c> tensor the caller owns.</summary>
    public Tensor Forward(IBackend backend, Tensor query, Tensor key, Tensor value, Tensor? additiveMask)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(value);
        Tensor inWeight = _inWeight ?? throw new InvalidOperationException("SAM 3 attention weights are not loaded.");
        Tensor inBias = _inBias ?? throw new InvalidOperationException("SAM 3 attention weights are not loaded.");
        Tensor outWeight = _outWeight ?? throw new InvalidOperationException("SAM 3 attention weights are not loaded.");
        Tensor outBias = _outBias ?? throw new InvalidOperationException("SAM 3 attention weights are not loaded.");

        Sam3Ops.RequireSequence(query, "attention query", _embedDim);
        Sam3Ops.RequireSequence(key, "attention key", _embedDim);
        Sam3Ops.RequireSequence(value, "attention value", _embedDim);
        if (key.Shape[1] != value.Shape[1])
            throw new ArgumentException($"Attention key rows {key.Shape[1]} and value rows {value.Shape[1]} differ.", nameof(value));
        long sq = query.Shape[1];
        long skv = key.Shape[1];
        if (additiveMask is not null)
            Sam3Ops.RequireShape(additiveMask, "attention bias", 1, _heads, sq, skv);

        Tensor q = Project(backend, query, inWeight, inBias, 0);
        Tensor k = Project(backend, key, inWeight, inBias, 1);
        Tensor v = Project(backend, value, inWeight, inBias, 2);
        Tensor merged = RtDetrAttention.MultiHead(backend, q, k, v, _heads, _headDim, additiveMask);
        q.Dispose();
        k.Dispose();
        v.Dispose();

        Tensor output = Sam3Ops.NewF32(1, sq, _embedDim);
        backend.Linear(output, merged, outWeight, outBias);
        merged.Dispose();
        return output;
    }

    /// <summary>Builds the additive bias <c>[1, heads, queryRows, keys]</c> that removes padded keys (true = padding) for every query.</summary>
    public static Tensor KeyPaddingBias(bool[] keyPadding, int queryRows, int heads)
    {
        ArgumentNullException.ThrowIfNull(keyPadding);
        if (keyPadding.Length == 0 || queryRows <= 0 || heads <= 0)
            throw new ArgumentException("Key padding bias needs at least one key, query row and head.");

        int keys = keyPadding.Length;
        Tensor bias = Sam3Ops.NewF32(1, heads, queryRows, keys);
        Span<float> values = bias.AsSpan<float>();
        int rows = heads * queryRows;
        for (int row = 0; row < rows; row++)
        {
            int offset = row * keys;
            for (int j = 0; j < keys; j++)
                values[offset + j] = keyPadding[j] ? float.NegativeInfinity : 0f;
        }
        return bias;
    }

    private Tensor Project(IBackend backend, Tensor input, Tensor weight, Tensor bias, int slot)
    {
        Tensor output = Sam3Ops.NewF32(1, input.Shape[1], _embedDim);
        backend.LinearWeightRows(output, input, weight, bias, slot * _embedDim, _embedDim);
        return output;
    }
}

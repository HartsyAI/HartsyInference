using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Pre-norm self-attention, cross-attention and ReLU feed-forward layer, batch-first.</summary>
public sealed class Sam3FusionEncoderLayer
{
    private readonly int _dModel;
    private readonly int _heads;
    private readonly int _ffnWidth;
    private readonly float _eps;
    private readonly Sam3MultiheadAttention _selfAttention;
    private readonly Sam3MultiheadAttention _crossAttention;
    private readonly Sam3Mlp _feedForward = new();
    private Tensor[] _normWeights = [];
    private Tensor[] _normBiases = [];

    /// <summary>Creates a layer of width <paramref name="dModel"/> with <paramref name="heads"/> attention heads and a
    /// feed-forward hidden width of <paramref name="ffnWidth"/>.</summary>
    public Sam3FusionEncoderLayer(int dModel, int heads, int ffnWidth, float layerNormEps)
    {
        if (dModel <= 0 || ffnWidth <= 0 || layerNormEps <= 0f)
            throw new ArgumentException("Encoder layer width, feed-forward width and epsilon must be positive.");
        _dModel = dModel;
        _heads = heads;
        _ffnWidth = ffnWidth;
        _eps = layerNormEps;
        _selfAttention = new Sam3MultiheadAttention(dModel, heads);
        _crossAttention = new Sam3MultiheadAttention(dModel, heads);
    }

    /// <summary>Loads the layer under <paramref name="prefix"/> (for example <c>layers.0.</c>): <c>self_attn.*</c>,
    /// <c>cross_attn_image.*</c>, <c>linear1/linear2</c> and <c>norm1..norm3</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(prefix);
        _selfAttention.LoadWeights(weights, prefix + "self_attn.");
        _crossAttention.LoadWeights(weights, prefix + "cross_attn_image.");

        Tensor linear1Weight = Sam3WeightAccess.Require(weights, prefix + "linear1.weight", _ffnWidth, _dModel);
        Tensor linear1Bias = Sam3WeightAccess.Require(weights, prefix + "linear1.bias", _ffnWidth);
        Tensor linear2Weight = Sam3WeightAccess.Require(weights, prefix + "linear2.weight", _dModel, _ffnWidth);
        Tensor linear2Bias = Sam3WeightAccess.Require(weights, prefix + "linear2.bias", _dModel);
        _feedForward.Bind([linear1Weight, linear2Weight], [linear1Bias, linear2Bias]);

        _normWeights =
        [
            Sam3WeightAccess.Require(weights, prefix + "norm1.weight", _dModel),
            Sam3WeightAccess.Require(weights, prefix + "norm2.weight", _dModel),
            Sam3WeightAccess.Require(weights, prefix + "norm3.weight", _dModel),
        ];
        _normBiases =
        [
            Sam3WeightAccess.Require(weights, prefix + "norm1.bias", _dModel),
            Sam3WeightAccess.Require(weights, prefix + "norm2.bias", _dModel),
            Sam3WeightAccess.Require(weights, prefix + "norm3.bias", _dModel),
        ];
    }

    /// <summary>Runs the layer on <paramref name="target"/> <c>[1, S, E]</c>. <paramref name="queryPos"/> is added to the
    /// self-attention query and key; <paramref name="memoryPos"/> to the cross-attention keys; <paramref name="memoryPadding"/>
    /// (true = padding) masks cross-attention keys. Returns a new tensor the caller owns; the inputs are not disposed.</summary>
    public Tensor Forward(IBackend backend, Tensor target, Tensor? queryPos, Tensor memory, Tensor? memoryPos, bool[]? memoryPadding)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(target);
        ArgumentNullException.ThrowIfNull(memory);
        if (_normWeights.Length == 0)
            throw new InvalidOperationException("SAM 3 encoder layer weights are not loaded.");
        Sam3Ops.RequireSequence(target, "encoder target", _dModel);
        Sam3Ops.RequireSequence(memory, "encoder memory", _dModel);
        if (queryPos is not null)
            Sam3Ops.RequireSequence(queryPos, "encoder query position", _dModel);
        if (memoryPos is not null)
            Sam3Ops.RequireSequence(memoryPos, "encoder memory position", _dModel);
        if (memoryPadding is not null && memoryPadding.Length != memory.Shape[1])
            throw new ArgumentException($"Memory padding has {memoryPadding.Length} entries for {memory.Shape[1]} memory rows.", nameof(memoryPadding));

        int rows = (int)target.Shape[1];

        Tensor normSelf = Sam3Ops.LayerNorm(backend, target, _normWeights[0], _normBiases[0], _eps);
        Tensor selfQuery = queryPos is null ? normSelf : Sam3Ops.Add(backend, normSelf, queryPos);
        Tensor selfOut = _selfAttention.Forward(backend, selfQuery, selfQuery, normSelf, null);
        if (queryPos is not null)
            selfQuery.Dispose();
        normSelf.Dispose();
        Tensor afterSelf = Sam3Ops.Add(backend, target, selfOut);
        selfOut.Dispose();

        Tensor normCross = Sam3Ops.LayerNorm(backend, afterSelf, _normWeights[1], _normBiases[1], _eps);
        Tensor crossKeys = memoryPos is null ? memory : Sam3Ops.Add(backend, memory, memoryPos);
        Tensor? padBias = memoryPadding is null ? null : Sam3MultiheadAttention.KeyPaddingBias(memoryPadding, rows, _heads);
        Tensor crossOut = _crossAttention.Forward(backend, normCross, crossKeys, memory, padBias);
        padBias?.Dispose();
        if (memoryPos is not null)
            crossKeys.Dispose();
        normCross.Dispose();
        Tensor afterCross = Sam3Ops.Add(backend, afterSelf, crossOut);
        afterSelf.Dispose();
        crossOut.Dispose();

        Tensor normFfn = Sam3Ops.LayerNorm(backend, afterCross, _normWeights[2], _normBiases[2], _eps);
        Tensor ffn = _feedForward.Forward(backend, normFfn);
        normFfn.Dispose();
        Tensor result = Sam3Ops.Add(backend, afterCross, ffn);
        afterCross.Dispose();
        ffn.Dispose();
        return result;
    }
}

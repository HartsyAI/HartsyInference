using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>One Fibo 2.0 Perceiver block: cross-attention from the learned queries to a text bundle, self-attention over
/// the queries, then SwiGLU. Residuals are plain (no gates), there is no adaLN and no RoPE.</summary>
public sealed class Fibo2PerceiverLayer : IDisposable
{
    private readonly int _dim;
    private readonly int _ffnDim;
    private readonly float _eps;
    private readonly AttentionLayerConfig _attentionConfig;
    private readonly SwiGluFfn _ffn;
    private readonly List<Tensor> _ownedSplits = [];

    private Tensor? _crossNorm, _selfNorm, _ffnNorm;
    private AttentionLayer? _crossAttention, _selfAttention;
    private int _disposed;

    /// <summary>Creates a Perceiver block of the config's width and heads.</summary>
    public Fibo2PerceiverLayer(Fibo2Config config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _dim = config.Dim;
        _ffnDim = config.FfnDim;
        _eps = config.NormEps;
        _attentionConfig = new AttentionLayerConfig
        {
            Heads = config.NumHeads,
            KvHeads = config.NumHeads,
            HeadDim = config.HeadDim,
            QkNorm = AttentionQkNorm.PerHead,
        };
        _ffn = new SwiGluFfn(config.Dim, config.FfnDim);
    }

    /// <summary>Loads the block under <paramref name="prefix"/> (for example <c>perceiver.layers.0</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _crossNorm = TensorCasts.LoadF32(weights, $"{prefix}.cross_attn_norm.weight");
        _selfNorm = TensorCasts.LoadF32(weights, $"{prefix}.self_attn_norm.weight");
        _ffnNorm = TensorCasts.LoadF32(weights, $"{prefix}.ffn_norm.weight");
        _crossAttention = Fibo2Weights.LoadCrossAttention(weights, $"{prefix}.cross_attention.", _attentionConfig, _dim, _eps,
            _ownedSplits);
        _selfAttention = Fibo2Weights.LoadCrossAttention(weights, $"{prefix}.self_attention.", _attentionConfig, _dim, _eps,
            _ownedSplits);
        Fibo2Weights.LoadSwiGlu(_ffn, weights, $"{prefix}.ffn.", _ffnDim, _ownedSplits);
    }

    /// <summary>Runs the block on the queries <c>[1, G, dim]</c> against the text bundle <c>[1, L, capFeatDim]</c>. The caller
    /// keeps both inputs; the result is new.</summary>
    public Tensor Forward(IBackend backend, Tensor queries, Tensor bundle)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (queries.Shape.Rank != 3 || queries.Shape[0] != 1 || queries.Shape[2] != _dim)
            throw new ArgumentException($"Perceiver queries must be [1, G, {_dim}], got {queries.Shape}.", nameof(queries));
        int count = (int)queries.Shape[1];
        TensorShape shape = new(1, count, _dim);

        Tensor crossIn = new(shape, DType.F32);
        backend.RmsNorm(crossIn, queries, _crossNorm!, _eps);
        AttentionCallContext crossContext = new() { ReleaseInputs = false };
        Tensor crossOut = _crossAttention!.Forward(backend, crossIn, bundle, in crossContext);
        crossIn.Dispose();
        Tensor afterCross = new(shape, DType.F32);
        backend.Add(afterCross, queries, crossOut);
        crossOut.Dispose();

        Tensor selfIn = new(shape, DType.F32);
        backend.RmsNorm(selfIn, afterCross, _selfNorm!, _eps);
        AttentionCallContext selfContext = new() { ReleaseInputs = false };
        Tensor selfOut = _selfAttention!.Forward(backend, selfIn, selfIn, in selfContext);
        selfIn.Dispose();
        Tensor afterSelf = new(shape, DType.F32);
        backend.Add(afterSelf, afterCross, selfOut);
        afterCross.Dispose();
        selfOut.Dispose();

        Tensor ffnIn = new(shape, DType.F32);
        backend.RmsNorm(ffnIn, afterSelf, _ffnNorm!, _eps);
        Tensor ffnOut = _ffn.Forward(backend, ffnIn, 1, count);
        ffnIn.Dispose();
        Tensor result = new(shape, DType.F32);
        backend.Add(result, afterSelf, ffnOut);
        afterSelf.Dispose();
        ffnOut.Dispose();
        return result;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (Tensor split in _ownedSplits)
            split.Dispose();
        _ownedSplits.Clear();
    }
}

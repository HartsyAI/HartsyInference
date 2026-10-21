using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam2;

/// <summary>SAM's mask-decoder attention: separate q/k/v projections into an internal dim (optionally
/// downsampled below the embedding dim), multi-head SDPA, then an output projection back to the embedding
/// dim. The internal dim is read from the loaded <c>q_proj</c> weight so both the ratio-1 self-attention and
/// the ratio-2 cross-attentions of the real checkpoint load without configuration.</summary>
public sealed class DecoderAttention
{
    private readonly int _heads;
    private int _internalDim;
    private int _headDim;

    private Tensor? _qW, _qB;
    private Tensor? _kW, _kB;
    private Tensor? _vW, _vB;
    private Tensor? _outW, _outB;
    private AttentionLayer? _attn;

    /// <summary>Creates the attention with a fixed head count; per-projection dims come from the weights.</summary>
    public DecoderAttention(int heads)
    {
        if (heads <= 0)
            throw new ArgumentOutOfRangeException(nameof(heads), heads, "Head count must be positive.");
        _heads = heads;
    }

    /// <summary>Loads q/k/v/out projections under <paramref name="prefix"/> (e.g. <c>...self_attn.</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _qW = F32(weights, $"{prefix}q_proj.weight");
        _qB = F32(weights, $"{prefix}q_proj.bias");
        _kW = F32(weights, $"{prefix}k_proj.weight");
        _kB = F32(weights, $"{prefix}k_proj.bias");
        _vW = F32(weights, $"{prefix}v_proj.weight");
        _vB = F32(weights, $"{prefix}v_proj.bias");
        _outW = F32(weights, $"{prefix}out_proj.weight");
        _outB = F32(weights, $"{prefix}out_proj.bias");

        _internalDim = (int)_qW.Shape[0];
        if (_internalDim % _heads != 0)
            throw new ArgumentException($"Attention internal dim {_internalDim} not divisible by heads {_heads}.");
        _headDim = _internalDim / _heads;
        _attn = new AttentionLayer(new AttentionLayerConfig { Heads = _heads, KvHeads = _heads, HeadDim = _headDim, ComputeDType = DType.F32 });
    }

    /// <summary>Yields every loaded weight tensor.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _qW, _qB, _kW, _kB, _vW, _vB, _outW, _outB })
        {
            if (t is not null) yield return t;
        }
    }

    /// <summary>Attends queries <c>[1,Nq,C]</c> over keys/values <c>[1,Nk,C]</c>; returns <c>[1,Nq,C]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor query, Tensor key, Tensor value)
    {
        if (_qW is null)
            throw new InvalidOperationException("DecoderAttention weights not loaded.");

        int b = (int)query.Shape[0];
        int nq = (int)query.Shape[1];
        int nk = (int)key.Shape[1];
        int embedDim = (int)query.Shape[2];

        Tensor qp = new Tensor(new TensorShape(b, nq, _internalDim), DType.F32);
        backend.Linear(qp, query, _qW!, _qB);
        Tensor kp = new Tensor(new TensorShape(b, nk, _internalDim), DType.F32);
        backend.Linear(kp, key, _kW!, _kB);
        Tensor vp = new Tensor(new TensorShape(b, nk, _internalDim), DType.F32);
        backend.Linear(vp, value, _vW!, _vB);

        Tensor merged = _attn!.AttendFlat(backend, qp, kp, vp, new TensorShape(b, nq, _internalDim), default);

        Tensor outT = new Tensor(new TensorShape(b, nq, embedDim), DType.F32);
        backend.Linear(outT, merged, _outW!, _outB);
        merged.Dispose();
        return outT;
    }

    private static Tensor F32(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        if (!weights.TryGetValue(key, out Tensor? t))
            throw new KeyNotFoundException($"DecoderAttention: missing weight '{key}'.");
        return t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
    }
}

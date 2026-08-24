using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.GroundingDino;

/// <summary>Standard batch-first multi-head attention with separate query/key/value/out_proj linears (HF
/// <c>GroundingDinoMultiheadAttention</c>), used by the text-enhancer self-attention and the decoder self- and
/// text-cross-attention. Scale is <c>1/sqrt(head_dim)</c>; an optional additive mask <c>[1, heads, Sq, Skv]</c> is
/// added to the pre-softmax scores.</summary>
public sealed unsafe class GroundingDinoMultiheadAttention(int hidden, int heads) : IDisposable
{
    private readonly int _hidden = hidden;
    private readonly int _heads = heads;
    private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB;
    private int _disposed;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, string outName = "out_proj")
    {
        _qW = w[$"{prefix}.query.weight"]; _qB = w[$"{prefix}.query.bias"];
        _kW = w[$"{prefix}.key.weight"]; _kB = w[$"{prefix}.key.bias"];
        _vW = w[$"{prefix}.value.weight"]; _vB = w[$"{prefix}.value.bias"];
        _oW = w[$"{prefix}.{outName}.weight"]; _oB = w[$"{prefix}.{outName}.bias"];
    }

    public Tensor Forward(IBackend backend, Tensor queries, Tensor keys, Tensor values, Tensor? additiveMask)
    {
        int sq = (int)queries.Shape[1], skv = (int)keys.Shape[1];
        int h = _hidden, nh = _heads, hd = h / nh;

        Tensor q = Lin(backend, queries, _qW!, _qB, sq, h);
        Tensor k = Lin(backend, keys, _kW!, _kB, skv, h);
        Tensor v = Lin(backend, values, _vW!, _vB, skv, h);

        Tensor ctx = RtDetrAttention.MultiHead(backend, q, k, v, nh, hd, additiveMask);
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor o = Lin(backend, ctx, _oW!, _oB, sq, h);
        ctx.Dispose();
        return o;
    }

    private static Tensor Lin(IBackend backend, Tensor input, Tensor w, Tensor? b, int rows, int outDim)
    {
        Tensor o = new(new TensorShape(1, rows, outDim), DType.F32);
        backend.Linear(o, input, w, b);
        return o;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}

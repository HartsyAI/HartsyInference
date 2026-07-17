using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.GroundingDino;

/// <summary>Standard batch-first multi-head attention with separate query/key/value/out_proj linears (HF
/// <c>GroundingDinoMultiheadAttention</c>), used by the text-enhancer self-attention and the decoder self- and
/// text-cross-attention. Scale is <c>1/sqrt(head_dim)</c>; an optional additive mask <c>[1, heads, Sq, Skv]</c> is
/// added to the pre-softmax scores.</summary>
public sealed unsafe class GroundingDinoMultiheadAttention : IDisposable
{
    private readonly int _hidden;
    private readonly int _heads;
    private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB;
    private int _disposed;

    public GroundingDinoMultiheadAttention(int hidden, int heads)
    {
        _hidden = hidden;
        _heads = heads;
    }

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

        Tensor qH = new(new TensorShape(1, nh, sq, hd), DType.F32);
        Tensor kH = new(new TensorShape(1, nh, skv, hd), DType.F32);
        Tensor vH = new(new TensorShape(1, nh, skv, hd), DType.F32);
        ToHeads(qH, q, sq, nh, hd); ToHeads(kH, k, skv, nh, hd); ToHeads(vH, v, skv, nh, hd);
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor attn = new(new TensorShape(1, nh, sq, hd), DType.F32);
        backend.ScaledDotProductAttention(attn, qH, kH, vH, additiveMask, 1f / MathF.Sqrt(hd));
        qH.Dispose(); kH.Dispose(); vH.Dispose();

        Tensor ctx = new(new TensorShape(1, sq, h), DType.F32);
        FromHeads(ctx, attn, sq, nh, hd);
        attn.Dispose();
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

    private static void ToHeads(Tensor dst, Tensor src, int s, int nh, int hd)
    {
        float* sp = (float*)src.DataPointer, dp = (float*)dst.DataPointer;
        for (int t = 0; t < s; t++)
            for (int hh = 0; hh < nh; hh++)
                for (int c = 0; c < hd; c++)
                    dp[((long)hh * s + t) * hd + c] = sp[(long)t * nh * hd + hh * hd + c];
    }

    private static void FromHeads(Tensor dst, Tensor src, int s, int nh, int hd)
    {
        float* sp = (float*)src.DataPointer, dp = (float*)dst.DataPointer;
        for (int t = 0; t < s; t++)
            for (int hh = 0; hh < nh; hh++)
                for (int c = 0; c < hd; c++)
                    dp[(long)t * nh * hd + hh * hd + c] = sp[((long)hh * s + t) * hd + c];
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}

using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>The attention of Idol's decoders (upstream <c>MultiHeadAttention</c>): self-attention over one packed <c>to_qkv</c> projection,
/// or cross-attention from <c>to_q</c> of the queries over <c>to_kv</c> of a context, with the per-head RMS norm on q and k and no rotation,
/// then <c>to_out</c>. Borrows its weights; owns the unit scale of the head norm.</summary>
internal sealed class IdolDecoderAttention : IDisposable
{
    private readonly int _width, _heads, _headDim;
    private readonly AttentionLayer _attention;
    private readonly bool _cross;
    private readonly HeadRmsNorm _headNorm;
    private readonly Tensor _qW, _qB, _kvW, _kvB, _outW, _outB, _qGamma, _kGamma;

    internal IdolDecoderAttention(IReadOnlyDictionary<string, Tensor> w, string prefix, bool cross, int width, int heads)
    {
        _width = width;
        _heads = heads;
        _headDim = width / heads;
        _attention = IdolAttention.Layer(_heads, _headDim);
        _cross = cross;
        _headNorm = new HeadRmsNorm(_headDim);
        if (cross)
        {
            _qW = w[$"{prefix}.to_q.weight"]; _qB = w[$"{prefix}.to_q.bias"];
            _kvW = w[$"{prefix}.to_kv.weight"]; _kvB = w[$"{prefix}.to_kv.bias"];
        }
        else
        {
            _qW = w[$"{prefix}.to_qkv.weight"]; _qB = w[$"{prefix}.to_qkv.bias"];
            _kvW = _qW; _kvB = _qB;
        }

        _outW = w[$"{prefix}.to_out.weight"]; _outB = w[$"{prefix}.to_out.bias"];
        _qGamma = w[$"{prefix}.q_rms_norm.gamma"].Reshape(new TensorShape(width));
        _kGamma = w[$"{prefix}.k_rms_norm.gamma"].Reshape(new TensorShape(width));
    }

    /// <summary><paramref name="x"/> <c>[1, L, C]</c> attends to itself (self) or to <paramref name="context"/> <c>[1, Lc, Cctx]</c> (cross); returns <c>[1, L, C]</c>.</summary>
    internal Tensor Forward(IBackend backend, Tensor x, Tensor? context)
    {
        int n = (int)x.Shape[1];
        Tensor q, k, v;
        int keys;
        if (_cross)
        {
            if (context is null)
                throw new ArgumentNullException(nameof(context));
            keys = (int)context.Shape[1];
            using Tensor qFlat = Project(backend, x, _qW, _qB, _width);
            using Tensor kv = Project(backend, context, _kvW, _kvB, 2L * _width);
            q = IdolAttention.Slice(backend, qFlat, 0, n, _heads, _headDim);
            k = IdolAttention.Slice(backend, kv, 0, keys, _heads, _headDim);
            v = IdolAttention.Slice(backend, kv, _width, keys, _heads, _headDim);
        }
        else
        {
            keys = n;
            using Tensor qkv = Project(backend, x, _qW, _qB, 3L * _width);
            q = IdolAttention.Slice(backend, qkv, 0, n, _heads, _headDim);
            k = IdolAttention.Slice(backend, qkv, _width, n, _heads, _headDim);
            v = IdolAttention.Slice(backend, qkv, 2 * _width, n, _heads, _headDim);
        }

        Tensor qn = _headNorm.Apply(backend, q, _qGamma, new TensorShape(1, n, _width));
        Tensor kn = _headNorm.Apply(backend, k, _kGamma, new TensorShape(1, keys, _width));
        using Tensor ctx = IdolAttention.Attend(backend, _attention, qn, kn, v, n, keys, _heads, _headDim);
        return Project(backend, ctx, _outW, _outB, _width);
    }

    private static Tensor Project(IBackend backend, Tensor x, Tensor weight, Tensor bias, long outDim)
    {
        Tensor o = new(new TensorShape(1, x.Shape[1], outDim), DType.F32);
        backend.Linear(o, x, weight, bias);
        return o;
    }

    public void Dispose() => _headNorm.Dispose();
}

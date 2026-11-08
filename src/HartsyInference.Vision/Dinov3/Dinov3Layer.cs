using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Dinov3;

/// <summary>One DINOv3 transformer block: pre-norm RoPE self-attention and a GELU MLP, each scaled by a learned
/// per-channel LayerScale before its residual add. The key projection has no bias (the reference masks the k-bias).</summary>
internal sealed unsafe class Dinov3Layer
{
    private readonly int _hidden, _numHeads, _headDim, _inter;
    private readonly float _eps;
    private Tensor? _n1W, _n1B, _ls1, _n2W, _n2B, _fc1W, _fc1B, _fc2W, _fc2B, _ls2;
    private AttentionLayer? _attn;
    private Tensor? _qW, _qB, _kW, _vW, _vB, _oW, _oB;

    public Dinov3Layer(int hidden, int numHeads, int inter, float eps)
    {
        _hidden = hidden; _numHeads = numHeads; _headDim = hidden / numHeads; _inter = inter; _eps = eps;
    }

    /// <summary>Loads one block from the HF DINOv3 layout under <paramref name="prefix"/> (e.g. <c>layer.3</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        // Every block tensor is checked against its exact shape before any kernel reads it: a short tensor would otherwise be
        // read through an unchecked pointer in LayerNorm, Linear or the affine broadcast.
        int h = _hidden, i = _inter;
        _n1W = Dinov3VisionEncoder.F32(Dinov3VisionEncoder.Require(w, $"{prefix}.norm1.weight", h));
        _n1B = Dinov3VisionEncoder.F32(Dinov3VisionEncoder.Require(w, $"{prefix}.norm1.bias", h));
        _qW = Dinov3VisionEncoder.F32(Dinov3VisionEncoder.Require(w, $"{prefix}.attention.q_proj.weight", h, h));
        _qB = Dinov3VisionEncoder.F32(Dinov3VisionEncoder.Require(w, $"{prefix}.attention.q_proj.bias", h));
        _kW = Dinov3VisionEncoder.F32(Dinov3VisionEncoder.Require(w, $"{prefix}.attention.k_proj.weight", h, h));
        _vW = Dinov3VisionEncoder.F32(Dinov3VisionEncoder.Require(w, $"{prefix}.attention.v_proj.weight", h, h));
        _vB = Dinov3VisionEncoder.F32(Dinov3VisionEncoder.Require(w, $"{prefix}.attention.v_proj.bias", h));
        _oW = Dinov3VisionEncoder.F32(Dinov3VisionEncoder.Require(w, $"{prefix}.attention.o_proj.weight", h, h));
        _oB = Dinov3VisionEncoder.F32(Dinov3VisionEncoder.Require(w, $"{prefix}.attention.o_proj.bias", h));
        _ls1 = Dinov3VisionEncoder.F32(Dinov3VisionEncoder.Require(w, $"{prefix}.layer_scale1.lambda1", h));
        _n2W = Dinov3VisionEncoder.F32(Dinov3VisionEncoder.Require(w, $"{prefix}.norm2.weight", h));
        _n2B = Dinov3VisionEncoder.F32(Dinov3VisionEncoder.Require(w, $"{prefix}.norm2.bias", h));
        _fc1W = Dinov3VisionEncoder.F32(Dinov3VisionEncoder.Require(w, $"{prefix}.mlp.up_proj.weight", i, h));
        _fc1B = Dinov3VisionEncoder.F32(Dinov3VisionEncoder.Require(w, $"{prefix}.mlp.up_proj.bias", i));
        _fc2W = Dinov3VisionEncoder.F32(Dinov3VisionEncoder.Require(w, $"{prefix}.mlp.down_proj.weight", h, i));
        _fc2B = Dinov3VisionEncoder.F32(Dinov3VisionEncoder.Require(w, $"{prefix}.mlp.down_proj.bias", h));
        _ls2 = Dinov3VisionEncoder.F32(Dinov3VisionEncoder.Require(w, $"{prefix}.layer_scale2.lambda1", h));
        _attn = new AttentionLayer(
            new AttentionLayerConfig { Heads = _numHeads, KvHeads = _numHeads, HeadDim = _headDim, ComputeDType = DType.F32 },
            new AttentionLayerWeights
            {
                Query = _qW, QueryBias = _qB, Key = _kW, KeyBias = null, Value = _vW, ValueBias = _vB, Output = _oW, OutputBias = _oB,
            });
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_n1W, _n1B, _qW, _kW, _vW, _vB, _qB, _oW, _oB, _ls1, _n2W, _n2B, _fc1W, _fc1B, _fc2W, _fc2B, _ls2];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    /// <summary><paramref name="x"/> <c>[B, S, hidden]</c> through the block; <paramref name="cos"/> and <paramref name="sin"/>
    /// are the <c>[B, S, headDim]</c> RoPE tables, borrowed for the call.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor cos, Tensor sin)
    {
        int batch = (int)x.Shape[0], seq = (int)x.Shape[1];
        TensorShape shape = new(batch, seq, _hidden);
        QkRotation rotation = new(QkRotationKind.RotateHalf, cos, sin, seq);

        Tensor n1 = new(shape, DType.F32); backend.LayerNorm(n1, x, _n1W!, _n1B!, _eps);
        Tensor attn = _attn!.Forward(backend, n1, n1, new AttentionCallContext { QueryRotation = rotation, KeyRotation = rotation });
        n1.Dispose();
        Tensor attnScaled = LayerScale(backend, attn, _ls1!, batch, seq);
        Tensor r1 = new(shape, DType.F32); backend.Add(r1, x, attnScaled); attnScaled.Dispose();

        Tensor n2 = new(shape, DType.F32); backend.LayerNorm(n2, r1, _n2W!, _n2B!, _eps);
        Tensor mlp = LayerScale(backend, Mlp(backend, n2, batch, seq), _ls2!, batch, seq); n2.Dispose();
        Tensor r2 = new(shape, DType.F32); backend.Add(r2, r1, mlp); r1.Dispose(); mlp.Dispose();
        return r2;
    }

    // t[b,s,c] *= gamma[c]. The affine kernel broadcasts gamma over rows only for one batch item.
    private Tensor LayerScale(IBackend backend, Tensor t, Tensor gamma, int batch, int seq)
    {
        if (batch == 1)
        {
            Tensor o = new(t.Shape, DType.F32);
            backend.AffineBroadcastLastDim(o, t, gamma, null);
            t.Dispose();
            return o;
        }
        float* tp = (float*)t.DataPointer; float* g = (float*)gamma.DataPointer;
        for (long i = 0; i < (long)batch * seq; i++)
            for (int c = 0; c < _hidden; c++) tp[i * _hidden + c] *= g[c];
        return t;
    }

    // Plain MLP with the exact-erf GELU that nn.GELU() uses in the reference.
    private Tensor Mlp(IBackend backend, Tensor input, int batch, int seq)
    {
        Tensor f1 = new(new TensorShape(batch, seq, _inter), DType.F32); backend.Linear(f1, input, _fc1W!, _fc1B!);
        Tensor act = new(f1.Shape, DType.F32); backend.GeluErf(act, f1); f1.Dispose();
        Tensor f2 = new(new TensorShape(batch, seq, _hidden), DType.F32); backend.Linear(f2, act, _fc2W!, _fc2B!); act.Dispose();
        return f2;
    }
}

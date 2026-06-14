using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Wan2.2 VAE mid-block attention (<c>AttentionBlock</c> in <c>vae2_2.py</c>): single-head self-attention applied independently per frame over the H·W spatial tokens. <c>RMS_norm → 1×1 Conv (qkv) → SDPA → 1×1 Conv (proj) → +residual</c>. Reuses <see cref="WanRmsNorm"/> + the backend <see cref="IBackend.Conv2D"/>/<see cref="IBackend.ScaledDotProductAttention"/>.</summary>
public sealed unsafe class Wan22AttentionBlock
{
    private readonly int _dim;
    private readonly WanRmsNorm _norm;
    private Tensor? _qkvW, _qkvB, _projW, _projB;

    public Wan22AttentionBlock(int dim)
    {
        _dim = dim;
        _norm = new WanRmsNorm(dim);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _norm.LoadWeights(weights[$"{prefix}.norm.gamma"]);
        _qkvW = weights[$"{prefix}.to_qkv.weight"];
        weights.TryGetValue($"{prefix}.to_qkv.bias", out _qkvB);
        _projW = weights[$"{prefix}.proj.weight"];
        weights.TryGetValue($"{prefix}.proj.bias", out _projB);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _norm.EnumerateWeights()) yield return t;
        if (_qkvW is not null) yield return _qkvW;
        if (_qkvB is not null) yield return _qkvB;
        if (_projW is not null) yield return _projW;
        if (_projB is not null) yield return _projB;
    }

    /// <summary>Forward over <c>[B, C, T, H, W]</c> → same shape (residual added).</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        int bt = b * t, hw = h * w;

        Tensor normed = _norm.Forward(x);              // [B,C,T,H,W], channel RMS per position
        Tensor frames = Vae3dLayout.ToFrames(normed, b, c, t, h, w);  // [BT,C,H,W]
        normed.Dispose();

        Tensor qkv = new Tensor(new TensorShape(bt, 3 * c, h, w), DType.F32);
        backend.Conv2D(qkv, frames, _qkvW!, _qkvB, 1, 1, 0, 0);
        frames.Dispose();

        // q,k,v: [BT, 1, HW, C]
        Tensor q = new Tensor(new TensorShape(bt, 1, hw, c), DType.F32);
        Tensor k = new Tensor(new TensorShape(bt, 1, hw, c), DType.F32);
        Tensor v = new Tensor(new TensorShape(bt, 1, hw, c), DType.F32);
        SplitQkv(qkv, q, k, v, bt, c, h, w);
        qkv.Dispose();

        float scale = 1.0f / MathF.Sqrt(c);
        Tensor attn = new Tensor(new TensorShape(bt, 1, hw, c), DType.F32);
        backend.ScaledDotProductAttention(attn, q, k, v, null, scale);
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor attn4d = new Tensor(new TensorShape(bt, c, h, w), DType.F32);
        TokensToFrame(attn, attn4d, bt, c, h, w);
        attn.Dispose();

        Tensor proj = new Tensor(new TensorShape(bt, c, h, w), DType.F32);
        backend.Conv2D(proj, attn4d, _projW!, _projB, 1, 1, 0, 0);
        attn4d.Dispose();

        Tensor proj5d = Vae3dLayout.FromFrames(proj, b, c, t, h, w);
        proj.Dispose();
        Tensor outT = new Tensor(x.Shape, DType.F32);
        backend.Add(outT, x, proj5d);
        proj5d.Dispose();
        return outT;
    }

    private static void SplitQkv(Tensor qkv, Tensor q, Tensor k, Tensor v, int bt, int c, int h, int w)
    {
        float* src = (float*)qkv.DataPointer;     // [bt, 3c, h, w]
        float* qp = (float*)q.DataPointer;        // [bt, 1, hw, c]
        float* kp = (float*)k.DataPointer;
        float* vp = (float*)v.DataPointer;
        long frame = (long)h * w;
        int hw = h * w;
        for (int i = 0; i < bt; i++)
            for (int ci = 0; ci < c; ci++)
                for (int token = 0; token < hw; token++)
                {
                    long srcBase = ((long)i * 3 * c) * frame + token;
                    long dstOff = ((long)i * hw + token) * c + ci;
                    qp[dstOff] = src[srcBase + (long)ci * frame];
                    kp[dstOff] = src[srcBase + (long)(c + ci) * frame];
                    vp[dstOff] = src[srcBase + (long)(2 * c + ci) * frame];
                }
    }

    private static void TokensToFrame(Tensor attn, Tensor outT, int bt, int c, int h, int w)
    {
        float* a = (float*)attn.DataPointer;   // [bt,1,hw,c]
        float* o = (float*)outT.DataPointer;   // [bt,c,h,w]
        long frame = (long)h * w;
        int hw = h * w;
        for (int i = 0; i < bt; i++)
            for (int ci = 0; ci < c; ci++)
                for (int token = 0; token < hw; token++)
                    o[((long)i * c + ci) * frame + token] = a[((long)i * hw + token) * c + ci];
    }
}

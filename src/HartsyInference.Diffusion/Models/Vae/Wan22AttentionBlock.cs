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

        Tensor normed = _norm.Forward(backend, x);              // [B,C,T,H,W], channel RMS per position
        Tensor frames = Vae3dLayout.ToFrames(backend, normed);  // [BT,C,H,W], on-device
        normed.Dispose();

        Tensor qkv = new Tensor(new TensorShape(bt, 3 * c, h, w), DType.F32);
        backend.Conv2D(qkv, frames, _qkvW!, _qkvB, 1, 1, 0, 0);
        frames.Dispose();

        // q,k,v: [BT, 1, HW, C]
        Tensor q = new Tensor(new TensorShape(bt, 1, hw, c), DType.F32);
        Tensor k = new Tensor(new TensorShape(bt, 1, hw, c), DType.F32);
        Tensor v = new Tensor(new TensorShape(bt, 1, hw, c), DType.F32);
        backend.SplitVaeQkv(q, k, v, qkv, bt, c, hw);
        qkv.Dispose();

        float scale = 1.0f / MathF.Sqrt(c);
        Tensor attn = new Tensor(new TensorShape(bt, 1, hw, c), DType.F32);
        backend.ScaledDotProductAttention(attn, q, k, v, null, scale);
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor attn4d = new Tensor(new TensorShape(bt, c, h, w), DType.F32);
        backend.VaeTokensToFrame(attn4d, attn, bt, c, hw);
        attn.Dispose();

        Tensor proj = new Tensor(new TensorShape(bt, c, h, w), DType.F32);
        backend.Conv2D(proj, attn4d, _projW!, _projB, 1, 1, 0, 0);
        attn4d.Dispose();

        Tensor proj5d = Vae3dLayout.FromFrames(backend, proj, b, c, t, h, w);
        proj.Dispose();
        Tensor outT = new Tensor(x.Shape, DType.F32);
        backend.Add(outT, x, proj5d);
        proj5d.Dispose();
        return outT;
    }

}

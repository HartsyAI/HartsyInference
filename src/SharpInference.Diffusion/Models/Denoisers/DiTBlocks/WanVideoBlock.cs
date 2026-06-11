using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Wan-Video DiT block (<c>WanTransformerBlock</c>), ported from diffusers. B=1 over <c>[S, dim]</c>: self-attention + per-head 3D RoPE → cross-attention to umT5 → gelu-approx FFN. AdaLN is 6-param (self-attn shift/scale/gate + FFN shift/scale/gate; cross-attn ungated). Pre-norms are FP32 LayerNorm (norm1/norm3 no-affine, norm2 affine when cross_attn_norm); QK-norm is RMSNorm-across-heads. Reuses backend <c>LayerNorm</c>/<c>RmsNorm</c>/<c>ScaledDotProductAttention</c> + <see cref="WanRope"/>.</summary>
public sealed unsafe class WanVideoBlock
{
    private readonly int _dim;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly float _eps;
    private readonly bool _crossAttnNorm;

    private Tensor? _scaleShift;        // [6, dim]
    private Tensor? _norm2W, _norm2B;   // FP32LayerNorm affine (cross), when cross_attn_norm
    // attn1 (self) + attn2 (cross)
    private Tensor?[] _q = new Tensor?[2], _qB = new Tensor?[2], _k = new Tensor?[2], _kB = new Tensor?[2],
        _v = new Tensor?[2], _vB = new Tensor?[2], _o = new Tensor?[2], _oB = new Tensor?[2], _nq = new Tensor?[2], _nk = new Tensor?[2];
    private Tensor? _ffProjW, _ffProjB, _ffOutW, _ffOutB;

    public WanVideoBlock(WanVideoConfig c, bool crossAttnNorm)
    {
        _dim = c.InnerDim;
        _heads = c.NumHeads;
        _headDim = c.HeadDim;
        _eps = c.Eps;
        _crossAttnNorm = crossAttnNorm;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _scaleShift = LoadF32(w, $"{prefix}.scale_shift_table");   // [1,6,dim] → flat 6*dim
        LoadAttn(w, $"{prefix}.attn1", 0);
        LoadAttn(w, $"{prefix}.attn2", 1);
        if (_crossAttnNorm) { _norm2W = LoadF32(w, $"{prefix}.norm2.weight"); w.TryGetValue($"{prefix}.norm2.bias", out _norm2B); }
        _ffProjW = w[$"{prefix}.ffn.net.0.proj.weight"]; w.TryGetValue($"{prefix}.ffn.net.0.proj.bias", out _ffProjB);
        _ffOutW = w[$"{prefix}.ffn.net.2.weight"]; w.TryGetValue($"{prefix}.ffn.net.2.bias", out _ffOutB);
    }

    private void LoadAttn(IReadOnlyDictionary<string, Tensor> w, string p, int i)
    {
        _q[i] = w[$"{p}.to_q.weight"]; w.TryGetValue($"{p}.to_q.bias", out Tensor? qb); _qB[i] = qb;
        _k[i] = w[$"{p}.to_k.weight"]; w.TryGetValue($"{p}.to_k.bias", out Tensor? kb); _kB[i] = kb;
        _v[i] = w[$"{p}.to_v.weight"]; w.TryGetValue($"{p}.to_v.bias", out Tensor? vb); _vB[i] = vb;
        _o[i] = w[$"{p}.to_out.0.weight"]; w.TryGetValue($"{p}.to_out.0.bias", out Tensor? ob); _oB[i] = ob;
        _nq[i] = LoadF32(w, $"{p}.norm_q.weight");
        _nk[i] = LoadF32(w, $"{p}.norm_k.weight");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_scaleShift is not null) yield return _scaleShift;
        if (_norm2W is not null) yield return _norm2W;
        if (_norm2B is not null) yield return _norm2B;
        for (int i = 0; i < 2; i++)
            foreach (Tensor? t in new[] { _q[i], _qB[i], _k[i], _kB[i], _v[i], _vB[i], _o[i], _oB[i], _nq[i], _nk[i] })
                if (t is not null) yield return t;
        foreach (Tensor? t in new[] { _ffProjW, _ffProjB, _ffOutW, _ffOutB }) if (t is not null) yield return t;
    }

    /// <summary>Forward over <c>[S, dim]</c>. <paramref name="temb"/> is the per-block modulation <c>[G, 6, dim]</c>
    /// (timestep_proj); the block adds its <c>scale_shift_table</c>. G=1 is the standard scalar-timestep path;
    /// G=latent-frames with <paramref name="tokensPerGroup"/> tokens each is the TI2V per-frame-timestep path
    /// (tokens are in (t,h,w) order, so each frame's tokens are contiguous). <paramref name="postCrossAttnHook"/>
    /// runs between the cross-attention residual and the FFN — Matrix-Game attaches its ActionModule there (the hook
    /// mutates the hidden state in place). <paramref name="selfAttnMask"/> is an optional additive mask broadcastable
    /// to <c>[1, heads, S, S]</c> — Matrix-Game 2's block-causal + local-window attention.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor encoder, Tensor temb, WanRope rope, Tensor cos, Tensor sin, int tokensPerGroup,
        Action<Tensor>? postCrossAttnHook = null, Tensor? selfAttnMask = null)
    {
        int s = (int)hidden.Shape[0];
        (Tensor shiftMsa, Tensor scaleMsa, Tensor gateMsa, Tensor cShift, Tensor cScale, Tensor cGate) = Modulation(temb);

        // 1. self-attn
        Tensor n1 = LayerNorm(hidden, null, null, s);
        ApplyShiftScale(n1, scaleMsa, shiftMsa, s, tokensPerGroup);
        Tensor attn1 = Attention(backend, n1, n1, 0, applyRope: true, rope, cos, sin, s, s, selfAttnMask);
        n1.Dispose();
        Tensor h1 = GatedAdd(hidden, attn1, gateMsa, s, tokensPerGroup);
        attn1.Dispose();

        // 2. cross-attn
        int l = (int)encoder.Shape[0];
        Tensor n2 = LayerNorm(h1, _norm2W, _norm2B, s);
        Tensor attn2 = Attention(backend, n2, encoder, 1, applyRope: false, rope, cos, sin, s, l);
        n2.Dispose();
        Tensor h2 = AddRows(h1, attn2, s);
        h1.Dispose();
        attn2.Dispose();

        postCrossAttnHook?.Invoke(h2);

        // 3. ffn
        Tensor n3 = LayerNorm(h2, null, null, s);
        ApplyShiftScale(n3, cScale, cShift, s, tokensPerGroup);
        Tensor ff = Ffn(backend, n3, s);
        n3.Dispose();
        Tensor outT = GatedAdd(h2, ff, cGate, s, tokensPerGroup);
        h2.Dispose();
        ff.Dispose();

        foreach (Tensor t in new[] { shiftMsa, scaleMsa, gateMsa, cShift, cScale, cGate }) t.Dispose();
        return outT;
    }

    private Tensor Attention(IBackend backend, Tensor qInput, Tensor kvInput, int idx, bool applyRope,
        WanRope rope, Tensor cos, Tensor sin, int sq, int sk, Tensor? mask = null)
    {
        Tensor q = new Tensor(new TensorShape(sq, _dim), DType.F32); backend.Linear(q, qInput, _q[idx]!, _qB[idx]);
        Tensor k = new Tensor(new TensorShape(sk, _dim), DType.F32); backend.Linear(k, kvInput, _k[idx]!, _kB[idx]);
        Tensor v = new Tensor(new TensorShape(sk, _dim), DType.F32); backend.Linear(v, kvInput, _v[idx]!, _vB[idx]);

        Tensor qn = new Tensor(q.Shape, DType.F32); backend.RmsNorm(qn, q, _nq[idx]!, _eps); q.Dispose();
        Tensor kn = new Tensor(k.Shape, DType.F32); backend.RmsNorm(kn, k, _nk[idx]!, _eps); k.Dispose();

        if (applyRope)   // per-head; [S,dim] is contiguous as [S,heads,headDim]
        {
            rope.ApplyRotary(qn, cos, sin, _heads);
            rope.ApplyRotary(kn, cos, sin, _heads);
        }

        Tensor qMh = ToBhsd(qn, sq); qn.Dispose();
        Tensor kMh = ToBhsd(kn, sk); kn.Dispose();
        Tensor vMh = ToBhsd(v, sk); v.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attn = new Tensor(new TensorShape(1, _heads, sq, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, mask, scale);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor flat = FromBhsd(attn, sq); attn.Dispose();
        Tensor outT = new Tensor(new TensorShape(sq, _dim), DType.F32); backend.Linear(outT, flat, _o[idx]!, _oB[idx]);
        flat.Dispose();
        return outT;
    }

    private Tensor Ffn(IBackend backend, Tensor x, int s)
    {
        int inner = (int)_ffProjW!.Shape[0];
        Tensor proj = new Tensor(new TensorShape(s, inner), DType.F32); backend.Linear(proj, x, _ffProjW!, _ffProjB);
        Tensor act = new Tensor(proj.Shape, DType.F32); backend.Gelu(act, proj); proj.Dispose();
        Tensor outT = new Tensor(new TensorShape(s, _dim), DType.F32); backend.Linear(outT, act, _ffOutW!, _ffOutB);
        act.Dispose();
        return outT;
    }

    /// <summary>Adds <c>scale_shift_table</c> to the timestep_proj <c>[G, 6, dim]</c>; returns 6 modulation tensors of <c>[G, dim]</c>.</summary>
    private (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) Modulation(Tensor temb)
    {
        int g = (int)temb.Shape[0];
        float* ss = (float*)_scaleShift!.DataPointer;
        float* tb = (float*)temb.DataPointer;
        Tensor[] outs = new Tensor[6];
        for (int m = 0; m < 6; m++)
        {
            Tensor o = new Tensor(new TensorShape(g, _dim), DType.F32);
            float* op = (float*)o.DataPointer;
            for (int gi = 0; gi < g; gi++)
                for (int d = 0; d < _dim; d++)
                    op[gi * _dim + d] = ss[m * _dim + d] + tb[((long)gi * 6 + m) * _dim + d];
            outs[m] = o;
        }
        return (outs[0], outs[1], outs[2], outs[3], outs[4], outs[5]);
    }

    /// <summary>FP32 LayerNorm over the last dim with optional affine.</summary>
    private Tensor LayerNorm(Tensor x, Tensor? weight, Tensor? bias, int s)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
        float* xp = (float*)x.DataPointer; float* op = (float*)o.DataPointer;
        float* wp = weight is null ? null : (float*)weight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;
        for (int i = 0; i < s; i++)
        {
            long off = (long)i * _dim;
            double mean = 0;
            for (int d = 0; d < _dim; d++) mean += xp[off + d];
            mean /= _dim;
            double var = 0;
            for (int d = 0; d < _dim; d++) { double dd = xp[off + d] - mean; var += dd * dd; }
            float inv = 1f / MathF.Sqrt((float)(var / _dim) + _eps);
            for (int d = 0; d < _dim; d++)
            {
                float n = (float)((xp[off + d] - mean) * inv);
                op[off + d] = wp is null ? n : n * wp[d] + (bp is null ? 0f : bp[d]);
            }
        }
        return o;
    }

    private void ApplyShiftScale(Tensor x, Tensor scale, Tensor shift, int s, int tokensPerGroup)
    {
        float* xp = (float*)x.DataPointer; float* sc = (float*)scale.DataPointer; float* sh = (float*)shift.DataPointer;
        for (int i = 0; i < s; i++)
        {
            long g = (long)(i / tokensPerGroup) * _dim;
            for (int d = 0; d < _dim; d++) xp[i * _dim + d] = xp[i * _dim + d] * (1f + sc[g + d]) + sh[g + d];
        }
    }

    private Tensor GatedAdd(Tensor residual, Tensor value, Tensor gate, int s, int tokensPerGroup)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
        float* rp = (float*)residual.DataPointer; float* vp = (float*)value.DataPointer; float* gp = (float*)gate.DataPointer; float* op = (float*)o.DataPointer;
        for (int i = 0; i < s; i++)
        {
            long g = (long)(i / tokensPerGroup) * _dim;
            for (int d = 0; d < _dim; d++) op[i * _dim + d] = rp[i * _dim + d] + gp[g + d] * vp[i * _dim + d];
        }
        return o;
    }

    private Tensor AddRows(Tensor a, Tensor b, int s)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
        long n = (long)s * _dim;
        float* ap = (float*)a.DataPointer; float* bp = (float*)b.DataPointer; float* op = (float*)o.DataPointer;
        for (long i = 0; i < n; i++) op[i] = ap[i] + bp[i];
        return o;
    }

    private Tensor ToBhsd(Tensor x, int s)
    {
        Tensor o = new Tensor(new TensorShape(1, _heads, s, _headDim), DType.F32);
        float* xp = (float*)x.DataPointer; float* op = (float*)o.DataPointer;
        for (int i = 0; i < s; i++)
            for (int h = 0; h < _heads; h++)
                Buffer.MemoryCopy(xp + (long)i * _dim + (long)h * _headDim, op + ((long)h * s + i) * _headDim, (long)_headDim * 4, (long)_headDim * 4);
        return o;
    }

    private Tensor FromBhsd(Tensor x, int s)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
        float* xp = (float*)x.DataPointer; float* op = (float*)o.DataPointer;
        for (int h = 0; h < _heads; h++)
            for (int i = 0; i < s; i++)
                Buffer.MemoryCopy(xp + ((long)h * s + i) * _headDim, op + (long)i * _dim + (long)h * _headDim, (long)_headDim * 4, (long)_headDim * 4);
        return o;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string k) { Tensor t = w[k]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }
}

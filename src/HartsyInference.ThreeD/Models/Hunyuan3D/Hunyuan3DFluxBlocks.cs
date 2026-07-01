using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Hunyuan3D-2 Flux DoubleStreamBlock (no RoPE), <b>GPU-resident</b>: all per-block glue (LayerNorm,
/// modulation, fused-QKV split, head permute, QK-norm, joint concat/split, gated residual) routes through
/// <see cref="IBackend"/> ops so the activation never leaves the device — no mid-forward <c>DataPointer</c> reads.
/// Mirrors the verified <c>ChromaDoubleStreamBlock</c> minus RoPE, with a fused <c>{img,txt}_attn.qkv</c>.
/// <b>Assumes B=1</b> (CFG runs as two batch-1 passes), matching the seq-dim split via <c>SliceRows</c>.</summary>
internal sealed unsafe class Hunyuan3DDoubleBlock
{
    private readonly int _width, _numHeads, _headDim, _mlpDim;
    private readonly QkNorm _imgQN, _imgKN, _txtQN, _txtKN;
    private readonly SwiGluFfn _imgFfn, _txtFfn;
    private Tensor? _imgModW, _imgModB, _txtModW, _txtModB;
    private Tensor? _imgQkvW, _imgQkvB, _imgProjW, _imgProjB;
    private Tensor? _txtQkvW, _txtQkvB, _txtProjW, _txtProjB;

    public Hunyuan3DDoubleBlock(int width, int numHeads, int mlpDim)
    {
        _width = width; _numHeads = numHeads; _headDim = width / numHeads; _mlpDim = mlpDim;
        _imgQN = new QkNorm(_headDim); _imgKN = new QkNorm(_headDim);
        _txtQN = new QkNorm(_headDim); _txtKN = new QkNorm(_headDim);
        _imgFfn = new SwiGluFfn(width, mlpDim); _txtFfn = new SwiGluFfn(width, mlpDim);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _imgModW = F(w, $"{p}.img_mod.lin.weight"); _imgModB = F(w, $"{p}.img_mod.lin.bias");
        _txtModW = F(w, $"{p}.txt_mod.lin.weight"); _txtModB = F(w, $"{p}.txt_mod.lin.bias");
        _imgQkvW = F(w, $"{p}.img_attn.qkv.weight"); _imgQkvB = F(w, $"{p}.img_attn.qkv.bias");
        _imgProjW = F(w, $"{p}.img_attn.proj.weight"); _imgProjB = F(w, $"{p}.img_attn.proj.bias");
        _txtQkvW = F(w, $"{p}.txt_attn.qkv.weight"); _txtQkvB = F(w, $"{p}.txt_attn.qkv.bias");
        _txtProjW = F(w, $"{p}.txt_attn.proj.weight"); _txtProjB = F(w, $"{p}.txt_attn.proj.bias");
        _imgQN.LoadWeights(F(w, $"{p}.img_attn.norm.query_norm.scale")); _imgKN.LoadWeights(F(w, $"{p}.img_attn.norm.key_norm.scale"));
        _txtQN.LoadWeights(F(w, $"{p}.txt_attn.norm.query_norm.scale")); _txtKN.LoadWeights(F(w, $"{p}.txt_attn.norm.key_norm.scale"));
        _imgFfn.LoadGeluWeights(F(w, $"{p}.img_mlp.0.weight"), F(w, $"{p}.img_mlp.0.bias"), F(w, $"{p}.img_mlp.2.weight"), F(w, $"{p}.img_mlp.2.bias"));
        _txtFfn.LoadGeluWeights(F(w, $"{p}.txt_mlp.0.weight"), F(w, $"{p}.txt_mlp.0.bias"), F(w, $"{p}.txt_mlp.2.weight"), F(w, $"{p}.txt_mlp.2.bias"));
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_imgModW, _imgModB, _txtModW, _txtModB, _imgQkvW, _imgQkvB, _imgProjW, _imgProjB, _txtQkvW, _txtQkvB, _txtProjW, _txtProjB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
        foreach (Tensor t in _imgQN.EnumerateWeights()) yield return t;
        foreach (Tensor t in _imgKN.EnumerateWeights()) yield return t;
        foreach (Tensor t in _txtQN.EnumerateWeights()) yield return t;
        foreach (Tensor t in _txtKN.EnumerateWeights()) yield return t;
        foreach (Tensor t in _imgFfn.EnumerateWeights()) yield return t;
        foreach (Tensor t in _txtFfn.EnumerateWeights()) yield return t;
    }

    public (Tensor img, Tensor txt) Forward(IBackend backend, Tensor img, Tensor txt, Tensor vec)
    {
        int nImg = (int)img.Shape[1], nTxt = (int)txt.Shape[1], total = nImg + nTxt;
        float scale = 1f / MathF.Sqrt(_headDim);
        TensorShape imgShape = new(1, nImg, _width), txtShape = new(1, nTxt, _width);
        TensorShape imgHeads = new(1, nImg, _numHeads, _headDim), txtHeads = new(1, nTxt, _numHeads, _headDim);
        TensorShape jointFlat = new(1, total, _width), jointMh = new(1, _numHeads, total, _headDim);

        Tensor[] im = Hunyuan3DGpuOps.ModParams(backend, vec, _imgModW!, _imgModB!, 6, _width);
        Tensor[] tm = Hunyuan3DGpuOps.ModParams(backend, vec, _txtModW!, _txtModB!, 6, _width);

        // Attention: modulated pre-norm → fused QKV → split → QK-norm → joint concat → permute → SDPA.
        (Tensor iQ, Tensor iK, Tensor iV) = Qkv(backend, img, im[0], im[1], _imgQkvW!, _imgQkvB!, _imgQN, _imgKN, imgShape, imgHeads);
        (Tensor tQ, Tensor tK, Tensor tV) = Qkv(backend, txt, tm[0], tm[1], _txtQkvW!, _txtQkvB!, _txtQN, _txtKN, txtShape, txtHeads);

        Tensor jQf = new(jointFlat, DType.F32); backend.Concat(jQf, [tQ, iQ], 1);
        Tensor jKf = new(jointFlat, DType.F32); backend.Concat(jKf, [tK, iK], 1);
        Tensor jVf = new(jointFlat, DType.F32); backend.Concat(jVf, [tV, iV], 1);
        iQ.Dispose(); iK.Dispose(); iV.Dispose(); tQ.Dispose(); tK.Dispose(); tV.Dispose();
        Tensor jQ = new(jointMh, DType.F32); backend.Permute0213(jQ, jQf, total, _numHeads, _headDim); jQf.Dispose();
        Tensor jK = new(jointMh, DType.F32); backend.Permute0213(jK, jKf, total, _numHeads, _headDim); jKf.Dispose();
        Tensor jV = new(jointMh, DType.F32); backend.Permute0213(jV, jVf, total, _numHeads, _headDim); jVf.Dispose();
        Tensor attn = new(jointMh, DType.F32); backend.ScaledDotProductAttention(attn, jQ, jK, jV, null, scale);
        jQ.Dispose(); jK.Dispose(); jV.Dispose();
        Tensor attnFlat = new(jointFlat, DType.F32); backend.Permute0213(attnFlat, attn, _numHeads, total, _headDim); attn.Dispose();
        Tensor tAttn = new(txtShape, DType.F32); backend.SliceRows(tAttn, attnFlat, 0);
        Tensor iAttn = new(imgShape, DType.F32); backend.SliceRows(iAttn, attnFlat, nTxt);
        attnFlat.Dispose();

        Tensor iAfter = AttnResidual(backend, img, iAttn, _imgProjW!, _imgProjB!, im[2], imgShape); iAttn.Dispose();
        Tensor tAfter = AttnResidual(backend, txt, tAttn, _txtProjW!, _txtProjB!, tm[2], txtShape); tAttn.Dispose();

        Tensor iFinal = MlpResidual(backend, iAfter, _imgFfn, im[3], im[4], im[5], imgShape, nImg); iAfter.Dispose();
        Tensor tFinal = MlpResidual(backend, tAfter, _txtFfn, tm[3], tm[4], tm[5], txtShape, nTxt); tAfter.Dispose();

        foreach (Tensor t in im) t.Dispose();
        foreach (Tensor t in tm) t.Dispose();
        return (iFinal, tFinal);
    }

    private (Tensor q, Tensor k, Tensor v) Qkv(IBackend backend, Tensor x, Tensor shift, Tensor scale,
        Tensor qkvW, Tensor qkvB, QkNorm qn, QkNorm kn, TensorShape flat, TensorShape heads)
    {
        Tensor mod = Hunyuan3DGpuOps.NormModulate(backend, x, shift, scale, flat);
        Tensor qkv = new(new TensorShape(1, flat[1], 3 * _width), DType.F32); backend.Linear(qkv, mod, qkvW, qkvB); mod.Dispose();
        Tensor q = new(heads, DType.F32); backend.SliceLastDim(q, qkv, 0);
        Tensor k = new(heads, DType.F32); backend.SliceLastDim(k, qkv, _width);
        Tensor v = new(heads, DType.F32); backend.SliceLastDim(v, qkv, 2 * _width);
        qkv.Dispose();
        Tensor qOut = new(heads, DType.F32); backend.RmsNorm(qOut, q, qn.Weight, qn.Eps); q.Dispose();
        Tensor kOut = new(heads, DType.F32); backend.RmsNorm(kOut, k, kn.Weight, kn.Eps); k.Dispose();
        return (qOut, kOut, v);
    }

    private static Tensor AttnResidual(IBackend backend, Tensor x, Tensor attn, Tensor projW, Tensor projB, Tensor gate, TensorShape shape)
    {
        Tensor proj = new(shape, DType.F32); backend.Linear(proj, attn, projW, projB);
        Tensor res = new(shape, DType.F32); backend.GatedResidualLastDim(res, x, proj, gate); proj.Dispose();
        return res;
    }

    private Tensor MlpResidual(IBackend backend, Tensor x, SwiGluFfn ffn, Tensor shift, Tensor scale, Tensor gate, TensorShape shape, int n)
    {
        Tensor mod = Hunyuan3DGpuOps.NormModulate(backend, x, shift, scale, shape);
        Tensor mlp = ffn.Forward(backend, mod, 1, n); mod.Dispose();
        Tensor res = new(shape, DType.F32); backend.GatedResidualLastDim(res, x, mlp, gate); mlp.Dispose();
        return res;
    }

    private static Tensor F(IReadOnlyDictionary<string, Tensor> w, string k) => Hunyuan3DDit.F32(w[k]);
}

/// <summary>Hunyuan3D-2 Flux SingleStreamBlock (no RoPE), <b>GPU-resident</b>: parallel QKV+MLP via <c>linear1</c>,
/// joint attention, then <c>linear2(cat(attn, gelu(mlp)))</c> gated residual. Mirrors hy3dgen <c>SingleStreamBlock</c>.</summary>
internal sealed unsafe class Hunyuan3DSingleBlock
{
    private readonly int _width, _numHeads, _headDim, _mlpDim;
    private readonly QkNorm _qn, _kn;
    private Tensor? _modW, _modB, _lin1W, _lin1B, _lin2W, _lin2B;

    public Hunyuan3DSingleBlock(int width, int numHeads, int mlpDim)
    {
        _width = width; _numHeads = numHeads; _headDim = width / numHeads; _mlpDim = mlpDim;
        _qn = new QkNorm(_headDim); _kn = new QkNorm(_headDim);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _modW = F(w, $"{p}.modulation.lin.weight"); _modB = F(w, $"{p}.modulation.lin.bias");
        _lin1W = F(w, $"{p}.linear1.weight"); _lin1B = F(w, $"{p}.linear1.bias");
        _lin2W = F(w, $"{p}.linear2.weight"); _lin2B = F(w, $"{p}.linear2.bias");
        _qn.LoadWeights(F(w, $"{p}.norm.query_norm.scale")); _kn.LoadWeights(F(w, $"{p}.norm.key_norm.scale"));
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_modW, _modB, _lin1W, _lin1B, _lin2W, _lin2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
        foreach (Tensor t in _qn.EnumerateWeights()) yield return t;
        foreach (Tensor t in _kn.EnumerateWeights()) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor x, Tensor vec)
    {
        int s = (int)x.Shape[1];
        float scale = 1f / MathF.Sqrt(_headDim);
        TensorShape flat = new(1, s, _width), heads = new(1, s, _numHeads, _headDim), mh = new(1, _numHeads, s, _headDim);

        Tensor[] m = Hunyuan3DGpuOps.ModParams(backend, vec, _modW!, _modB!, 3, _width);
        Tensor mod = Hunyuan3DGpuOps.NormModulate(backend, x, m[0], m[1], flat);
        Tensor lin1 = new(new TensorShape(1, s, 3 * _width + _mlpDim), DType.F32); backend.Linear(lin1, mod, _lin1W!, _lin1B!); mod.Dispose();
        Tensor qkv = new(new TensorShape(1, s, 3 * _width), DType.F32); backend.SliceLastDim(qkv, lin1, 0);
        Tensor mlp = new(new TensorShape(1, s, _mlpDim), DType.F32); backend.SliceLastDim(mlp, lin1, 3 * _width);
        lin1.Dispose();

        Tensor q = new(heads, DType.F32); backend.SliceLastDim(q, qkv, 0);
        Tensor k = new(heads, DType.F32); backend.SliceLastDim(k, qkv, _width);
        Tensor v = new(heads, DType.F32); backend.SliceLastDim(v, qkv, 2 * _width);
        qkv.Dispose();
        Tensor qn = new(heads, DType.F32); backend.RmsNorm(qn, q, _qn.Weight, _qn.Eps); q.Dispose();
        Tensor kn = new(heads, DType.F32); backend.RmsNorm(kn, k, _kn.Weight, _kn.Eps); k.Dispose();
        Tensor qP = new(mh, DType.F32); backend.Permute0213(qP, qn, s, _numHeads, _headDim); qn.Dispose();
        Tensor kP = new(mh, DType.F32); backend.Permute0213(kP, kn, s, _numHeads, _headDim); kn.Dispose();
        Tensor vP = new(mh, DType.F32); backend.Permute0213(vP, v, s, _numHeads, _headDim); v.Dispose();
        Tensor attn = new(mh, DType.F32); backend.ScaledDotProductAttention(attn, qP, kP, vP, null, scale); qP.Dispose(); kP.Dispose(); vP.Dispose();
        Tensor attnFlat = new(flat, DType.F32); backend.Permute0213(attnFlat, attn, _numHeads, s, _headDim); attn.Dispose();

        Tensor mlpAct = new(new TensorShape(1, s, _mlpDim), DType.F32); backend.Gelu(mlpAct, mlp); mlp.Dispose();
        Tensor cat = new(new TensorShape(1, s, _width + _mlpDim), DType.F32); backend.Concat(cat, [attnFlat, mlpAct], 2); attnFlat.Dispose(); mlpAct.Dispose();
        Tensor outp = new(flat, DType.F32); backend.Linear(outp, cat, _lin2W!, _lin2B!); cat.Dispose();
        Tensor res = new(flat, DType.F32); backend.GatedResidualLastDim(res, x, outp, m[2]); outp.Dispose();
        foreach (Tensor t in m) t.Dispose();
        return res;
    }

    private static Tensor F(IReadOnlyDictionary<string, Tensor> w, string k) => Hunyuan3DDit.F32(w[k]);
}

/// <summary>GPU-resident glue for the Hunyuan3D Flux DiT: modulation-param extraction and the
/// <c>(1+scale)·LayerNormNoAffine(x)+shift</c> modulation, all via <see cref="IBackend"/> ops (no host reads).</summary>
internal static unsafe class Hunyuan3DGpuOps
{
    /// <summary>SiLU(vec) → Linear → [B, K·W], then <see cref="IBackend.SliceLastDim"/> into K params each [B,W]
    /// (hy3dgen Modulation chunk order: shift, scale, gate [×2 for double]). All GPU-resident.</summary>
    public static Tensor[] ModParams(IBackend backend, Tensor vec, Tensor modW, Tensor modB, int k, int width)
    {
        int b = (int)vec.Shape[0];
        Tensor act = new(vec.Shape, DType.F32); backend.Silu(act, vec);
        Tensor mod = new(new TensorShape(b, k * width), DType.F32); backend.Linear(mod, act, modW, modB); act.Dispose();
        Tensor[] p = new Tensor[k];
        for (int i = 0; i < k; i++) { p[i] = new(new TensorShape(b, width), DType.F32); backend.SliceLastDim(p[i], mod, i * width); }
        mod.Dispose();
        return p;
    }

    /// <summary><c>(1 + scale)·LayerNormNoAffine(x, eps 1e-6) + shift</c>, scale/shift broadcast over seq ([B,W]).</summary>
    public static Tensor NormModulate(IBackend backend, Tensor x, Tensor shift, Tensor scale, TensorShape shape)
    {
        Tensor normed = new(shape, DType.F32); backend.LayerNormNoAffine(normed, x, 1e-6f);
        Tensor scalePlus1 = new(scale.Shape, DType.F32); backend.AddScalar(scalePlus1, scale, 1f);
        Tensor outp = new(shape, DType.F32); backend.AffineBroadcastLastDim(outp, normed, scalePlus1, shift);
        normed.Dispose(); scalePlus1.Dispose();
        return outp;
    }
}

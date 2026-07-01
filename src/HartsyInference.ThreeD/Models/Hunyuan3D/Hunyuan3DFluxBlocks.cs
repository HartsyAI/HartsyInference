using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Hunyuan3D-2 Flux DoubleStreamBlock (no RoPE): dual-stream joint attention between the latent (img)
/// and DINOv2 conditioning (txt) streams, then independent GELU-tanh MLPs. Mirrors
/// <c>hy3dgen ... DoubleStreamBlock</c> with fused <c>{img,txt}_attn.qkv</c> and QKNorm (RMSNorm on headDim).</summary>
internal sealed unsafe class Hunyuan3DDoubleBlock
{
    private readonly int _width, _numHeads, _headDim, _mlpDim;
    private readonly AdaLNModulation _imgMod, _txtMod;
    private readonly QkNorm _imgQN, _imgKN, _txtQN, _txtKN;
    private readonly SwiGluFfn _imgFfn, _txtFfn;
    private Tensor? _imgQkvW, _imgQkvB, _imgProjW, _imgProjB;
    private Tensor? _txtQkvW, _txtQkvB, _txtProjW, _txtProjB;

    public Hunyuan3DDoubleBlock(int width, int numHeads, int mlpDim)
    {
        _width = width; _numHeads = numHeads; _headDim = width / numHeads; _mlpDim = mlpDim;
        _imgMod = new AdaLNModulation(width, 6); _txtMod = new AdaLNModulation(width, 6);
        _imgQN = new QkNorm(_headDim); _imgKN = new QkNorm(_headDim);
        _txtQN = new QkNorm(_headDim); _txtKN = new QkNorm(_headDim);
        _imgFfn = new SwiGluFfn(width, mlpDim); _txtFfn = new SwiGluFfn(width, mlpDim);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _imgMod.LoadWeights(F(w, $"{p}.img_mod.lin.weight"), F(w, $"{p}.img_mod.lin.bias"));
        _txtMod.LoadWeights(F(w, $"{p}.txt_mod.lin.weight"), F(w, $"{p}.txt_mod.lin.bias"));
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
        foreach (Tensor t in _imgMod.EnumerateWeights()) yield return t;
        foreach (Tensor t in _txtMod.EnumerateWeights()) yield return t;
        Tensor?[] all = [_imgQkvW, _imgQkvB, _imgProjW, _imgProjB, _txtQkvW, _txtQkvB, _txtProjW, _txtProjB];
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
        int b = (int)img.Shape[0], nImg = (int)img.Shape[1], nTxt = (int)txt.Shape[1], total = nImg + nTxt;

        Tensor[] im = _imgMod.Forward(backend, vec);
        Tensor[] tm = _txtMod.Forward(backend, vec);

        // Modulated pre-norm → fused QKV → split → QKNorm → multi-head.
        (Tensor imgQ, Tensor imgK, Tensor imgV) = QkvHeads(backend, img, im[0], im[1], _imgQkvW!, _imgQkvB!, _imgQN, _imgKN, b, nImg);
        (Tensor txtQ, Tensor txtK, Tensor txtV) = QkvHeads(backend, txt, tm[0], tm[1], _txtQkvW!, _txtQkvB!, _txtQN, _txtKN, b, nTxt);

        // Joint attention over concat[txt, img] (txt FIRST).
        TensorShape jointMh = new(b, _numHeads, total, _headDim);
        Tensor jq = new(jointMh, DType.F32), jk = new(jointMh, DType.F32), jv = new(jointMh, DType.F32);
        Hunyuan3DDitOps.ConcatSeqMh(jq, txtQ, imgQ, b, _numHeads, nTxt, nImg, _headDim);
        Hunyuan3DDitOps.ConcatSeqMh(jk, txtK, imgK, b, _numHeads, nTxt, nImg, _headDim);
        Hunyuan3DDitOps.ConcatSeqMh(jv, txtV, imgV, b, _numHeads, nTxt, nImg, _headDim);
        imgQ.Dispose(); imgK.Dispose(); imgV.Dispose(); txtQ.Dispose(); txtK.Dispose(); txtV.Dispose();

        Tensor attn = new(jointMh, DType.F32);
        backend.ScaledDotProductAttention(attn, jq, jk, jv, null, 1f / MathF.Sqrt(_headDim));
        jq.Dispose(); jk.Dispose(); jv.Dispose();

        Tensor txtAttnMh = new(new TensorShape(b, _numHeads, nTxt, _headDim), DType.F32);
        Tensor imgAttnMh = new(new TensorShape(b, _numHeads, nImg, _headDim), DType.F32);
        Hunyuan3DDitOps.SplitSeqMh(txtAttnMh, imgAttnMh, attn, b, _numHeads, nTxt, nImg, _headDim);
        attn.Dispose();

        // img/txt attention output → proj → gated residual (gate = mod[2]).
        Tensor img1 = AttnProjResidual(backend, img, imgAttnMh, _imgProjW!, _imgProjB!, im[2], b, nImg);
        imgAttnMh.Dispose();
        Tensor txt1 = AttnProjResidual(backend, txt, txtAttnMh, _txtProjW!, _txtProjB!, tm[2], b, nTxt);
        txtAttnMh.Dispose();

        // MLP: norm2 → modulate(mod[3],mod[4]) → GELU MLP → gated residual (mod[5]).
        Tensor img2 = MlpResidual(backend, img1, _imgFfn, im[3], im[4], im[5], b, nImg);
        img1.Dispose();
        Tensor txt2 = MlpResidual(backend, txt1, _txtFfn, tm[3], tm[4], tm[5], b, nTxt);
        txt1.Dispose();

        foreach (Tensor t in im) t.Dispose();
        foreach (Tensor t in tm) t.Dispose();
        return (img2, txt2);
    }

    private (Tensor q, Tensor k, Tensor v) QkvHeads(IBackend backend, Tensor x, Tensor shift, Tensor scale,
        Tensor qkvW, Tensor qkvB, QkNorm qn, QkNorm kn, int b, int s)
    {
        Tensor normed = new(x.Shape, DType.F32); Hunyuan3DDitOps.LayerNormNoAffine(normed, x, b, s, _width);
        Tensor modd = AdaLNModulation.ApplyModulation(normed, shift, scale, b, s, _width); normed.Dispose();
        Tensor qkv = new(new TensorShape(b, s, 3 * _width), DType.F32); backend.Linear(qkv, modd, qkvW, qkvB); modd.Dispose();
        Tensor q = new(new TensorShape(b, s, _width), DType.F32), k = new(new TensorShape(b, s, _width), DType.F32), v = new(new TensorShape(b, s, _width), DType.F32);
        Hunyuan3DDitOps.SplitQkv(qkv, q, k, v, b, s, _width); qkv.Dispose();
        int vecs = b * s * _numHeads;
        Tensor qn2 = new(q.Shape, DType.F32); qn.Forward(qn2, q, vecs); q.Dispose();
        Tensor kn2 = new(k.Shape, DType.F32); kn.Forward(kn2, k, vecs); k.Dispose();
        Tensor qMh = new(new TensorShape(b, _numHeads, s, _headDim), DType.F32);
        Tensor kMh = new(new TensorShape(b, _numHeads, s, _headDim), DType.F32);
        Tensor vMh = new(new TensorShape(b, _numHeads, s, _headDim), DType.F32);
        Hunyuan3DDitOps.ToHeads(qMh, qn2, b, s, _numHeads, _headDim); qn2.Dispose();
        Hunyuan3DDitOps.ToHeads(kMh, kn2, b, s, _numHeads, _headDim); kn2.Dispose();
        Hunyuan3DDitOps.ToHeads(vMh, v, b, s, _numHeads, _headDim); v.Dispose();
        return (qMh, kMh, vMh);
    }

    private Tensor AttnProjResidual(IBackend backend, Tensor x, Tensor attnMh, Tensor projW, Tensor projB, Tensor gate, int b, int s)
    {
        Tensor attn = new(new TensorShape(b, s, _width), DType.F32); Hunyuan3DDitOps.FromHeads(attn, attnMh, b, s, _numHeads, _headDim);
        Tensor proj = new(new TensorShape(b, s, _width), DType.F32); backend.Linear(proj, attn, projW, projB); attn.Dispose();
        Tensor res = AdaLNModulation.ApplyGatedResidual(x, proj, gate, b, s, _width); proj.Dispose();
        return res;
    }

    private Tensor MlpResidual(IBackend backend, Tensor x, SwiGluFfn ffn, Tensor shift, Tensor scale, Tensor gate, int b, int s)
    {
        Tensor normed = new(x.Shape, DType.F32); Hunyuan3DDitOps.LayerNormNoAffine(normed, x, b, s, _width);
        Tensor modd = AdaLNModulation.ApplyModulation(normed, shift, scale, b, s, _width); normed.Dispose();
        Tensor mlp = ffn.Forward(backend, modd, b, s); modd.Dispose();
        Tensor res = AdaLNModulation.ApplyGatedResidual(x, mlp, gate, b, s, _width); mlp.Dispose();
        return res;
    }

    private static Tensor F(IReadOnlyDictionary<string, Tensor> w, string k) => Hunyuan3DDit.F32(w[k]);
}

/// <summary>Hunyuan3D-2 Flux SingleStreamBlock (no RoPE): parallel QKV+MLP through <c>linear1</c>, joint attention,
/// then <c>linear2(cat(attn, gelu(mlp)))</c> with a gated residual. Mirrors <c>hy3dgen ... SingleStreamBlock</c>.</summary>
internal sealed unsafe class Hunyuan3DSingleBlock
{
    private readonly int _width, _numHeads, _headDim, _mlpDim;
    private readonly AdaLNModulation _mod;
    private readonly QkNorm _qn, _kn;
    private Tensor? _lin1W, _lin1B, _lin2W, _lin2B;

    public Hunyuan3DSingleBlock(int width, int numHeads, int mlpDim)
    {
        _width = width; _numHeads = numHeads; _headDim = width / numHeads; _mlpDim = mlpDim;
        _mod = new AdaLNModulation(width, 3);
        _qn = new QkNorm(_headDim); _kn = new QkNorm(_headDim);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _mod.LoadWeights(F(w, $"{p}.modulation.lin.weight"), F(w, $"{p}.modulation.lin.bias"));
        _lin1W = F(w, $"{p}.linear1.weight"); _lin1B = F(w, $"{p}.linear1.bias");
        _lin2W = F(w, $"{p}.linear2.weight"); _lin2B = F(w, $"{p}.linear2.bias");
        _qn.LoadWeights(F(w, $"{p}.norm.query_norm.scale")); _kn.LoadWeights(F(w, $"{p}.norm.key_norm.scale"));
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _mod.EnumerateWeights()) yield return t;
        Tensor?[] all = [_lin1W, _lin1B, _lin2W, _lin2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
        foreach (Tensor t in _qn.EnumerateWeights()) yield return t;
        foreach (Tensor t in _kn.EnumerateWeights()) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor x, Tensor vec)
    {
        int b = (int)x.Shape[0], s = (int)x.Shape[1];
        Tensor[] m = _mod.Forward(backend, vec);   // [shift, scale, gate]

        Tensor normed = new(x.Shape, DType.F32); Hunyuan3DDitOps.LayerNormNoAffine(normed, x, b, s, _width);
        Tensor modd = AdaLNModulation.ApplyModulation(normed, m[0], m[1], b, s, _width); normed.Dispose();

        // linear1 → split [3*width qkv, mlpDim].
        Tensor lin1 = new(new TensorShape(b, s, 3 * _width + _mlpDim), DType.F32); backend.Linear(lin1, modd, _lin1W!, _lin1B!); modd.Dispose();
        Tensor qkv = new(new TensorShape(b, s, 3 * _width), DType.F32);
        Tensor mlp = new(new TensorShape(b, s, _mlpDim), DType.F32);
        Hunyuan3DDitOps.SplitLastDim(lin1, qkv, mlp, b, s, 3 * _width, _mlpDim); lin1.Dispose();

        Tensor q = new(new TensorShape(b, s, _width), DType.F32), k = new(new TensorShape(b, s, _width), DType.F32), v = new(new TensorShape(b, s, _width), DType.F32);
        Hunyuan3DDitOps.SplitQkv(qkv, q, k, v, b, s, _width); qkv.Dispose();
        int vecs = b * s * _numHeads;
        Tensor qn2 = new(q.Shape, DType.F32); _qn.Forward(qn2, q, vecs); q.Dispose();
        Tensor kn2 = new(k.Shape, DType.F32); _kn.Forward(kn2, k, vecs); k.Dispose();
        Tensor qMh = new(new TensorShape(b, _numHeads, s, _headDim), DType.F32);
        Tensor kMh = new(new TensorShape(b, _numHeads, s, _headDim), DType.F32);
        Tensor vMh = new(new TensorShape(b, _numHeads, s, _headDim), DType.F32);
        Hunyuan3DDitOps.ToHeads(qMh, qn2, b, s, _numHeads, _headDim); qn2.Dispose();
        Hunyuan3DDitOps.ToHeads(kMh, kn2, b, s, _numHeads, _headDim); kn2.Dispose();
        Hunyuan3DDitOps.ToHeads(vMh, v, b, s, _numHeads, _headDim); v.Dispose();

        Tensor attnMh = new(new TensorShape(b, _numHeads, s, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attnMh, qMh, kMh, vMh, null, 1f / MathF.Sqrt(_headDim));
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();
        Tensor attn = new(new TensorShape(b, s, _width), DType.F32); Hunyuan3DDitOps.FromHeads(attn, attnMh, b, s, _numHeads, _headDim); attnMh.Dispose();

        // gelu-tanh on the mlp stream, cat(attn, mlp) → linear2.
        Hunyuan3DDitOps.GeluTanhInPlace(mlp);
        Tensor cat = new(new TensorShape(b, s, _width + _mlpDim), DType.F32);
        Hunyuan3DDitOps.ConcatLastDim(cat, attn, mlp, b, s, _width, _mlpDim); attn.Dispose(); mlp.Dispose();
        Tensor outp = new(new TensorShape(b, s, _width), DType.F32); backend.Linear(outp, cat, _lin2W!, _lin2B!); cat.Dispose();

        Tensor res = AdaLNModulation.ApplyGatedResidual(x, outp, m[2], b, s, _width); outp.Dispose();
        foreach (Tensor t in m) t.Dispose();
        return res;
    }

    private static Tensor F(IReadOnlyDictionary<string, Tensor> w, string k) => Hunyuan3DDit.F32(w[k]);
}

/// <summary>Host-side tensor glue for the Hunyuan3D Flux DiT (layernorm-no-affine, head reshape, joint concat/split,
/// fused-QKV split, GELU-tanh, final shift-first modulation). Correctness-first; a GPU-resident rewrite is a perf
/// follow-up. NOTE: never feed a <c>Reshape</c> of a CUDA op's output to the next op — see the activation-cache
/// identity gotcha; these helpers all take/return distinct tensors and read via lazy-synced <c>DataPointer</c>.</summary>
internal static unsafe class Hunyuan3DDitOps
{
    public static void LayerNormNoAffine(Tensor o, Tensor x, int b, int s, int dim)
    {
        float* ip = (float*)x.DataPointer; float* op = (float*)o.DataPointer;
        for (long r = 0; r < (long)b * s; r++)
        {
            float* row = ip + r * dim; float* orow = op + r * dim;
            float mean = 0f; for (int d = 0; d < dim; d++) mean += row[d]; mean /= dim;
            float var = 0f; for (int d = 0; d < dim; d++) { float df = row[d] - mean; var += df * df; } var /= dim;
            float inv = 1f / MathF.Sqrt(var + 1e-6f);
            for (int d = 0; d < dim; d++) orow[d] = (row[d] - mean) * inv;
        }
    }

    /// <summary>In-place <c>x = (1+scale)·x + shift</c> where mod packs [shift(0), scale(1)] as [B, 2·dim].</summary>
    public static void ModulateShiftFirst(Tensor x, Tensor mod, int b, int s, int dim)
    {
        float* xp = (float*)x.DataPointer; float* mp = (float*)mod.DataPointer;
        for (int bb = 0; bb < b; bb++)
        {
            float* shift = mp + (long)bb * 2 * dim; float* scale = shift + dim;
            for (int ss = 0; ss < s; ss++)
            {
                float* row = xp + ((long)bb * s + ss) * dim;
                for (int d = 0; d < dim; d++) row[d] = row[d] * (1f + scale[d]) + shift[d];
            }
        }
    }

    /// <summary>Splits a fused QKV tensor [B,S,3W] into q,k,v each [B,S,W] (q=first W, k=second, v=third).</summary>
    public static void SplitQkv(Tensor qkv, Tensor q, Tensor k, Tensor v, int b, int s, int w)
    {
        float* sp = (float*)qkv.DataPointer; float* qp = (float*)q.DataPointer; float* kp = (float*)k.DataPointer; float* vp = (float*)v.DataPointer;
        long rows = (long)b * s;
        for (long r = 0; r < rows; r++)
        {
            float* src = sp + r * 3 * w;
            new ReadOnlySpan<float>(src, w).CopyTo(new Span<float>(qp + r * w, w));
            new ReadOnlySpan<float>(src + w, w).CopyTo(new Span<float>(kp + r * w, w));
            new ReadOnlySpan<float>(src + 2 * w, w).CopyTo(new Span<float>(vp + r * w, w));
        }
    }

    /// <summary>Splits [B,S,a+bb] into first [B,S,a] and second [B,S,bb] along the last dim.</summary>
    public static void SplitLastDim(Tensor x, Tensor first, Tensor second, int b, int s, int a, int bb)
    {
        float* sp = (float*)x.DataPointer; float* fp = (float*)first.DataPointer; float* sp2 = (float*)second.DataPointer;
        long rows = (long)b * s;
        for (long r = 0; r < rows; r++)
        {
            float* src = sp + r * (a + bb);
            new ReadOnlySpan<float>(src, a).CopyTo(new Span<float>(fp + r * a, a));
            new ReadOnlySpan<float>(src + a, bb).CopyTo(new Span<float>(sp2 + r * bb, bb));
        }
    }

    /// <summary>Concatenates [B,S,a] and [B,S,bb] along the last dim → [B,S,a+bb].</summary>
    public static void ConcatLastDim(Tensor o, Tensor first, Tensor second, int b, int s, int a, int bb)
    {
        float* op = (float*)o.DataPointer; float* fp = (float*)first.DataPointer; float* sp = (float*)second.DataPointer;
        long rows = (long)b * s;
        for (long r = 0; r < rows; r++)
        {
            float* dst = op + r * (a + bb);
            new ReadOnlySpan<float>(fp + r * a, a).CopyTo(new Span<float>(dst, a));
            new ReadOnlySpan<float>(sp + r * bb, bb).CopyTo(new Span<float>(dst + a, bb));
        }
    }

    public static void GeluTanhInPlace(Tensor t)
    {
        float* p = (float*)t.DataPointer; long n = t.ElementCount;
        const float c = 0.7978845608028654f; // sqrt(2/pi)
        for (long i = 0; i < n; i++) { float x = p[i]; p[i] = 0.5f * x * (1f + MathF.Tanh(c * (x + 0.044715f * x * x * x))); }
    }

    public static void ToHeads(Tensor o, Tensor x, int b, int s, int h, int d)
    {
        float* sp = (float*)x.DataPointer; float* dp = (float*)o.DataPointer; int w = h * d;
        for (int bb = 0; bb < b; bb++) for (int ss = 0; ss < s; ss++) for (int hh = 0; hh < h; hh++)
            new ReadOnlySpan<float>(sp + ((long)bb * s + ss) * w + hh * d, d).CopyTo(new Span<float>(dp + (((long)bb * h + hh) * s + ss) * d, d));
    }

    public static void FromHeads(Tensor o, Tensor x, int b, int s, int h, int d)
    {
        float* sp = (float*)x.DataPointer; float* dp = (float*)o.DataPointer; int w = h * d;
        for (int bb = 0; bb < b; bb++) for (int ss = 0; ss < s; ss++) for (int hh = 0; hh < h; hh++)
            new ReadOnlySpan<float>(sp + (((long)bb * h + hh) * s + ss) * d, d).CopyTo(new Span<float>(dp + ((long)bb * s + ss) * w + hh * d, d));
    }

    /// <summary>Concatenates two multi-head tensors [B,H,S1,D] + [B,H,S2,D] along the seq dim → [B,H,S1+S2,D].</summary>
    public static void ConcatSeqMh(Tensor o, Tensor first, Tensor second, int b, int h, int s1, int s2, int d)
    {
        float* fp = (float*)first.DataPointer; float* sp = (float*)second.DataPointer; float* op = (float*)o.DataPointer;
        int tot = s1 + s2, seg1 = s1 * d, seg2 = s2 * d;
        for (int bb = 0; bb < b; bb++) for (int hh = 0; hh < h; hh++)
        {
            long ob = (((long)bb * h + hh) * tot) * d;
            new ReadOnlySpan<float>(fp + (((long)bb * h + hh) * s1) * d, seg1).CopyTo(new Span<float>(op + ob, seg1));
            new ReadOnlySpan<float>(sp + (((long)bb * h + hh) * s2) * d, seg2).CopyTo(new Span<float>(op + ob + seg1, seg2));
        }
    }

    public static void SplitSeqMh(Tensor first, Tensor second, Tensor x, int b, int h, int s1, int s2, int d)
    {
        float* xp = (float*)x.DataPointer; float* fp = (float*)first.DataPointer; float* sp = (float*)second.DataPointer;
        int tot = s1 + s2, seg1 = s1 * d, seg2 = s2 * d;
        for (int bb = 0; bb < b; bb++) for (int hh = 0; hh < h; hh++)
        {
            long ib = (((long)bb * h + hh) * tot) * d;
            new ReadOnlySpan<float>(xp + ib, seg1).CopyTo(new Span<float>(fp + (((long)bb * h + hh) * s1) * d, seg1));
            new ReadOnlySpan<float>(xp + ib + seg1, seg2).CopyTo(new Span<float>(sp + (((long)bb * h + hh) * s2) * d, seg2));
        }
    }

    /// <summary>Concatenates two [B,S,W] tensors along seq → [B, S1+S2, W].</summary>
    public static Tensor ConcatSeq(Tensor first, Tensor second, int b, int s1, int s2, int w)
    {
        Tensor o = new(new TensorShape(b, s1 + s2, w), DType.F32);
        float* fp = (float*)first.DataPointer; float* sp = (float*)second.DataPointer; float* op = (float*)o.DataPointer;
        int seg1 = s1 * w, seg2 = s2 * w;
        for (int bb = 0; bb < b; bb++)
        {
            long ob = (long)bb * (s1 + s2) * w;
            new ReadOnlySpan<float>(fp + (long)bb * s1 * w, seg1).CopyTo(new Span<float>(op + ob, seg1));
            new ReadOnlySpan<float>(sp + (long)bb * s2 * w, seg2).CopyTo(new Span<float>(op + ob + seg1, seg2));
        }
        return o;
    }

    /// <summary>Drops the first <paramref name="skip"/> seq tokens: [B, skip+n, W] → [B, n, W].</summary>
    public static Tensor SliceSeq(Tensor x, int b, int skip, int n, int w)
    {
        Tensor o = new(new TensorShape(b, n, w), DType.F32);
        float* xp = (float*)x.DataPointer; float* op = (float*)o.DataPointer;
        int seg = n * w;
        for (int bb = 0; bb < b; bb++)
            new ReadOnlySpan<float>(xp + ((long)bb * (skip + n) + skip) * w, seg).CopyTo(new Span<float>(op + (long)bb * n * w, seg));
        return o;
    }
}

using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Trellis;

/// <summary>TRELLIS <c>ModulatedTransformerCrossBlock</c>: modulated self-attn (adaLN scale/shift + gate, per-head
/// QK-RMSNorm) → cross-attn to cond (affine norm2, ungated) → modulated tanh-GELU MLP. adaLN chunks 6 as
/// shift/scale/gate ×{msa,mlp}. norm1/norm3 non-affine (eps 1e-6), norm2 affine. All F32.</summary>
internal sealed unsafe class SsFlowBlock
{
    private const int W = 1024, H = 16, Hd = 64, Mlp = 4096;
    private static readonly float Scale = 1f / MathF.Sqrt(Hd);

    private Tensor _modW = null!, _modB = null!;
    private Tensor _n2W = null!, _n2B = null!;
    private Tensor _qkvW = null!, _qkvB = null!, _oW = null!, _oB = null!, _qGamma = null!, _kGamma = null!;
    private Tensor _cqW = null!, _cqB = null!, _ckvW = null!, _ckvB = null!, _coW = null!, _coB = null!;
    private Tensor _m0W = null!, _m0B = null!, _m2W = null!, _m2B = null!;

    public static SsFlowBlock Load(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        SsFlowBlock b = new();
        Tensor F(string k) => SparseStructureFlow.F(w, $"{p}.{k}");
        // Projection GEMM weights kept NATIVE F16 (checkpoint dtype, no upcast): with an F32 activation
        // ResolveGemmDtype→F16 → an F16 tensor-core GEMM with F32 accumulation (~2× vs the TF32 F32 GEMM), half the
        // weight VRAM, and no per-call weight cast. Norm/affine params (norm2, qk-gamma) stay F32 — F16-affine op
        // support is uneven and those are cheap. Parity gated by TrellisStage1/Stage2ParityTests.
        Tensor Fw(string k) => w[$"{p}.{k}"];
        b._modW = Fw("adaLN_modulation.1.weight"); b._modB = F("adaLN_modulation.1.bias");
        b._n2W = F("norm2.weight"); b._n2B = F("norm2.bias");
        b._qkvW = Fw("self_attn.to_qkv.weight"); b._qkvB = F("self_attn.to_qkv.bias");
        b._oW = Fw("self_attn.to_out.weight"); b._oB = F("self_attn.to_out.bias");
        b._qGamma = F("self_attn.q_rms_norm.gamma"); b._kGamma = F("self_attn.k_rms_norm.gamma");
        b._cqW = Fw("cross_attn.to_q.weight"); b._cqB = F("cross_attn.to_q.bias");
        b._ckvW = Fw("cross_attn.to_kv.weight"); b._ckvB = F("cross_attn.to_kv.bias");
        b._coW = Fw("cross_attn.to_out.weight"); b._coB = F("cross_attn.to_out.bias");
        b._m0W = Fw("mlp.mlp.0.weight"); b._m0B = F("mlp.mlp.0.bias");
        b._m2W = Fw("mlp.mlp.2.weight"); b._m2B = F("mlp.mlp.2.bias");
        return b;
    }

    public IEnumerable<Tensor> Weights() => new[] { _modW, _modB, _n2W, _n2B, _qkvW, _qkvB, _oW, _oB, _qGamma, _kGamma,
        _cqW, _cqB, _ckvW, _ckvB, _coW, _coB, _m0W, _m0B, _m2W, _m2B };

    public Tensor Forward(IBackend backend, Tensor x, Tensor vec, Tensor cond, Tensor ones64)
    {
        int n = (int)x.Shape[1], lc = (int)cond.Shape[1];
        TensorShape flat = new(1, n, W), heads = new(1, n, H, Hd), mh = new(1, H, n, Hd);

        // adaLN: SiLU(vec) → Linear → [1,6144]; slice 6 [1,W] params.
        Tensor sil = new(vec.Shape, DType.F32); backend.Silu(sil, vec);
        Tensor mod = new(new TensorShape(1, 6 * W), DType.F32); backend.Linear(mod, sil, _modW, _modB); sil.Dispose();
        Tensor[] m = new Tensor[6];
        for (int i = 0; i < 6; i++) { m[i] = new(new TensorShape(1, W), DType.F32); backend.SliceLastDim(m[i], mod, i * W); }
        mod.Dispose();   // m: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

        // ── self-attention ──
        Tensor h = new(flat, DType.F32); backend.LayerNormModulate(h, x, m[1], m[0], 1e-6f);
        Tensor qkv = new(new TensorShape(1, n, 3 * W), DType.F32); backend.Linear(qkv, h, _qkvW, _qkvB); h.Dispose();
        Tensor q = Head(backend, qkv, 0, n), k = Head(backend, qkv, W, n), v = Head(backend, qkv, 2 * W, n); qkv.Dispose();
        Tensor qP = PermQkNorm(backend, q, n, _qGamma, ones64), kP = PermQkNorm(backend, k, n, _kGamma, ones64);
        Tensor vP = new(mh, DType.F32); backend.Permute0213(vP, v, n, H, Hd); v.Dispose();
        Tensor attn = new(mh, DType.F32); backend.ScaledDotProductAttention(attn, qP, kP, vP, null, Scale, allowF16: true);
        qP.Dispose(); kP.Dispose(); vP.Dispose();
        Tensor attnFlat = new(flat, DType.F32); backend.Permute0213(attnFlat, attn, H, n, Hd); attn.Dispose();
        Tensor sOut = new(flat, DType.F32); backend.Linear(sOut, attnFlat, _oW, _oB); attnFlat.Dispose();
        Tensor x1 = new(flat, DType.F32); backend.GatedResidualLastDim(x1, x, sOut, m[2]); sOut.Dispose(); x.Dispose();

        // ── cross-attention (affine norm2, ungated) ──
        Tensor hc = new(flat, DType.F32); backend.LayerNorm(hc, x1, _n2W, _n2B, 1e-6f);
        Tensor cq = new(new TensorShape(1, n, W), DType.F32); backend.Linear(cq, hc, _cqW, _cqB); hc.Dispose();
        // Feed cq DIRECTLY (not cq.Reshape(heads)): [1,n,W] and [1,n,H,Hd] share memory and Permute0213 uses the
        // explicit n,H,Hd args — a Reshape view is a fresh Tensor identity that cache-misses the activation and forces
        // a full D2H+H2D round-trip of the 16 MB cq every block (was the dominant TRELLIS H2D_MISS_BIG cost).
        Tensor cqP = new(mh, DType.F32); backend.Permute0213(cqP, cq, n, H, Hd); cq.Dispose();
        Tensor kv = new(new TensorShape(1, lc, 2 * W), DType.F32); backend.Linear(kv, cond, _ckvW, _ckvB);
        Tensor ck = Head(backend, kv, 0, lc), cv = Head(backend, kv, W, lc); kv.Dispose();
        TensorShape kvMh = new(1, H, lc, Hd);
        Tensor ckP = new(kvMh, DType.F32); backend.Permute0213(ckP, ck, lc, H, Hd); ck.Dispose();
        Tensor cvP = new(kvMh, DType.F32); backend.Permute0213(cvP, cv, lc, H, Hd); cv.Dispose();
        Tensor cAttn = new(mh, DType.F32); backend.ScaledDotProductAttention(cAttn, cqP, ckP, cvP, null, Scale, allowF16: true);
        cqP.Dispose(); ckP.Dispose(); cvP.Dispose();
        Tensor cAttnFlat = new(flat, DType.F32); backend.Permute0213(cAttnFlat, cAttn, H, n, Hd); cAttn.Dispose();
        Tensor cOut = new(flat, DType.F32); backend.Linear(cOut, cAttnFlat, _coW, _coB); cAttnFlat.Dispose();
        Tensor x2 = new(flat, DType.F32); backend.Add(x2, x1, cOut); x1.Dispose(); cOut.Dispose();

        // ── MLP ──
        Tensor hm = new(flat, DType.F32); backend.LayerNormModulate(hm, x2, m[4], m[3], 1e-6f);
        Tensor f1 = new(new TensorShape(1, n, Mlp), DType.F32); backend.Linear(f1, hm, _m0W, _m0B); hm.Dispose();
        Tensor act = new(f1.Shape, DType.F32); backend.Gelu(act, f1); f1.Dispose();
        Tensor f2 = new(flat, DType.F32); backend.Linear(f2, act, _m2W, _m2B); act.Dispose();
        Tensor x3 = new(flat, DType.F32); backend.GatedResidualLastDim(x3, x2, f2, m[5]); x2.Dispose(); f2.Dispose();

        foreach (Tensor t in m) t.Dispose();
        return x3;
    }

    private static Tensor Head(IBackend backend, Tensor packed, int offset, int rows)
    {
        Tensor o = new(new TensorShape(1, rows, H, Hd), DType.F32); backend.SliceLastDim(o, packed, offset); return o;
    }

    /// <summary>Permute [1,n,H,Hd]→[1,H,n,Hd] then per-head QK-RMSNorm: rmsnorm(·)·gamma (== normalize·√Hd·gamma).</summary>
    private static Tensor PermQkNorm(IBackend backend, Tensor x, int n, Tensor gamma, Tensor ones64)
    {
        TensorShape mh = new(1, H, n, Hd);
        Tensor p = new(mh, DType.F32); backend.Permute0213(p, x, n, H, Hd); x.Dispose();
        Tensor normed = new(mh, DType.F32); backend.RmsNorm(normed, p, ones64, 0f); p.Dispose();
        Tensor o = new(mh, DType.F32); backend.AffineBroadcastLastDim(o, normed, gamma, null); normed.Dispose();
        return o;
    }
}

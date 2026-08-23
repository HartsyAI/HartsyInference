using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>LTX-Video DiT block (<c>LTXVideoTransformerBlock</c>), ported from diffusers. Operates on a single sample (B=1) as <c>[S, dim]</c>: self-attention with 3D RoPE → cross-attention to T5 → gelu-approx FFN, all gated by AdaLN-Single (a per-block <c>scale_shift_table[6,dim]</c> added to the shared timestep embedding). Pre-norms are RMSNorm-no-affine; QK-norm is RMSNorm-across-heads (affine). Reuses backend <c>RmsNorm</c>/<c>ScaledDotProductAttention</c> + <see cref="LtxRope"/>.</summary>
public sealed unsafe class LtxVideoBlock
{
    private readonly int _dim;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _crossDim;
    private readonly float _normEps;
    private readonly float _qkEps;
    private readonly Tensor _onesDim;   // unit weight for the no-affine RMS pre-norms

    private Tensor? _scaleShift;         // [6, dim]
    // attn1 (self) + attn2 (cross): to_q/k/v(+bias), to_out(+bias), norm_q/norm_k
    private Tensor?[] _q = new Tensor?[2], _qB = new Tensor?[2], _k = new Tensor?[2], _kB = new Tensor?[2],
        _v = new Tensor?[2], _vB = new Tensor?[2], _o = new Tensor?[2], _oB = new Tensor?[2], _nq = new Tensor?[2], _nk = new Tensor?[2];
    private Tensor? _ffProjW, _ffProjB, _ffOutW, _ffOutB;

    public LtxVideoBlock(LtxVideoConfig c)
    {
        _dim = c.InnerDim;
        _heads = c.NumHeads;
        _headDim = c.HeadDim;
        _crossDim = c.CrossAttentionDim;
        _normEps = c.NormEps;
        _qkEps = c.QkNormEps;
        _onesDim = Ones(_dim);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _scaleShift = TensorCasts.LoadF32(w, $"{prefix}.scale_shift_table");
        LoadAttn(w, $"{prefix}.attn1", 0);
        LoadAttn(w, $"{prefix}.attn2", 1);
        _ffProjW = LoadF16(w, $"{prefix}.ff.net.0.proj.weight");
        w.TryGetValue($"{prefix}.ff.net.0.proj.bias", out _ffProjB);
        _ffOutW = LoadF16(w, $"{prefix}.ff.net.2.weight");
        w.TryGetValue($"{prefix}.ff.net.2.bias", out _ffOutB);
    }

    private void LoadAttn(IReadOnlyDictionary<string, Tensor> w, string p, int i)
    {
        _q[i] = LoadF16(w, $"{p}.to_q.weight"); w.TryGetValue($"{p}.to_q.bias", out Tensor? qb); _qB[i] = qb;
        _k[i] = LoadF16(w, $"{p}.to_k.weight"); w.TryGetValue($"{p}.to_k.bias", out Tensor? kb); _kB[i] = kb;
        _v[i] = LoadF16(w, $"{p}.to_v.weight"); w.TryGetValue($"{p}.to_v.bias", out Tensor? vb); _vB[i] = vb;
        _o[i] = LoadF16(w, $"{p}.to_out.0.weight"); w.TryGetValue($"{p}.to_out.0.bias", out Tensor? ob); _oB[i] = ob;
        _nq[i] = TensorCasts.LoadF32(w, $"{p}.norm_q.weight");
        _nk[i] = TensorCasts.LoadF32(w, $"{p}.norm_k.weight");
    }

    // The big 2D Linear weights load as F16 so the DiT GEMMs use F16 tensor cores (~2× the TF32 path LTX-0.9's native
    // F32 weights would otherwise take). ResolveGemmDtype(F32 activation, F16 weight) → F16: the activation is auto-cast
    // to F16 per GEMM and the F32 bias is added to the F32 output afterward. LTX is a GELU-FFN DiT (like Wan) so F16 is
    // safe (no SwiGLU overflow) and preferable to BF16 (finer mantissa over a deep DiT). Norms/scale-shift stay F32.
    private static Tensor LoadF16(IReadOnlyDictionary<string, Tensor> w, string key)
    {
        Tensor t = w[key];
        return t.DType == DType.F16 ? t : t.CastTo(DType.F16);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_scaleShift is not null) yield return _scaleShift;
        for (int i = 0; i < 2; i++)
            foreach (Tensor? t in new[] { _q[i], _qB[i], _k[i], _kB[i], _v[i], _vB[i], _o[i], _oB[i], _nq[i], _nk[i] })
                if (t is not null) yield return t;
        foreach (Tensor? t in new[] { _ffProjW, _ffProjB, _ffOutW, _ffOutB })
            if (t is not null) yield return t;
    }

    /// <summary>Forward over <c>[S, dim]</c>. <paramref name="temb"/> is the shared timestep embedding <c>[6, dim]</c> (the block adds its own <c>scale_shift_table</c>). <paramref name="encoder"/> is the projected T5 <c>[L, dim]</c>; <paramref name="encoderMask"/> is an optional additive cross-attn mask <c>[1,1,S,L]</c>/<c>[1,1,1,L]</c> (null = full).
    /// <para>I2V conditioning (Tier 3.4): when <paramref name="temb0"/> (the SAME <c>[6,dim]</c> embedding built at
    /// timestep 0) and <paramref name="modIndex"/> (I32 <c>[S]</c>, 0 = conditioned/frame-0 token, 1 = denoising
    /// token) are both supplied, AdaLN modulation is looked up per-token from a 2-row table instead of broadcasting
    /// one shared vector — mirrors diffusers' per-token <c>embedded_timestep</c> (built from
    /// <c>timestep.unsqueeze(-1) * (1 - conditioning_mask)</c>) using the SAME row-indexed-table primitive already
    /// proven by <c>MiniMaxH3Transformer</c>'s multi-segment/multi-timestep modulation — not a new mechanism.</para></summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor encoder, Tensor temb, LtxRope rope, Tensor cos, Tensor sin, Tensor? encoderMask,
        Tensor? temb0 = null, Tensor? modIndex = null)
    {
        int s = (int)hidden.Shape[0];
        bool perToken = temb0 is not null && modIndex is not null;
        // AdaLN: scale_shift_table[6,dim] + temb[6,dim] → 6 vectors [dim] (or, per-token, 6 tables [2,dim]).
        (Tensor shiftMsa, Tensor scaleMsa, Tensor gateMsa, Tensor shiftMlp, Tensor scaleMlp, Tensor gateMlp) =
            perToken ? Modulation2Row(backend, temb, temb0!) : Modulation(backend, temb);

        // ── self-attn ──
        Tensor n1 = perToken
            ? ApplyShiftScaleRowIndexed(backend, RmsNoAffine(backend, hidden, s), scaleMsa, shiftMsa, modIndex!, s)
            : ApplyShiftScale(backend, RmsNoAffine(backend, hidden, s), scaleMsa, shiftMsa, s);
        Tensor attn1 = Attention(backend, n1, n1, 0, applyRope: true, rope, cos, sin, null, s, s);
        n1.Dispose();
        Tensor afterAttn1 = perToken
            ? GatedAddRowIndexed(backend, hidden, attn1, gateMsa, modIndex!, s)
            : GatedAdd(backend, hidden, attn1, gateMsa, s);
        attn1.Dispose();

        // ── cross-attn (to T5) ──
        int l = (int)encoder.Shape[0];
        Tensor attn2 = Attention(backend, afterAttn1, encoder, 1, applyRope: false, rope, cos, sin, encoderMask, s, l);
        Tensor afterAttn2 = AddRows(backend, afterAttn1, attn2, s);
        afterAttn1.Dispose();
        attn2.Dispose();

        // ── FFN ──
        Tensor n2 = perToken
            ? ApplyShiftScaleRowIndexed(backend, RmsNoAffine(backend, afterAttn2, s), scaleMlp, shiftMlp, modIndex!, s)
            : ApplyShiftScale(backend, RmsNoAffine(backend, afterAttn2, s), scaleMlp, shiftMlp, s);
        Tensor ff = Ffn(backend, n2, s);
        n2.Dispose();
        Tensor outT = perToken
            ? GatedAddRowIndexed(backend, afterAttn2, ff, gateMlp, modIndex!, s)
            : GatedAdd(backend, afterAttn2, ff, gateMlp, s);
        afterAttn2.Dispose();
        ff.Dispose();

        foreach (Tensor t in new[] { shiftMsa, scaleMsa, gateMsa, shiftMlp, scaleMlp, gateMlp }) t.Dispose();
        return outT;
    }

    private Tensor Attention(IBackend backend, Tensor qInput, Tensor kvInput, int idx, bool applyRope,
        LtxRope rope, Tensor cos, Tensor sin, Tensor? mask, int sq, int sk)
    {
        Tensor q = new Tensor(new TensorShape(sq, _dim), DType.F32);
        backend.Linear(q, qInput, _q[idx]!, _qB[idx]);
        Tensor k = new Tensor(new TensorShape(sk, _dim), DType.F32);
        backend.Linear(k, kvInput, _k[idx]!, _kB[idx]);
        Tensor v = new Tensor(new TensorShape(sk, _dim), DType.F32);
        backend.Linear(v, kvInput, _v[idx]!, _vB[idx]);

        // RMS-norm across heads (full dim), affine.
        Tensor qn = new Tensor(q.Shape, DType.F32); backend.RmsNorm(qn, q, _nq[idx]!, _qkEps); q.Dispose();
        Tensor kn = new Tensor(k.Shape, DType.F32); backend.RmsNorm(kn, k, _nk[idx]!, _qkEps); k.Dispose();

        if (applyRope) { rope.ApplyRotary(backend, qn, cos, sin); rope.ApplyRotary(backend, kn, cos, sin); }

        Tensor qMh = ToBhsd(backend, qn, sq); qn.Dispose();
        Tensor kMh = ToBhsd(backend, kn, sk); kn.Dispose();
        Tensor vMh = ToBhsd(backend, v, sk); v.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attn = new Tensor(new TensorShape(1, _heads, sq, _headDim), DType.F32);
        // allowF16: Q/K are RMS-normed → bounded scores → F16 attention is safe (halves score-matrix traffic).
        backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, mask, scale, allowF16: true);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor flat = FromBhsd(backend, attn, sq); attn.Dispose();
        Tensor outT = new Tensor(new TensorShape(sq, _dim), DType.F32);
        backend.Linear(outT, flat, _o[idx]!, _oB[idx]);
        flat.Dispose();
        return outT;
    }

    private Tensor Ffn(IBackend backend, Tensor x, int s)
    {
        int inner = (int)_ffProjW!.Shape[0];
        Tensor proj = new Tensor(new TensorShape(s, inner), DType.F32);
        backend.Linear(proj, x, _ffProjW!, _ffProjB);
        Tensor act = new Tensor(proj.Shape, DType.F32);
        backend.Gelu(act, proj);   // gelu-approximate (tanh) — backend Gelu is the tanh approximation
        proj.Dispose();
        Tensor outT = new Tensor(new TensorShape(s, _dim), DType.F32);
        backend.Linear(outT, act, _ffOutW!, _ffOutB);
        act.Dispose();
        return outT;
    }

    // ada[m] = scale_shift_table[m] + temb[m], each [1, dim], GPU-resident (SliceRows + GatedResidual) so downstream
    // AddScalar/AffineBroadcast/GatedResidual uploads all HIT the activation cache — no per-block cache-miss SyncStream
    // (the fix that took Wan-1.3B 67→28s). Was a host DataPointer loop.
    private (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) Modulation(IBackend backend, Tensor temb)
    {
        Tensor ones = OnesRow();
        Tensor[] outs = new Tensor[6];
        for (int m = 0; m < 6; m++)
        {
            Tensor ssM = new Tensor(new TensorShape(1, _dim), DType.F32);
            backend.SliceRows(ssM, _scaleShift!, m);
            Tensor tbM = new Tensor(new TensorShape(1, _dim), DType.F32);
            backend.SliceRows(tbM, temb, m);
            Tensor o = new Tensor(new TensorShape(1, _dim), DType.F32);
            backend.GatedResidualLastDim(o, tbM, ssM, ones);   // tbM + 1·ssM
            ssM.Dispose(); tbM.Dispose();
            outs[m] = o;
        }
        return (outs[0], outs[1], outs[2], outs[3], outs[4], outs[5]);
    }

    // Per-token variant of Modulation: builds 6 x [2, dim] tables instead of 6 x [1, dim] vectors — row 0 = the
    // conditioned (t=0) embedding, row 1 = the denoising (t=t_cur) embedding, so AffineBroadcastRowIndexed/
    // GatedResidualRowIndexed can look up the right row per token via modIndex. Same GPU-resident SliceRows/
    // GatedResidualLastDim build as Modulation, plus one Concat per slot to stack the two rows.
    private (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) Modulation2Row(IBackend backend, Tensor temb, Tensor temb0)
    {
        Tensor ones = OnesRow();
        Tensor[] outs = new Tensor[6];
        for (int m = 0; m < 6; m++)
        {
            using Tensor ssM = new Tensor(new TensorShape(1, _dim), DType.F32);
            backend.SliceRows(ssM, _scaleShift!, m);
            using Tensor tb0M = new Tensor(new TensorShape(1, _dim), DType.F32);
            backend.SliceRows(tb0M, temb0, m);
            using Tensor tbM = new Tensor(new TensorShape(1, _dim), DType.F32);
            backend.SliceRows(tbM, temb, m);
            using Tensor row0 = new Tensor(new TensorShape(1, _dim), DType.F32);
            backend.GatedResidualLastDim(row0, tb0M, ssM, ones);
            using Tensor row1 = new Tensor(new TensorShape(1, _dim), DType.F32);
            backend.GatedResidualLastDim(row1, tbM, ssM, ones);
            Tensor table = new Tensor(new TensorShape(2, _dim), DType.F32);
            backend.Concat(table, [row0, row1], 0);
            outs[m] = table;
        }
        return (outs[0], outs[1], outs[2], outs[3], outs[4], outs[5]);
    }

    // out[r] = x[r]·(1+scaleTable[modIndex[r]]) + shiftTable[modIndex[r]] — per-token version of ApplyShiftScale.
    // scaleTable is the RAW (no +1) table; AffineBroadcastRowIndexed applies the +1 internally (see its own doc).
    private Tensor ApplyShiftScaleRowIndexed(IBackend backend, Tensor x, Tensor scaleTable, Tensor shiftTable, Tensor modIndex, int s)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
        backend.AffineBroadcastRowIndexed(o, x, scaleTable, shiftTable, modIndex);
        x.Dispose();
        return o;
    }

    // out[r] = residual[r] + gateTable[modIndex[r]]·value[r] — per-token version of GatedAdd.
    private Tensor GatedAddRowIndexed(IBackend backend, Tensor residual, Tensor value, Tensor gateTable, Tensor modIndex, int s)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
        backend.GatedResidualRowIndexed(o, residual, value, gateTable, modIndex);
        return o;
    }

    private Tensor RmsNoAffine(IBackend backend, Tensor x, int s)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
        backend.RmsNorm(o, x, _onesDim, _normEps);
        return o;
    }

    // out = x·(1+scale) + shift, scale/shift [1,dim] broadcast — GPU-resident (AddScalar + AffineBroadcast), non-in-place
    // (disposes x, returns the result). Was a host DataPointer loop.
    private Tensor ApplyShiftScale(IBackend backend, Tensor x, Tensor scale, Tensor shift, int s)
    {
        using Tensor scaleP1 = new Tensor(scale.Shape, DType.F32);
        backend.AddScalar(scaleP1, scale, 1.0f);
        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
        backend.AffineBroadcastLastDim(o, x, scaleP1, shift);
        x.Dispose();
        return o;
    }

    // out = residual + gate·value, gate [1,dim] broadcast — GPU-resident. Was a host DataPointer loop.
    private Tensor GatedAdd(IBackend backend, Tensor residual, Tensor value, Tensor gate, int s)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
        backend.GatedResidualLastDim(o, residual, value, gate);
        return o;
    }

    // out = a + b — GPU-resident via GatedResidualLastDim(a, b, ones). Was a host loop.
    private Tensor AddRows(IBackend backend, Tensor a, Tensor b, int s)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
        backend.GatedResidualLastDim(o, a, b, OnesRow());
        return o;
    }

    // [s, dim]=[s, heads, headDim] → [1, heads, s, headDim], GPU-resident via Permute0213 (was a host DataPointer loop).
    private Tensor ToBhsd(IBackend backend, Tensor x, int s)
    {
        Tensor o = new Tensor(new TensorShape(1, _heads, s, _headDim), DType.F32);
        backend.Permute0213(o, x, s, _heads, _headDim);
        return o;
    }

    // [1, heads, s, headDim] → [s, dim] (inverse of ToBhsd), GPU-resident via Permute0213.
    private Tensor FromBhsd(IBackend backend, Tensor x, int s)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
        backend.Permute0213(o, x, _heads, s, _headDim);
        return o;
    }

    // Cached [1, dim] ones for the "a + 1·b" GatedResidual add-idiom; device-promoted after first use.
    private Tensor? _onesRow;
    private Tensor OnesRow()
    {
        if (_onesRow is null)
        {
            _onesRow = new Tensor(new TensorShape(1, _dim), DType.F32);
            float* p = (float*)_onesRow.DataPointer;
            for (int i = 0; i < _dim; i++) p[i] = 1f;
        }
        return _onesRow;
    }

    private static Tensor Ones(int n)
    {
        Tensor t = new Tensor(new TensorShape(n), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < n; i++) p[i] = 1f;
        return t;
    }
}

using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Microsoft Lens dual-stream MMDiT block (<c>LensTransformerBlock</c>). Mirrors upstream's per-stream modulation: <c>mod1, mod2 = Linear(SiLU(temb)).chunk(2)</c>, then <c>_modulate(x, mod) = (x*(1+scale)+shift, gate)</c> with <c>mod.chunk(3) = (shift, scale, gate)</c> — net effect is the standard <c>(shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)</c> 6-output order produced by <see cref="AdaLNModulation"/> with <c>numParams=6</c>. Joint attention concats <c>[img, txt]</c> per stream, applies complex-polar RoPE separately before concat, runs joint SDPA, splits back. <b>Returns <c>(text, image)</c></b> (encoder first, image second — matches upstream's <c>return encoder_hidden_states, hidden_states</c> in <c>LensTransformerBlock.forward</c>). Stream norms are RMSNorm with learned scale (vs QwenImageBlock's LayerNormNoAffine); FFN is SwiGLU with <c>w1/w2/w3</c> naming (vs QwenImageBlock's GELU); QKV is bias=True (upstream <c>img_qkv</c>/<c>txt_qkv</c> is split into <c>to_q/k/v</c> at checkpoint conversion).</summary>
public sealed unsafe class LensTransformerBlock
{
    /// <summary>F16-range damping for the attention value stream. Lens RMS-norms Q and K but NOT V, and its
    /// undamped residual stream drives raw <c>|V|</c> past F16's 65504 from the third denoise step on — every
    /// F16-ingesting attention backend then produces INF for that one <c>(head, dim)</c> column, which softmax·V
    /// smears across every token, and the whole joint sequence goes NaN for the rest of the forward (the
    /// symptom was a solid-black image, since <see cref="HartsyInference.Diffusion.Utilities.ImagePostProcessor.TensorToRgbBytes"/>
    /// maps NaN to byte 0). The backend picks such a path on its own — SageAttention's INT8 flash kernel casts V
    /// to F16 in its transpose prologue, and it is default-on for no-mask MHA at <c>head_dim ∈ {64,128}</c> — so
    /// <c>allowF16: false</c> at the call site is not enough. Attention is exactly linear in V, so scaling V down
    /// and the attention output back up is an algebraic identity, and a power-of-two factor costs nothing to carry
    /// through it — the scaling shifts exponents only, with no mantissa drift of its own. 1/256 sizes off the
    /// measured worst case (1024², 34-token prompt): peak
    /// <c>|V| = 139264</c> in blocks 44-46, sampled against the diffusers reference over the whole 4-, 20- and
    /// 50-step Lens schedules, which the damp brings to 544 — two orders of magnitude of headroom, while the
    /// F16 subnormal floor stays ~7 decades below the values that carry any weight in a convex combination.</summary>
    private const float ValueF16Damp = 1.0f / 256.0f;

    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _mlpDim;
    private readonly float _streamNormEps;

    private readonly AdaLNModulation _imgModulation;
    private readonly AdaLNModulation _txtModulation;

    private Tensor? _imgNorm1Weight, _imgNorm2Weight;
    private Tensor? _txtNorm1Weight, _txtNorm2Weight;

    private Tensor? _toQWeight, _toQBias;
    private Tensor? _toKWeight, _toKBias;
    private Tensor? _toVWeight, _toVBias;
    private Tensor? _toOutWeight, _toOutBias;

    private Tensor? _addQWeight, _addQBias;
    private Tensor? _addKWeight, _addKBias;
    private Tensor? _addVWeight, _addVBias;
    private Tensor? _toAddOutWeight, _toAddOutBias;

    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;
    private readonly QkNorm _normAddedQ;
    private readonly QkNorm _normAddedK;

    private readonly SwiGluFfn _imgFfn;
    private readonly SwiGluFfn _txtFfn;

    /// <summary>Creates a Lens dual-stream block. <paramref name="hiddenSize"/> = 1536, <paramref name="numHeads"/> = 24, <paramref name="headDim"/> = 64, <paramref name="mlpDim"/> = 4096 (SwiGLU 8/3 ratio) for the released weights.</summary>
    public LensTransformerBlock(int hiddenSize, int numHeads, int headDim, int mlpDim,
        float qkNormEps = 1e-5f, float streamNormEps = 1e-6f)
    {
        if (numHeads * headDim != hiddenSize)
            throw new ArgumentException($"numHeads * headDim ({numHeads} * {headDim}) must equal hiddenSize ({hiddenSize}).");

        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = headDim;
        _mlpDim = mlpDim;
        _streamNormEps = streamNormEps;

        _imgModulation = new AdaLNModulation(hiddenSize, 6);
        _txtModulation = new AdaLNModulation(hiddenSize, 6);

        _normQ = new QkNorm(headDim, qkNormEps);
        _normK = new QkNorm(headDim, qkNormEps);
        _normAddedQ = new QkNorm(headDim, qkNormEps);
        _normAddedK = new QkNorm(headDim, qkNormEps);

        _imgFfn = new SwiGluFfn(hiddenSize, mlpDim);
        _txtFfn = new SwiGluFfn(hiddenSize, mlpDim);
    }

    /// <summary>Loads weights under <c>transformer_blocks.{i}.*</c>. Expects fused <c>img_qkv</c>/<c>txt_qkv</c> to have been pre-split into <c>to_q/to_k/to_v</c> and <c>add_q_proj/add_k_proj/add_v_proj</c> by <see cref="LensCheckpointConverter"/> (same pattern Sd3 uses). FFN is SwiGLU mode (no biases); stream norms are learned-scale RMSNorm.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _imgModulation.LoadWeights(
            weights[$"{prefix}.img_mod.1.weight"],
            weights.TryGetValue($"{prefix}.img_mod.1.bias", out Tensor? imgModBias) ? imgModBias : null);
        _txtModulation.LoadWeights(
            weights[$"{prefix}.txt_mod.1.weight"],
            weights.TryGetValue($"{prefix}.txt_mod.1.bias", out Tensor? txtModBias) ? txtModBias : null);

        _imgNorm1Weight = CastToF32IfNeeded(weights[$"{prefix}.img_norm1.weight"]);
        _imgNorm2Weight = CastToF32IfNeeded(weights[$"{prefix}.img_norm2.weight"]);
        _txtNorm1Weight = CastToF32IfNeeded(weights[$"{prefix}.txt_norm1.weight"]);
        _txtNorm2Weight = CastToF32IfNeeded(weights[$"{prefix}.txt_norm2.weight"]);

        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _toOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];
        _toOutBias = weights[$"{prefix}.attn.to_out.0.bias"];
        _toQBias = weights[$"{prefix}.attn.to_q.bias"];
        _toKBias = weights[$"{prefix}.attn.to_k.bias"];
        _toVBias = weights[$"{prefix}.attn.to_v.bias"];

        _addQWeight = weights[$"{prefix}.attn.add_q_proj.weight"];
        _addKWeight = weights[$"{prefix}.attn.add_k_proj.weight"];
        _addVWeight = weights[$"{prefix}.attn.add_v_proj.weight"];
        _toAddOutWeight = weights[$"{prefix}.attn.to_add_out.weight"];
        _toAddOutBias = weights[$"{prefix}.attn.to_add_out.bias"];
        _addQBias = weights[$"{prefix}.attn.add_q_proj.bias"];
        _addKBias = weights[$"{prefix}.attn.add_k_proj.bias"];
        _addVBias = weights[$"{prefix}.attn.add_v_proj.bias"];

        _normQ.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);
        _normAddedQ.LoadWeights(weights[$"{prefix}.attn.norm_added_q.weight"]);
        _normAddedK.LoadWeights(weights[$"{prefix}.attn.norm_added_k.weight"]);

        _imgFfn.LoadSwiGluWeights(
            weights[$"{prefix}.img_mlp.w1.weight"], null,
            weights[$"{prefix}.img_mlp.w3.weight"], null,
            weights[$"{prefix}.img_mlp.w2.weight"], null);

        _txtFfn.LoadSwiGluWeights(
            weights[$"{prefix}.txt_mlp.w1.weight"], null,
            weights[$"{prefix}.txt_mlp.w3.weight"], null,
            weights[$"{prefix}.txt_mlp.w2.weight"], null);
    }

    /// <summary>Yields every block weight for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in _imgModulation.EnumerateWeights()) yield return w;
        foreach (Tensor w in _txtModulation.EnumerateWeights()) yield return w;
        if (_imgNorm1Weight is not null) yield return _imgNorm1Weight;
        if (_imgNorm2Weight is not null) yield return _imgNorm2Weight;
        if (_txtNorm1Weight is not null) yield return _txtNorm1Weight;
        if (_txtNorm2Weight is not null) yield return _txtNorm2Weight;
        if (_toQWeight is not null) yield return _toQWeight;
        if (_toQBias is not null) yield return _toQBias;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toKBias is not null) yield return _toKBias;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toVBias is not null) yield return _toVBias;
        if (_toOutWeight is not null) yield return _toOutWeight;
        if (_toOutBias is not null) yield return _toOutBias;
        if (_addQWeight is not null) yield return _addQWeight;
        if (_addQBias is not null) yield return _addQBias;
        if (_addKWeight is not null) yield return _addKWeight;
        if (_addKBias is not null) yield return _addKBias;
        if (_addVWeight is not null) yield return _addVWeight;
        if (_addVBias is not null) yield return _addVBias;
        if (_toAddOutWeight is not null) yield return _toAddOutWeight;
        if (_toAddOutBias is not null) yield return _toAddOutBias;
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normAddedQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normAddedK.EnumerateWeights()) yield return w;
        foreach (Tensor w in _imgFfn.EnumerateWeights()) yield return w;
        foreach (Tensor w in _txtFfn.EnumerateWeights()) yield return w;
    }

    /// <summary>Forward pass. Returns <c>(text, image)</c> matching upstream's return order (encoder first, image second).</summary>
    // GPU-residency rewrite (mirrors the verified QwenImageBlock): every glue op (RMSNorm / AdaLN modulation /
    // QK-norm / reshape-to-heads / joint concat / split / gated residual / RoPE) runs as an IBackend GPU op so the
    // activation stays device-resident across the whole block — no per-op DataPointer reads / D2H sync barriers
    // (the old DiTUtils/QkNorm/AdaLNModulation/LensRope host path D2H-synced every multi-MB intermediate many times
    // per block × 48 blocks × steps, which was the ~25 s/step cost). RoPE runs as ONE device pass over the
    // GPU-concatenated joint [img, txt] sequence on the PRE-permute [B, S, H, D] layout (per-row independent, so
    // identical to the old per-stream post-permute host pass); the cos/sin tables are position-only and cached
    // across blocks/steps. Batch is always 1 in the pipeline (CFG runs as two batch-1 passes); the batch>1
    // fallback ropes via the host LensRope.ApplyJoint after the head permute.
    public (Tensor text, Tensor image) Forward(IBackend backend, Tensor image, Tensor text, Tensor temb,
        LensRope rope, int imgPackedH, int imgPackedW, int txtPositionStart)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];
        int totalSeqLen = imgSeqLen + txtSeqLen;
        float scale = 1.0f / MathF.Sqrt(_headDim);

        Tensor[] imgMod = _imgModulation.Forward(backend, temb);
        Tensor[] txtMod = _txtModulation.Forward(backend, temb);

        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);
        // [B, S, H, D] views (byte-identical to [B, S, hidden]) so QK-norm + permute need no reshape copy.
        TensorShape imgHeads = new TensorShape(batch, imgSeqLen, _numHeads, _headDim);
        TensorShape txtHeads = new TensorShape(batch, txtSeqLen, _numHeads, _headDim);
        TensorShape jointFlat = new TensorShape(batch, totalSeqLen, _hiddenSize);
        TensorShape jointMh = new TensorShape(batch, _numHeads, totalSeqLen, _headDim);

        // ── 1. Stream RMSNorm (learned scale) + AdaLN modulate: x*(1+scale)+shift ──
        Tensor imgNormed = new Tensor(imgShape, DType.F32);
        backend.RmsNorm(imgNormed, image, _imgNorm1Weight!, _streamNormEps);
        Tensor imgModulated = DiTUtils.Modulate(backend, imgNormed, imgMod[0], imgMod[1], imgShape);
        imgNormed.Dispose();

        Tensor txtNormed = new Tensor(txtShape, DType.F32);
        backend.RmsNorm(txtNormed, text, _txtNorm1Weight!, _streamNormEps);
        Tensor txtModulated = DiTUtils.Modulate(backend, txtNormed, txtMod[0], txtMod[1], txtShape);
        txtNormed.Dispose();

        // ── 2. Q/K/V projections (declared [B, S, H, D]) ──
        Tensor imgQ = new Tensor(imgHeads, DType.F32);
        backend.Linear(imgQ, imgModulated, _toQWeight!, _toQBias);
        Tensor imgK = new Tensor(imgHeads, DType.F32);
        backend.Linear(imgK, imgModulated, _toKWeight!, _toKBias);
        Tensor imgV = new Tensor(imgHeads, DType.F32);
        backend.Linear(imgV, imgModulated, _toVWeight!, _toVBias);
        imgModulated.Dispose();

        Tensor txtQ = new Tensor(txtHeads, DType.F32);
        backend.Linear(txtQ, txtModulated, _addQWeight!, _addQBias);
        Tensor txtK = new Tensor(txtHeads, DType.F32);
        backend.Linear(txtK, txtModulated, _addKWeight!, _addKBias);
        Tensor txtV = new Tensor(txtHeads, DType.F32);
        backend.Linear(txtV, txtModulated, _addVWeight!, _addVBias);
        txtModulated.Dispose();

        // ── 3. QK-norm (per-head RMSNorm over the last dim = headDim) ──
        Tensor imgQn = new Tensor(imgHeads, DType.F32);
        backend.RmsNorm(imgQn, imgQ, _normQ.Weight, _normQ.Eps);
        imgQ.Dispose();
        Tensor imgKn = new Tensor(imgHeads, DType.F32);
        backend.RmsNorm(imgKn, imgK, _normK.Weight, _normK.Eps);
        imgK.Dispose();
        Tensor txtQn = new Tensor(txtHeads, DType.F32);
        backend.RmsNorm(txtQn, txtQ, _normAddedQ.Weight, _normAddedQ.Eps);
        txtQ.Dispose();
        Tensor txtKn = new Tensor(txtHeads, DType.F32);
        backend.RmsNorm(txtKn, txtK, _normAddedK.Weight, _normAddedK.Eps);
        txtK.Dispose();

        // ── 4. Concat [img, txt] along the seq dim (upstream LensJointAttention order — image FIRST, the
        // opposite of Qwen-Image; the split after attention matches). Contiguous row-concat in [B,S,hidden]. ──
        Tensor jointQf = new Tensor(jointFlat, DType.F32);
        backend.Concat(jointQf, new Tensor[] { imgQn, txtQn }, 1);
        Tensor jointKf = new Tensor(jointFlat, DType.F32);
        backend.Concat(jointKf, new Tensor[] { imgKn, txtKn }, 1);
        Tensor jointVraw = new Tensor(jointFlat, DType.F32);
        backend.Concat(jointVraw, new Tensor[] { imgV, txtV }, 1);
        imgQn.Dispose(); txtQn.Dispose();
        imgKn.Dispose(); txtKn.Dispose();
        imgV.Dispose(); txtV.Dispose();

        // Damp V into F16 range before it reaches the backend's attention kernels — see ValueF16Damp.
        Tensor jointVf = new Tensor(jointFlat, DType.F32);
        backend.Scale(jointVf, jointVraw, ValueF16Damp);
        jointVraw.Dispose();

        // ── 5. RoPE on the joint [img, txt] Q,K — device kernel on the pre-permute [B, S, H, D] layout. ──
        if (batch == 1)
        {
            (Tensor ropeCos, Tensor ropeSin) = rope.GetOrBuildJointTables(
                imgPackedH, imgPackedW, txtSeqLen, txtPositionStart);
            backend.WanRopeInterleaved(jointQf, ropeCos, ropeSin, totalSeqLen, _numHeads, _headDim);
            backend.WanRopeInterleaved(jointKf, ropeCos, ropeSin, totalSeqLen, _numHeads, _headDim);
        }

        // ── 6. Permute [B, S, H, D] → [B, H, S, D] for SDPA ──
        Tensor jointQ = new Tensor(jointMh, DType.F32);
        backend.Permute0213(jointQ, jointQf, totalSeqLen, _numHeads, _headDim);
        jointQf.Dispose();
        Tensor jointK = new Tensor(jointMh, DType.F32);
        backend.Permute0213(jointK, jointKf, totalSeqLen, _numHeads, _headDim);
        jointKf.Dispose();
        Tensor jointV = new Tensor(jointMh, DType.F32);
        backend.Permute0213(jointV, jointVf, totalSeqLen, _numHeads, _headDim);
        jointVf.Dispose();

        if (batch != 1)
            rope.ApplyJoint(jointQ, jointK, batch, _numHeads, imgPackedH, imgPackedW, txtSeqLen, txtPositionStart);

        // ── 7. Joint SDPA (no mask) on the damped V; the output is un-damped below. allowF16 must stay OFF:
        // Q/K are RMS-normed (bounded scores) but V is NOT (probe-verified: block_36+ went full-NaN with the
        // fused F16 path even before the residual stream reached its late-step magnitudes). ──
        Tensor jointAttnDamped = new Tensor(jointMh, DType.F32);
        backend.ScaledDotProductAttention(jointAttnDamped, jointQ, jointK, jointV, null, scale);
        jointQ.Dispose();
        jointK.Dispose();
        jointV.Dispose();

        Tensor jointAttnOut = new Tensor(jointMh, DType.F32);
        backend.Scale(jointAttnOut, jointAttnDamped, 1.0f / ValueF16Damp);
        jointAttnDamped.Dispose();

        // ── 8. Permute back [B, H, S, D] → [B, S, hidden], then split [img, txt] (B=1: contiguous rows) ──
        Tensor jointAttnFlat = new Tensor(jointFlat, DType.F32);
        backend.Permute0213(jointAttnFlat, jointAttnOut, _numHeads, totalSeqLen, _headDim);
        jointAttnOut.Dispose();

        Tensor imgAttn = new Tensor(imgShape, DType.F32);
        backend.SliceRows(imgAttn, jointAttnFlat, 0);
        Tensor txtAttn = new Tensor(txtShape, DType.F32);
        backend.SliceRows(txtAttn, jointAttnFlat, imgSeqLen);
        jointAttnFlat.Dispose();

        // ── 9. Output projections + gated residual (input + gate*value) ──
        Tensor imgAttnProj = new Tensor(imgShape, DType.F32);
        backend.Linear(imgAttnProj, imgAttn, _toOutWeight!, _toOutBias);
        imgAttn.Dispose();
        Tensor imgAfterAttn = new Tensor(imgShape, DType.F32);
        backend.GatedResidualLastDim(imgAfterAttn, image, imgAttnProj, imgMod[2]);
        imgAttnProj.Dispose();

        Tensor txtAttnProj = new Tensor(txtShape, DType.F32);
        backend.Linear(txtAttnProj, txtAttn, _toAddOutWeight!, _toAddOutBias);
        txtAttn.Dispose();
        Tensor txtAfterAttn = new Tensor(txtShape, DType.F32);
        backend.GatedResidualLastDim(txtAfterAttn, text, txtAttnProj, txtMod[2]);
        txtAttnProj.Dispose();

        // ── 10. Image MLP path ──
        Tensor imgMlpNormed = new Tensor(imgShape, DType.F32);
        backend.RmsNorm(imgMlpNormed, imgAfterAttn, _imgNorm2Weight!, _streamNormEps);
        Tensor imgMlpModulated = DiTUtils.Modulate(backend, imgMlpNormed, imgMod[3], imgMod[4], imgShape);
        imgMlpNormed.Dispose();
        Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();
        Tensor imgFinal = new Tensor(imgShape, DType.F32);
        backend.GatedResidualLastDim(imgFinal, imgAfterAttn, imgMlpOut, imgMod[5]);
        imgMlpOut.Dispose();
        imgAfterAttn.Dispose();

        // ── 11. Text MLP path ──
        Tensor txtMlpNormed = new Tensor(txtShape, DType.F32);
        backend.RmsNorm(txtMlpNormed, txtAfterAttn, _txtNorm2Weight!, _streamNormEps);
        Tensor txtMlpModulated = DiTUtils.Modulate(backend, txtMlpNormed, txtMod[3], txtMod[4], txtShape);
        txtMlpNormed.Dispose();
        Tensor txtMlpOut = _txtFfn.Forward(backend, txtMlpModulated, batch, txtSeqLen);
        txtMlpModulated.Dispose();
        Tensor txtFinal = new Tensor(txtShape, DType.F32);
        backend.GatedResidualLastDim(txtFinal, txtAfterAttn, txtMlpOut, txtMod[5]);
        txtMlpOut.Dispose();
        txtAfterAttn.Dispose();

        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < txtMod.Length; i++) txtMod[i].Dispose();

        return (txtFinal, imgFinal);
    }

    private static Tensor CastToF32IfNeeded(Tensor t) =>
        t.DType == DType.F32 ? t : t.CastTo(DType.F32);
}

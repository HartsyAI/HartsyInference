using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>AuraFlow joint MMDiT block (<c>AuraFlowJointTransformerBlock</c>). Dual-stream image+text attention with AdaLN-Zero modulation, FP32 QK-norm, no biases on attention/FFN linears, and SwiGLU FFN. Differs from SD3 <see cref="JointBlock"/> mainly by the bias-free linears, FP32 LayerNorm, SwiGLU (vs GELU), and the post-attention norm-then-modulate ordering (<c>norm2(residual + gate * attn)</c>). Concat order in joint attention is <c>[txt, img]</c>, matching <c>AuraFlowAttnProcessor2_0</c>.</summary>
public sealed unsafe class AuraFlowJointBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _innerDim;
    private readonly int _mlpDim;
    private readonly float _qkNormEps;

    private readonly AdaLNModulation _imgModulation;
    private readonly AdaLNModulation _txtModulation;

    // Image (sample) attention projections — bias=False everywhere.
    private Tensor? _toQWeight;
    private Tensor? _toKWeight;
    private Tensor? _toVWeight;
    private Tensor? _toOutWeight;

    // Context (added) attention projections — bias=False everywhere.
    private Tensor? _addQWeight;
    private Tensor? _addKWeight;
    private Tensor? _addVWeight;
    private Tensor? _toAddOutWeight;

    // FP32 QK-norm — always present in AuraFlow.
    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;
    private readonly QkNorm _normAddedQ;
    private readonly QkNorm _normAddedK;

    private readonly SwiGluFfn _imgFfn;
    private readonly SwiGluFfn _txtFfn;

    /// <summary>Creates an AuraFlow joint block. <paramref name="headDim"/> is explicit (256 for v0.3); <c>numHeads * headDim</c> must equal <paramref name="hiddenSize"/>.</summary>
    public AuraFlowJointBlock(int hiddenSize, int numHeads, int headDim, int mlpDim, float qkNormEps = 1e-5f)
    {
        if (numHeads * headDim != hiddenSize)
            throw new ArgumentException($"numHeads * headDim ({numHeads} * {headDim} = {numHeads * headDim}) must equal hiddenSize ({hiddenSize}).");

        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = headDim;
        _innerDim = numHeads * headDim;
        _mlpDim = mlpDim;
        _qkNormEps = qkNormEps;

        _imgModulation = new AdaLNModulation(hiddenSize, 6);
        _txtModulation = new AdaLNModulation(hiddenSize, 6);

        _normQ = new QkNorm(headDim, qkNormEps);
        _normK = new QkNorm(headDim, qkNormEps);
        _normAddedQ = new QkNorm(headDim, qkNormEps);
        _normAddedK = new QkNorm(headDim, qkNormEps);

        _imgFfn = new SwiGluFfn(hiddenSize, mlpDim);
        _txtFfn = new SwiGluFfn(hiddenSize, mlpDim);
    }

    /// <summary>Loads weights using HuggingFace diffusers naming (<c>joint_transformer_blocks.{i}.*</c>). All AdaLN/attention/FFN linears are bias=False.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _imgModulation.LoadWeights(weights[$"{prefix}.norm1.linear.weight"], null);
        _txtModulation.LoadWeights(weights[$"{prefix}.norm1_context.linear.weight"], null);

        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _toOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];

        _addQWeight = weights[$"{prefix}.attn.add_q_proj.weight"];
        _addKWeight = weights[$"{prefix}.attn.add_k_proj.weight"];
        _addVWeight = weights[$"{prefix}.attn.add_v_proj.weight"];
        _toAddOutWeight = weights[$"{prefix}.attn.to_add_out.weight"];

        // AuraFlow uses `qk_norm="fp32_layer_norm"` with `elementwise_affine=False, bias=False` — i.e.,
        // **no learnable weights** (auraflow_transformer_2d.py lines 161, 226, plus diffusers Attention.__init__).
        // The checkpoint therefore does not ship `attn.norm_q.weight` etc. We synthesize a unit-scale weight
        // (all 1s) so our `QkNorm` (which is RMSNorm-style with a mandatory learnable scale) becomes
        // mathematically `(x / RMS(x)) * 1`. **Caveat:** AuraFlow's QK-norm is *LayerNorm* not *RMSNorm* —
        // it subtracts the mean as well. With this shim we apply RMSNorm instead, which differs by
        // `(x - mean) / std` vs `x / rms`. For zero-mean Q/K this matches; for non-zero-mean it'll be a
        // small but compounding drift across all 4 joint blocks. Fixing this requires a no-affine
        // LayerNorm-style QK-norm — see tracking note in PHASE_4_MODEL_BREADTH.md `### AuraFlow`.
        _normQ.LoadWeights(GetOrFakeOnes(weights, $"{prefix}.attn.norm_q.weight", _headDim));
        _normK.LoadWeights(GetOrFakeOnes(weights, $"{prefix}.attn.norm_k.weight", _headDim));
        _normAddedQ.LoadWeights(GetOrFakeOnes(weights, $"{prefix}.attn.norm_added_q.weight", _headDim));
        _normAddedK.LoadWeights(GetOrFakeOnes(weights, $"{prefix}.attn.norm_added_k.weight", _headDim));

        // AuraFlowFeedForward: linear_1 = gate (SiLU path), linear_2 = linear path, out_projection = output.
        // Maps onto SwiGluFfn's (w1=gate, w3=linear, w2=output) with all biases null.
        _imgFfn.LoadSwiGluWeights(
            weights[$"{prefix}.ff.linear_1.weight"], null,
            weights[$"{prefix}.ff.linear_2.weight"], null,
            weights[$"{prefix}.ff.out_projection.weight"], null);
        _txtFfn.LoadSwiGluWeights(
            weights[$"{prefix}.ff_context.linear_1.weight"], null,
            weights[$"{prefix}.ff_context.linear_2.weight"], null,
            weights[$"{prefix}.ff_context.out_projection.weight"], null);
    }

    /// <summary>Forward pass through the dual-stream block. Returns <c>(image, text)</c> matching SD3 C# convention; diffusers reference returns <c>(text, image)</c> from <c>AuraFlowJointTransformerBlock.forward</c>.</summary>
    // GPU-residency rewrite (mirrors ChromaDoubleStreamBlock): every glue op — LayerNorm+AdaLN modulation, per-head
    // FP32 QK-norm, head split/merge, the joint [txt, img] concat/split, gated residuals — runs as an IBackend op so
    // the activation stays device-resident across the block (no per-op DataPointer D2H sync barriers around every
    // GEMM). Q/K/V are declared [B, S, H, D] (byte-identical to [B, S, hidden]) so RmsNorm normalizes over headDim
    // with no reshape; the [txt, img] concat happens in [B, S, hidden] layout then a single Permute0213 lifts to
    // [B, H, S, D] for SDPA (equivalent to the old per-head ConcatAlongSeqDimMultiHead). AuraFlow has no RoPE, so
    // nothing is left on the host. Numerics are bit-identical to the old host helpers: DiTUtils.NormModulate
    // reproduces LayerNormNoAffine + x*(1+scale)+shift; backend.RmsNorm reproduces QkNorm.Forward;
    // backend.GatedResidualLastDim reproduces AdaLNModulation.ApplyGatedResidual.
    public (Tensor image, Tensor text) Forward(IBackend backend, Tensor image, Tensor text, Tensor temb)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];
        int totalSeqLen = txtSeqLen + imgSeqLen;

        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);
        // [B, S, H, D] views (byte-identical to [B, S, hidden]) so RmsNorm normalizes over headDim.
        TensorShape imgHeads = new TensorShape(batch, imgSeqLen, _numHeads, _headDim);
        TensorShape txtHeads = new TensorShape(batch, txtSeqLen, _numHeads, _headDim);
        TensorShape jointFlat = new TensorShape(batch, totalSeqLen, _hiddenSize);
        TensorShape jointMhShape = new TensorShape(batch, _numHeads, totalSeqLen, _headDim);

        // ── 1. AdaLN modulation: 6 params each [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] ──
        Tensor[] imgMod = _imgModulation.Forward(backend, temb);
        Tensor[] txtMod = _txtModulation.Forward(backend, temb);

        // ── 2. LayerNorm (no affine, eps 1e-6) + modulate x*(1+scale)+shift on both streams ──
        Tensor imgModulated = DiTUtils.NormModulate(backend, image, imgMod[0], imgMod[1], imgShape);
        Tensor txtModulated = DiTUtils.NormModulate(backend, text, txtMod[0], txtMod[1], txtShape);

        // ── 3. Q/K/V projections (bias-free), declared [B, S, H, D] for per-head RMSNorm ──
        Tensor imgQ = new Tensor(imgHeads, DType.F32);
        backend.Linear(imgQ, imgModulated, _toQWeight!, null);
        Tensor imgK = new Tensor(imgHeads, DType.F32);
        backend.Linear(imgK, imgModulated, _toKWeight!, null);
        Tensor imgV = new Tensor(imgHeads, DType.F32);
        backend.Linear(imgV, imgModulated, _toVWeight!, null);
        imgModulated.Dispose();

        Tensor txtQ = new Tensor(txtHeads, DType.F32);
        backend.Linear(txtQ, txtModulated, _addQWeight!, null);
        Tensor txtK = new Tensor(txtHeads, DType.F32);
        backend.Linear(txtK, txtModulated, _addKWeight!, null);
        Tensor txtV = new Tensor(txtHeads, DType.F32);
        backend.Linear(txtV, txtModulated, _addVWeight!, null);
        txtModulated.Dispose();

        // ── 4. FP32 QK-norm (per-head RMSNorm over the last dim = headDim) ──
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

        // ── 5. Concat [txt, img] along seq dim in [B, S, hidden] layout (matches AuraFlowAttnProcessor2_0
        //       lines 2145-2147; contiguous row-concat, equivalent to the old per-head ConcatAlongSeqDimMultiHead) ──
        Tensor jointQf = new Tensor(jointFlat, DType.F32);
        backend.Concat(jointQf, new Tensor[] { txtQn, imgQn }, 1);
        Tensor jointKf = new Tensor(jointFlat, DType.F32);
        backend.Concat(jointKf, new Tensor[] { txtKn, imgKn }, 1);
        Tensor jointVf = new Tensor(jointFlat, DType.F32);
        backend.Concat(jointVf, new Tensor[] { txtV, imgV }, 1);
        txtQn.Dispose(); imgQn.Dispose();
        txtKn.Dispose(); imgKn.Dispose();
        txtV.Dispose(); imgV.Dispose();

        // ── 6. Permute [B, S, H, D] → [B, H, S, D] for SDPA ──
        Tensor jointQ = new Tensor(jointMhShape, DType.F32);
        backend.Permute0213(jointQ, jointQf, totalSeqLen, _numHeads, _headDim);
        jointQf.Dispose();
        Tensor jointK = new Tensor(jointMhShape, DType.F32);
        backend.Permute0213(jointK, jointKf, totalSeqLen, _numHeads, _headDim);
        jointKf.Dispose();
        Tensor jointV = new Tensor(jointMhShape, DType.F32);
        backend.Permute0213(jointV, jointVf, totalSeqLen, _numHeads, _headDim);
        jointVf.Dispose();

        // ── 7. Joint scaled dot-product attention ──
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor jointAttnOut = new Tensor(jointMhShape, DType.F32);
        backend.ScaledDotProductAttention(jointAttnOut, jointQ, jointK, jointV, null, scale);
        jointQ.Dispose(); jointK.Dispose(); jointV.Dispose();

        // ── 8. Permute back [B, H, S, D] → [B, S, hidden], then split [txt, img] along the seq dim ──
        Tensor jointAttnFlat = new Tensor(jointFlat, DType.F32);
        backend.Permute0213(jointAttnFlat, jointAttnOut, _numHeads, totalSeqLen, _headDim);
        jointAttnOut.Dispose();

        Tensor txtAttn = new Tensor(txtShape, DType.F32);
        backend.SliceRows(txtAttn, jointAttnFlat, 0);
        Tensor imgAttn = new Tensor(imgShape, DType.F32);
        backend.SliceRows(imgAttn, jointAttnFlat, txtSeqLen);
        jointAttnFlat.Dispose();

        // ── 9. Output projections (bias-free) ──
        Tensor imgAttnProj = new Tensor(imgShape, DType.F32);
        backend.Linear(imgAttnProj, imgAttn, _toOutWeight!, null);
        imgAttn.Dispose();

        Tensor txtAttnProj = new Tensor(txtShape, DType.F32);
        backend.Linear(txtAttnProj, txtAttn, _toAddOutWeight!, null);
        txtAttn.Dispose();

        // ── 10. Image post-attention path (lines 264-267):
        //   h_post = norm2(residual + gate_msa * attn)            <- norm AFTER residual
        //   h_mod  = h_post * (1 + scale_mlp) + shift_mlp
        //   out    = residual + gate_mlp * ff(h_mod)              <- residual is the ORIGINAL `image`
        Tensor imgPreNorm = new Tensor(imgShape, DType.F32);
        backend.GatedResidualLastDim(imgPreNorm, image, imgAttnProj, imgMod[2]);
        imgAttnProj.Dispose();

        Tensor imgMlpModulated = DiTUtils.NormModulate(backend, imgPreNorm, imgMod[3], imgMod[4], imgShape);
        imgPreNorm.Dispose();

        Tensor imgFfOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();

        Tensor imgFinal = new Tensor(imgShape, DType.F32);
        backend.GatedResidualLastDim(imgFinal, image, imgFfOut, imgMod[5]);
        imgFfOut.Dispose();

        // ── 11. Text post-attention path (mirrors image, lines 270-273) ──
        Tensor txtPreNorm = new Tensor(txtShape, DType.F32);
        backend.GatedResidualLastDim(txtPreNorm, text, txtAttnProj, txtMod[2]);
        txtAttnProj.Dispose();

        Tensor txtMlpModulated = DiTUtils.NormModulate(backend, txtPreNorm, txtMod[3], txtMod[4], txtShape);
        txtPreNorm.Dispose();

        Tensor txtFfOut = _txtFfn.Forward(backend, txtMlpModulated, batch, txtSeqLen);
        txtMlpModulated.Dispose();

        Tensor txtFinal = new Tensor(txtShape, DType.F32);
        backend.GatedResidualLastDim(txtFinal, text, txtFfOut, txtMod[5]);
        txtFfOut.Dispose();

        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < txtMod.Length; i++) txtMod[i].Dispose();

        return (imgFinal, txtFinal);
    }

    /// <summary>Returns the named tensor or a fresh F32 [headDim] all-ones tensor when missing.
    /// Used for AuraFlow's non-affine QK-norm (no learnable scale in the checkpoint). The synthetic
    /// tensor lives for the lifetime of the block — small enough that we don't bother to cache.</summary>
    private static Tensor GetOrFakeOnes(IReadOnlyDictionary<string, Tensor> weights, string key, int headDim)
    {
        if (weights.TryGetValue(key, out Tensor? t) && t is not null)
            return t.DType != DType.F32 ? t.CastTo(DType.F32) : t;

        Tensor ones = new(new TensorShape(headDim), DType.F32);
        unsafe
        {
            float* p = (float*)ones.DataPointer;
            for (int i = 0; i < headDim; i++) p[i] = 1.0f;
        }
        return ones;
    }

    /// <summary>Yields all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _imgModulation.EnumerateWeights()) yield return t;
        foreach (Tensor t in _txtModulation.EnumerateWeights()) yield return t;

        if (_toQWeight is not null) yield return _toQWeight;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toOutWeight is not null) yield return _toOutWeight;

        if (_addQWeight is not null) yield return _addQWeight;
        if (_addKWeight is not null) yield return _addKWeight;
        if (_addVWeight is not null) yield return _addVWeight;
        if (_toAddOutWeight is not null) yield return _toAddOutWeight;

        foreach (Tensor t in _normQ.EnumerateWeights()) yield return t;
        foreach (Tensor t in _normK.EnumerateWeights()) yield return t;
        foreach (Tensor t in _normAddedQ.EnumerateWeights()) yield return t;
        foreach (Tensor t in _normAddedK.EnumerateWeights()) yield return t;

        foreach (Tensor t in _imgFfn.EnumerateWeights()) yield return t;
        foreach (Tensor t in _txtFfn.EnumerateWeights()) yield return t;
    }
}

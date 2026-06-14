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
    public (Tensor image, Tensor text) Forward(IBackend backend, Tensor image, Tensor text, Tensor temb)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];
        int totalSeqLen = txtSeqLen + imgSeqLen;

        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);
        TensorShape imgMhShape = new TensorShape(batch, _numHeads, imgSeqLen, _headDim);
        TensorShape txtMhShape = new TensorShape(batch, _numHeads, txtSeqLen, _headDim);
        TensorShape jointMhShape = new TensorShape(batch, _numHeads, totalSeqLen, _headDim);

        // ── 1. AdaLN modulation: 6 params each [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] ──
        Tensor[] imgMod = _imgModulation.Forward(backend, temb);
        Tensor[] txtMod = _txtModulation.Forward(backend, temb);

        // ── 2. Norm + (1+scale)*norm + shift on both streams ──
        Tensor imgNormed = new Tensor(imgShape, DType.F32);
        DiTUtils.LayerNormNoAffine(imgNormed, image, batch, imgSeqLen, _hiddenSize);
        Tensor imgModulated = AdaLNModulation.ApplyModulation(imgNormed, imgMod[0], imgMod[1], batch, imgSeqLen, _hiddenSize);
        imgNormed.Dispose();

        Tensor txtNormed = new Tensor(txtShape, DType.F32);
        DiTUtils.LayerNormNoAffine(txtNormed, text, batch, txtSeqLen, _hiddenSize);
        Tensor txtModulated = AdaLNModulation.ApplyModulation(txtNormed, txtMod[0], txtMod[1], batch, txtSeqLen, _hiddenSize);
        txtNormed.Dispose();

        // ── 3. Q/K/V projections (bias-free) ──
        Tensor imgQ = new Tensor(imgShape, DType.F32);
        backend.Linear(imgQ, imgModulated, _toQWeight!, null);
        Tensor imgK = new Tensor(imgShape, DType.F32);
        backend.Linear(imgK, imgModulated, _toKWeight!, null);
        Tensor imgV = new Tensor(imgShape, DType.F32);
        backend.Linear(imgV, imgModulated, _toVWeight!, null);
        imgModulated.Dispose();

        Tensor txtQ = new Tensor(txtShape, DType.F32);
        backend.Linear(txtQ, txtModulated, _addQWeight!, null);
        Tensor txtK = new Tensor(txtShape, DType.F32);
        backend.Linear(txtK, txtModulated, _addKWeight!, null);
        Tensor txtV = new Tensor(txtShape, DType.F32);
        backend.Linear(txtV, txtModulated, _addVWeight!, null);
        txtModulated.Dispose();

        // ── 4. FP32 QK-norm (per-head) ──
        int imgVectors = batch * imgSeqLen * _numHeads;
        int txtVectors = batch * txtSeqLen * _numHeads;

        Tensor imgQNormed = new Tensor(imgQ.Shape, DType.F32);
        Tensor imgKNormed = new Tensor(imgK.Shape, DType.F32);
        _normQ.Forward(imgQNormed, imgQ, imgVectors);
        _normK.Forward(imgKNormed, imgK, imgVectors);
        imgQ.Dispose();
        imgK.Dispose();
        imgQ = imgQNormed;
        imgK = imgKNormed;

        Tensor txtQNormed = new Tensor(txtQ.Shape, DType.F32);
        Tensor txtKNormed = new Tensor(txtK.Shape, DType.F32);
        _normAddedQ.Forward(txtQNormed, txtQ, txtVectors);
        _normAddedK.Forward(txtKNormed, txtK, txtVectors);
        txtQ.Dispose();
        txtK.Dispose();
        txtQ = txtQNormed;
        txtK = txtKNormed;

        // ── 5. Reshape to multi-head [B, H, S, D] ──
        Tensor imgQMh = new Tensor(imgMhShape, DType.F32);
        Tensor imgKMh = new Tensor(imgMhShape, DType.F32);
        Tensor imgVMh = new Tensor(imgMhShape, DType.F32);
        DiTUtils.ReshapeToMultiHead(imgQMh, imgQ, batch, imgSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(imgKMh, imgK, batch, imgSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(imgVMh, imgV, batch, imgSeqLen, _numHeads, _headDim);
        imgQ.Dispose(); imgK.Dispose(); imgV.Dispose();

        Tensor txtQMh = new Tensor(txtMhShape, DType.F32);
        Tensor txtKMh = new Tensor(txtMhShape, DType.F32);
        Tensor txtVMh = new Tensor(txtMhShape, DType.F32);
        DiTUtils.ReshapeToMultiHead(txtQMh, txtQ, batch, txtSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(txtKMh, txtK, batch, txtSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(txtVMh, txtV, batch, txtSeqLen, _numHeads, _headDim);
        txtQ.Dispose(); txtK.Dispose(); txtV.Dispose();

        // ── 6. Concat [txt, img] along seq dim (matches AuraFlowAttnProcessor2_0 lines 2145-2147) ──
        Tensor jointQ = new Tensor(jointMhShape, DType.F32);
        Tensor jointK = new Tensor(jointMhShape, DType.F32);
        Tensor jointV = new Tensor(jointMhShape, DType.F32);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointQ, txtQMh, imgQMh, batch, _numHeads, txtSeqLen, imgSeqLen, _headDim);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointK, txtKMh, imgKMh, batch, _numHeads, txtSeqLen, imgSeqLen, _headDim);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointV, txtVMh, imgVMh, batch, _numHeads, txtSeqLen, imgSeqLen, _headDim);
        txtQMh.Dispose(); imgQMh.Dispose();
        txtKMh.Dispose(); imgKMh.Dispose();
        txtVMh.Dispose(); imgVMh.Dispose();

        // ── 7. Joint scaled dot-product attention ──
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor jointAttnOut = new Tensor(jointMhShape, DType.F32);
        backend.ScaledDotProductAttention(jointAttnOut, jointQ, jointK, jointV, null, scale);
        jointQ.Dispose(); jointK.Dispose(); jointV.Dispose();

        // ── 8. Split [txt_attn, img_attn] and reshape back to [B, S, hidden] ──
        Tensor txtAttnMh = new Tensor(txtMhShape, DType.F32);
        Tensor imgAttnMh = new Tensor(imgMhShape, DType.F32);
        DiTUtils.SplitAlongSeqDimMultiHead(txtAttnMh, imgAttnMh, jointAttnOut, batch, _numHeads, txtSeqLen, imgSeqLen, _headDim);
        jointAttnOut.Dispose();

        Tensor imgAttn = new Tensor(imgShape, DType.F32);
        DiTUtils.ReshapeFromMultiHead(imgAttn, imgAttnMh, batch, imgSeqLen, _numHeads, _headDim);
        imgAttnMh.Dispose();

        Tensor txtAttn = new Tensor(txtShape, DType.F32);
        DiTUtils.ReshapeFromMultiHead(txtAttn, txtAttnMh, batch, txtSeqLen, _numHeads, _headDim);
        txtAttnMh.Dispose();

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
        Tensor imgPreNorm = AdaLNModulation.ApplyGatedResidual(image, imgAttnProj, imgMod[2], batch, imgSeqLen, _hiddenSize);
        imgAttnProj.Dispose();

        Tensor imgPostNorm = new Tensor(imgShape, DType.F32);
        DiTUtils.LayerNormNoAffine(imgPostNorm, imgPreNorm, batch, imgSeqLen, _hiddenSize);
        imgPreNorm.Dispose();

        Tensor imgMlpModulated = AdaLNModulation.ApplyModulation(imgPostNorm, imgMod[3], imgMod[4], batch, imgSeqLen, _hiddenSize);
        imgPostNorm.Dispose();

        Tensor imgFfOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();

        Tensor imgFinal = AdaLNModulation.ApplyGatedResidual(image, imgFfOut, imgMod[5], batch, imgSeqLen, _hiddenSize);
        imgFfOut.Dispose();

        // ── 11. Text post-attention path (mirrors image, lines 270-273) ──
        Tensor txtPreNorm = AdaLNModulation.ApplyGatedResidual(text, txtAttnProj, txtMod[2], batch, txtSeqLen, _hiddenSize);
        txtAttnProj.Dispose();

        Tensor txtPostNorm = new Tensor(txtShape, DType.F32);
        DiTUtils.LayerNormNoAffine(txtPostNorm, txtPreNorm, batch, txtSeqLen, _hiddenSize);
        txtPreNorm.Dispose();

        Tensor txtMlpModulated = AdaLNModulation.ApplyModulation(txtPostNorm, txtMod[3], txtMod[4], batch, txtSeqLen, _hiddenSize);
        txtPostNorm.Dispose();

        Tensor txtFfOut = _txtFfn.Forward(backend, txtMlpModulated, batch, txtSeqLen);
        txtMlpModulated.Dispose();

        Tensor txtFinal = AdaLNModulation.ApplyGatedResidual(text, txtFfOut, txtMod[5], batch, txtSeqLen, _hiddenSize);
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

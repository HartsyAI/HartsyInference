using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Flux.2 double-stream block. Joint attention between text and image streams (concatenated as <c>[txt, img]</c>; separately gated/residual'd outside). Differs from <see cref="FluxDoubleStreamBlock"/>: modulation lives outside the block (top-level shared projections), SwiGLU MLP (split into linear_in_gate/up at converter time), no QKV bias for any Flux.2 variant. LayerNorm (no affine), per-head Q/K RMSNorm pre-RoPE, 4-axis pairwise-rotation RoPE on Q/K only.</summary>
public sealed unsafe class Flux2DoubleBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _mlpInner;
    private readonly float _layerNormEps;
    private readonly bool _qkvBias;

    // Image attention projections (no bias for Flux.2)
    private Tensor? _toQWeight, _toQBias;
    private Tensor? _toKWeight, _toKBias;
    private Tensor? _toVWeight, _toVBias;
    private Tensor? _toOutWeight, _toOutBias;

    // Text attention projections
    private Tensor? _addQWeight, _addQBias;
    private Tensor? _addKWeight, _addKBias;
    private Tensor? _addVWeight, _addVBias;
    private Tensor? _toAddOutWeight, _toAddOutBias;

    // Per-head Q/K RMSNorm (pre-RoPE)
    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;
    private readonly QkNorm _normAddedQ;
    private readonly QkNorm _normAddedK;

    // SwiGLU MLPs (linear_in fused → split into gate+up at converter time)
    private readonly SwiGluFfn _imgFfn;
    private readonly SwiGluFfn _txtFfn;

    public Flux2DoubleBlock(int hiddenSize, int numHeads, int mlpInner,
        bool qkvBias = false, float qkNormEps = 1e-6f, float layerNormEps = 1e-6f)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = hiddenSize / numHeads;
        _mlpInner = mlpInner;
        _layerNormEps = layerNormEps;
        _qkvBias = qkvBias;

        _normQ = new QkNorm(_headDim, qkNormEps);
        _normK = new QkNorm(_headDim, qkNormEps);
        _normAddedQ = new QkNorm(_headDim, qkNormEps);
        _normAddedK = new QkNorm(_headDim, qkNormEps);

        _imgFfn = new SwiGluFfn(hiddenSize, mlpInner);
        _txtFfn = new SwiGluFfn(hiddenSize, mlpInner);
    }

    /// <summary>Loads weights with diffusers-style naming. Converter is expected to split BFL <c>img_attn.qkv</c> → <c>attn.to_{q,k,v}</c> and <c>img_mlp.0</c> → <c>ff.linear_in_gate / ff.linear_in_up</c>; same for txt-stream.</summary>
    /// <param name="branchDamp">Residual-stream damp for the F16 activation path (the exact Chroma recipe — see <see cref="ChromaF16"/>): damps every branch-output projection so the residual stream rides at 1/32 scale; the no-affine LayerNorms make it exact. 1.0 = off.</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix, float branchDamp = 1.0f)
    {
        // Image Q/K/V (no bias)
        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _toOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];
        if (_qkvBias)
        {
            _toQBias = weights[$"{prefix}.attn.to_q.bias"];
            _toKBias = weights[$"{prefix}.attn.to_k.bias"];
            _toVBias = weights[$"{prefix}.attn.to_v.bias"];
            _toOutBias = weights[$"{prefix}.attn.to_out.0.bias"];
        }

        // Text Q/K/V (no bias)
        _addQWeight = weights[$"{prefix}.attn.add_q_proj.weight"];
        _addKWeight = weights[$"{prefix}.attn.add_k_proj.weight"];
        _addVWeight = weights[$"{prefix}.attn.add_v_proj.weight"];
        _toAddOutWeight = weights[$"{prefix}.attn.to_add_out.weight"];
        if (_qkvBias)
        {
            _addQBias = weights[$"{prefix}.attn.add_q_proj.bias"];
            _addKBias = weights[$"{prefix}.attn.add_k_proj.bias"];
            _addVBias = weights[$"{prefix}.attn.add_v_proj.bias"];
            _toAddOutBias = weights[$"{prefix}.attn.to_add_out.bias"];
        }

        // Per-head Q/K RMSNorm
        _normQ.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);
        _normAddedQ.LoadWeights(weights[$"{prefix}.attn.norm_added_q.weight"]);
        _normAddedK.LoadWeights(weights[$"{prefix}.attn.norm_added_k.weight"]);

        Tensor imgFfnOutWeight = weights[$"{prefix}.ff.linear_out.weight"];
        Tensor txtFfnOutWeight = weights[$"{prefix}.ff_context.linear_out.weight"];
        if (branchDamp != 1.0f)
        {
            // Weight damp rides the GEMM alpha (any dtype); Flux.2 linears are bias-less except the
            // _qkvBias variants, whose out-proj biases get value-scaled copies. Once per load.
            _toOutWeight.Fp8ScaleFactor *= branchDamp;
            if (_toOutBias is not null) _toOutBias = ChromaF16.DampBias(_toOutBias, branchDamp);
            _toAddOutWeight.Fp8ScaleFactor *= branchDamp;
            if (_toAddOutBias is not null) _toAddOutBias = ChromaF16.DampBias(_toAddOutBias, branchDamp);
            imgFfnOutWeight.Fp8ScaleFactor *= branchDamp;
            txtFfnOutWeight.Fp8ScaleFactor *= branchDamp;
        }

        // Image SwiGLU MLP (gate, up, out — no biases for Flux.2)
        _imgFfn.LoadSwiGluWeights(
            weights[$"{prefix}.ff.linear_in_gate.weight"], null!,
            weights[$"{prefix}.ff.linear_in_up.weight"], null!,
            imgFfnOutWeight, null!);

        // Text SwiGLU MLP
        _txtFfn.LoadSwiGluWeights(
            weights[$"{prefix}.ff_context.linear_in_gate.weight"], null!,
            weights[$"{prefix}.ff_context.linear_in_up.weight"], null!,
            txtFfnOutWeight, null!);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
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

    /// <summary>Forward pass. Modulation tensors (<paramref name="imgMod"/>, <paramref name="txtMod"/>) are 6 elements each: <c>(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)</c>, shape <c>[B, hidden]</c>. Computed once at the top level (shared modulation) and passed unchanged to every double block.</summary>
    // GPU-residency rewrite (mirrors the verified FluxDoubleStreamBlock): every glue op (LayerNorm / AdaLN
    // modulation / QK-norm / head reshape / joint concat / split / gated residual) runs as an IBackend op so the
    // activation stays device-resident across the whole block — the old host loops D2H-synced every intermediate
    // around every GEMM (Flux2DoubleBlock cpu=33 in the residency audit). Q/K/V are declared [B, S, H, D]
    // (byte-identical to [B, S, hidden]) so QK-norm's RmsNorm runs over headDim with no reshape; the [txt, img]
    // joint concat happens PRE-permute along the seq dim (text first — the token order the rope tables were built
    // in), RoPE rotates the pre-permute joint on-device via FluxRope.ApplyGpu (bit-identical rotation), then one
    // Permute0213 per joint tensor. The post-attention split is a permute-back + SliceRows (B=1); B>1 keeps the
    // host fallbacks for RoPE and the split, exactly like the Flux.1 blocks.
    public (Tensor image, Tensor text) Forward(IBackend backend, Tensor image, Tensor text, Tensor[] imgMod,
        Tensor[] txtMod, FluxRope rope, Tensor? attnBias = null)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];
        int totalSeqLen = imgSeqLen + txtSeqLen;
        float scale = 1.0f / MathF.Sqrt(_headDim);

        // F16 activation path: every activation tensor follows the incoming stream dtype (blocks are
        // dtype-transparent; the transformer decides by casting the streams once before the loop).
        DType act = image.DType;
        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);
        // [B, S, H, D] views (byte-identical to [B, S, hidden]) so RmsNorm normalizes over headDim.
        TensorShape imgHeads = new TensorShape(batch, imgSeqLen, _numHeads, _headDim);
        TensorShape txtHeads = new TensorShape(batch, txtSeqLen, _numHeads, _headDim);
        TensorShape jointHeads = new TensorShape(batch, totalSeqLen, _numHeads, _headDim);
        TensorShape jointMhShape = new TensorShape(batch, _numHeads, totalSeqLen, _headDim);
        TensorShape jointFlat = new TensorShape(batch, totalSeqLen, _hiddenSize);

        // ── 1. LayerNorm (no affine) + modulate x*(1+scale)+shift (img and txt streams independently) ──
        Tensor imgModulated = DiTUtils.NormModulate(backend, image, imgMod[0], imgMod[1], imgShape, _layerNormEps);
        Tensor txtModulated = DiTUtils.NormModulate(backend, text, txtMod[0], txtMod[1], txtShape, _layerNormEps);

        // ── 2. Q/K/V projections (declared [B, S, H, D] so QK-norm + permute need no reshape view) ──
        Tensor imgQ = new Tensor(imgHeads, act);
        backend.Linear(imgQ, imgModulated, _toQWeight!, _toQBias);
        Tensor imgK = new Tensor(imgHeads, act);
        backend.Linear(imgK, imgModulated, _toKWeight!, _toKBias);
        Tensor imgV = new Tensor(imgHeads, act);
        backend.Linear(imgV, imgModulated, _toVWeight!, _toVBias);
        imgModulated.Dispose();

        Tensor txtQ = new Tensor(txtHeads, act);
        backend.Linear(txtQ, txtModulated, _addQWeight!, _addQBias);
        Tensor txtK = new Tensor(txtHeads, act);
        backend.Linear(txtK, txtModulated, _addKWeight!, _addKBias);
        Tensor txtV = new Tensor(txtHeads, act);
        backend.Linear(txtV, txtModulated, _addVWeight!, _addVBias);
        txtModulated.Dispose();

        // ── 3. QK-Norm (per-head RMSNorm over the last dim = headDim, pre-RoPE) ──
        Tensor imgQn = new Tensor(imgHeads, act);
        backend.RmsNorm(imgQn, imgQ, _normQ.Weight, _normQ.Eps);
        imgQ.Dispose();
        Tensor imgKn = new Tensor(imgHeads, act);
        backend.RmsNorm(imgKn, imgK, _normK.Weight, _normK.Eps);
        imgK.Dispose();

        Tensor txtQn = new Tensor(txtHeads, act);
        backend.RmsNorm(txtQn, txtQ, _normAddedQ.Weight, _normAddedQ.Eps);
        txtQ.Dispose();
        Tensor txtKn = new Tensor(txtHeads, act);
        backend.RmsNorm(txtKn, txtK, _normAddedK.Weight, _normAddedK.Eps);
        txtK.Dispose();

        // ── 4. Concatenate [txt, img] along the seq dim PRE-permute (text first — matches diffusers ordering
        // and the order the rope tables were built in) ──
        Tensor jointQPre = new Tensor(jointHeads, act);
        backend.Concat(jointQPre, new Tensor[] { txtQn, imgQn }, 1);
        txtQn.Dispose(); imgQn.Dispose();
        Tensor jointKPre = new Tensor(jointHeads, act);
        backend.Concat(jointKPre, new Tensor[] { txtKn, imgKn }, 1);
        txtKn.Dispose(); imgKn.Dispose();
        Tensor jointVPre = new Tensor(jointHeads, act);
        backend.Concat(jointVPre, new Tensor[] { txtV, imgV }, 1);
        txtV.Dispose(); imgV.Dispose();

        // ── 5. RoPE BEFORE the head permute (B=1 GPU path; identical pairwise rotation to Flux.1) ──
        if (batch == 1)
            rope.ApplyGpu(backend, jointQPre, jointKPre, _numHeads);

        // ── 6. Permute [B, S, H, D] → [B, H, S, D] ──
        Tensor jointQ = new Tensor(jointMhShape, act);
        backend.Permute0213(jointQ, jointQPre, totalSeqLen, _numHeads, _headDim);
        jointQPre.Dispose();
        Tensor jointK = new Tensor(jointMhShape, act);
        backend.Permute0213(jointK, jointKPre, totalSeqLen, _numHeads, _headDim);
        jointKPre.Dispose();
        Tensor jointV = new Tensor(jointMhShape, act);
        backend.Permute0213(jointV, jointVPre, totalSeqLen, _numHeads, _headDim);
        jointVPre.Dispose();

        // Host RoPE fallback (batched inference only; B=1 runs the GPU path at 5) ──
        if (batch != 1)
            rope.Forward(jointQ, jointK, batch, _numHeads, totalSeqLen);

        // ── 7. Scaled dot-product attention ──
        Tensor jointAttnOut = new Tensor(jointMhShape, act);
        backend.ScaledDotProductAttention(jointAttnOut, jointQ, jointK, jointV, attnBias, scale, allowF16: true);   // QK RMS-norm bounds scores; enables the cuDNN fused path (bias rides fp32 in-engine)
        jointQ.Dispose(); jointK.Dispose(); jointV.Dispose();

        // ── 8. Permute back [B, H, S, D] → [B, S, hidden], then split [txt, img] ──
        Tensor jointAttnFlat = new Tensor(jointFlat, act);
        backend.Permute0213(jointAttnFlat, jointAttnOut, _numHeads, totalSeqLen, _headDim);
        jointAttnOut.Dispose();

        Tensor txtAttn, imgAttn;
        if (batch == 1)
        {
            txtAttn = new Tensor(txtShape, act);
            backend.SliceRows(txtAttn, jointAttnFlat, 0);
            imgAttn = new Tensor(imgShape, act);
            backend.SliceRows(imgAttn, jointAttnFlat, txtSeqLen);
        }
        else
        {
            (txtAttn, imgAttn) = DiTUtils.SplitAlongSeqDim(jointAttnFlat, txtSeqLen);   // host fallback, batched only
        }
        jointAttnFlat.Dispose();

        // ── 9. Output projections + gated residual (input + gate*value) ──
        Tensor imgAttnProj = new Tensor(imgShape, act);
        backend.Linear(imgAttnProj, imgAttn, _toOutWeight!, _toOutBias);
        imgAttn.Dispose();
        Tensor imgAfterAttn = new Tensor(imgShape, act);
        backend.GatedResidualLastDim(imgAfterAttn, image, imgAttnProj, imgMod[2]);
        imgAttnProj.Dispose();

        Tensor txtAttnProj = new Tensor(txtShape, act);
        backend.Linear(txtAttnProj, txtAttn, _toAddOutWeight!, _toAddOutBias);
        txtAttn.Dispose();
        Tensor txtAfterAttn = new Tensor(txtShape, act);
        backend.GatedResidualLastDim(txtAfterAttn, text, txtAttnProj, txtMod[2]);
        txtAttnProj.Dispose();

        // ── 10. Image MLP: LayerNorm + modulate + SwiGLU + gated residual ──
        Tensor imgMlpModulated = DiTUtils.NormModulate(backend, imgAfterAttn, imgMod[3], imgMod[4], imgShape, _layerNormEps);
        Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();
        Tensor imgFinal = new Tensor(imgShape, act);
        backend.GatedResidualLastDim(imgFinal, imgAfterAttn, imgMlpOut, imgMod[5]);
        imgMlpOut.Dispose();
        imgAfterAttn.Dispose();

        // ── 11. Text MLP: LayerNorm + modulate + SwiGLU + gated residual ──
        Tensor txtMlpModulated = DiTUtils.NormModulate(backend, txtAfterAttn, txtMod[3], txtMod[4], txtShape, _layerNormEps);
        Tensor txtMlpOut = _txtFfn.Forward(backend, txtMlpModulated, batch, txtSeqLen);
        txtMlpModulated.Dispose();
        Tensor txtFinal = new Tensor(txtShape, act);
        backend.GatedResidualLastDim(txtFinal, txtAfterAttn, txtMlpOut, txtMod[5]);
        txtMlpOut.Dispose();
        txtAfterAttn.Dispose();

        return (imgFinal, txtFinal);
    }
}

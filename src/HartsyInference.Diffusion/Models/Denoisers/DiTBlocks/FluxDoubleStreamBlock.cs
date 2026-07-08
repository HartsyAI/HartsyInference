using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Flux DoubleStreamBlock: dual-stream joint attention between image and text tokens with RoPE. Similar to SD3 JointBlock but adds RoPE before attention and uses GELU(tanh) MLP.</summary>
public sealed class FluxDoubleStreamBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _mlpDim;
    private readonly bool _qkvBias;

    // Image modulation: SiLU + Linear → 6 params (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
    private readonly AdaLNModulation _imgModulation;

    // Text modulation: SiLU + Linear → 6 params
    private readonly AdaLNModulation _txtModulation;

    // Image attention projections
    private Tensor? _toQWeight, _toQBias;
    private Tensor? _toKWeight, _toKBias;
    private Tensor? _toVWeight, _toVBias;
    private Tensor? _toOutWeight, _toOutBias;

    // Text attention projections
    private Tensor? _addQWeight, _addQBias;
    private Tensor? _addKWeight, _addKBias;
    private Tensor? _addVWeight, _addVBias;
    private Tensor? _toAddOutWeight, _toAddOutBias;

    // QK-norm (always used in Flux)
    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;
    private readonly QkNorm _normAddedQ;
    private readonly QkNorm _normAddedK;

    // Image MLP: GELU(tanh)
    private readonly SwiGluFfn _imgFfn;

    // Text MLP: GELU(tanh)
    private readonly SwiGluFfn _txtFfn;

    /// <summary>Creates a FluxDoubleStreamBlock.</summary>
    /// <param name="hiddenSize">Model hidden dimension (3072 for Flux.1).</param>
    /// <param name="numHeads">Number of attention heads (24 for Flux.1).</param>
    /// <param name="mlpDim">MLP inner dimension (4 * hiddenSize = 12288).</param>
    /// <param name="qkvBias">Whether Q/K/V projections have bias (true for Flux.1).</param>
    /// <param name="qkNormEps">QK-norm RMSNorm epsilon.</param>
    public FluxDoubleStreamBlock(int hiddenSize, int numHeads, int mlpDim, bool qkvBias = true, float qkNormEps = 1e-6f)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = hiddenSize / numHeads;
        _mlpDim = mlpDim;
        _qkvBias = qkvBias;

        _imgModulation = new AdaLNModulation(hiddenSize, 6);
        _txtModulation = new AdaLNModulation(hiddenSize, 6);

        _normQ = new QkNorm(_headDim, qkNormEps);
        _normK = new QkNorm(_headDim, qkNormEps);
        _normAddedQ = new QkNorm(_headDim, qkNormEps);
        _normAddedK = new QkNorm(_headDim, qkNormEps);

        _imgFfn = new SwiGluFfn(hiddenSize, mlpDim);
        _txtFfn = new SwiGluFfn(hiddenSize, mlpDim);
    }

    /// <summary>Loads weights from named tensors using diffusers naming: transformer_blocks.{i}.* </summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _imgModulation.LoadWeights(
            weights[$"{prefix}.norm1.linear.weight"],
            weights[$"{prefix}.norm1.linear.bias"]);

        _txtModulation.LoadWeights(
            weights[$"{prefix}.norm1_context.linear.weight"],
            weights[$"{prefix}.norm1_context.linear.bias"]);

        // Image Q/K/V
        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _toOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];
        _toOutBias = weights[$"{prefix}.attn.to_out.0.bias"];

        if (_qkvBias)
        {
            _toQBias = weights[$"{prefix}.attn.to_q.bias"];
            _toKBias = weights[$"{prefix}.attn.to_k.bias"];
            _toVBias = weights[$"{prefix}.attn.to_v.bias"];
        }

        // Text Q/K/V
        _addQWeight = weights[$"{prefix}.attn.add_q_proj.weight"];
        _addKWeight = weights[$"{prefix}.attn.add_k_proj.weight"];
        _addVWeight = weights[$"{prefix}.attn.add_v_proj.weight"];
        _toAddOutWeight = weights[$"{prefix}.attn.to_add_out.weight"];
        _toAddOutBias = weights[$"{prefix}.attn.to_add_out.bias"];

        if (_qkvBias)
        {
            _addQBias = weights[$"{prefix}.attn.add_q_proj.bias"];
            _addKBias = weights[$"{prefix}.attn.add_k_proj.bias"];
            _addVBias = weights[$"{prefix}.attn.add_v_proj.bias"];
        }

        // QK-norm
        _normQ.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);
        _normAddedQ.LoadWeights(weights[$"{prefix}.attn.norm_added_q.weight"]);
        _normAddedK.LoadWeights(weights[$"{prefix}.attn.norm_added_k.weight"]);

        // Image MLP (GELU mode): net.0.proj → projection, net.2 → output
        _imgFfn.LoadGeluWeights(
            weights[$"{prefix}.ff.net.0.proj.weight"],
            weights[$"{prefix}.ff.net.0.proj.bias"],
            weights[$"{prefix}.ff.net.2.weight"],
            weights[$"{prefix}.ff.net.2.bias"]);

        // Text MLP (GELU mode)
        _txtFfn.LoadGeluWeights(
            weights[$"{prefix}.ff_context.net.0.proj.weight"],
            weights[$"{prefix}.ff_context.net.0.proj.bias"],
            weights[$"{prefix}.ff_context.net.2.weight"],
            weights[$"{prefix}.ff_context.net.2.bias"]);
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in _imgModulation.EnumerateWeights()) yield return w;
        foreach (Tensor w in _txtModulation.EnumerateWeights()) yield return w;
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

    /// <summary>Forward pass. Image and text streams share joint attention with RoPE, then run independent MLPs.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="image">Image tokens [B, N_img, hidden].</param>
    /// <param name="text">Text tokens [B, N_txt, hidden].</param>
    /// <param name="temb">Timestep embedding [B, hidden].</param>
    /// <param name="rope">Precomputed FluxRope (must have Precompute called for current resolution).</param>
    /// <returns>Updated (image, text) tensors.</returns>
    // GPU-residency rewrite (mirrors the verified HunyuanImageBlock): every glue op (LayerNorm / AdaLN modulation /
    // QK-norm / head reshape / joint concat / split / gated residual) runs as an IBackend op so the activation stays
    // device-resident across the whole block — no per-op DataPointer D2H sync barriers (the old LayerNormNoAffine /
    // AdaLNModulation / QkNorm.Forward / ReshapeTo(From)MultiHead / ConcatAlongSeqDim / SplitAlongSeqDim host loops
    // D2H-synced every intermediate around every GEMM). Head reshape = declaring Q/K/V directly as [B, S, H, D]
    // (byte-identical to [B, S, hidden]) so QK-norm's RmsNorm runs over headDim with no reshape. The [txt, img]
    // joint concat happens PRE-permute along the seq dim (Concat dim 1 of [B, S, H, D] — text first, matching the
    // rope table order), so RoPE runs on the pre-permute joint via FluxRope.ApplyGpu (same interleaved-pair
    // rotation, bit-identical to the host path), then one Permute0213 per joint tensor. The post-attention split is
    // a permute-back to [B, S, hidden] + SliceRows (B=1: text rows first, then image); B>1 keeps host fallbacks for
    // RoPE and the split, exactly like the Hunyuan blocks.
    public (Tensor image, Tensor text) Forward(IBackend backend, Tensor image, Tensor text, Tensor temb, FluxRope rope, Tensor? attnBias = null)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];
        int totalSeqLen = imgSeqLen + txtSeqLen;
        float scale = 1.0f / MathF.Sqrt(_headDim);

        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);
        // [B, S, H, D] views (byte-identical to [B, S, hidden]) so RmsNorm normalizes over headDim.
        TensorShape imgHeads = new TensorShape(batch, imgSeqLen, _numHeads, _headDim);
        TensorShape txtHeads = new TensorShape(batch, txtSeqLen, _numHeads, _headDim);
        TensorShape jointHeads = new TensorShape(batch, totalSeqLen, _numHeads, _headDim);
        TensorShape jointMhShape = new TensorShape(batch, _numHeads, totalSeqLen, _headDim);
        TensorShape jointFlat = new TensorShape(batch, totalSeqLen, _hiddenSize);

        // ── 1. AdaLN modulation ──
        Tensor[] imgMod = _imgModulation.Forward(backend, temb);
        Tensor[] txtMod = _txtModulation.Forward(backend, temb);

        // ── 2. LayerNorm (no affine, eps 1e-6) + modulate x*(1+scale)+shift ──
        Tensor imgModulated = DiTUtils.NormModulate(backend, image, imgMod[0], imgMod[1], imgShape);
        Tensor txtModulated = DiTUtils.NormModulate(backend, text, txtMod[0], txtMod[1], txtShape);

        // ── 3. Q/K/V projections (declared [B, S, H, D] so QK-norm + permute need no reshape view) ──
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

        // ── 4. QK-Norm (per-head RMSNorm over the last dim = headDim) ──
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

        // ── 5. Concatenate [txt, img] along the seq dim PRE-permute ([B, S, H, D], text first — same token
        // order the old post-permute ConcatAlongSeqDim produced, and the order the rope tables were built in) ──
        Tensor jointQPre = new Tensor(jointHeads, DType.F32);
        backend.Concat(jointQPre, new Tensor[] { txtQn, imgQn }, 1);
        txtQn.Dispose(); imgQn.Dispose();
        Tensor jointKPre = new Tensor(jointHeads, DType.F32);
        backend.Concat(jointKPre, new Tensor[] { txtKn, imgKn }, 1);
        txtKn.Dispose(); imgKn.Dispose();
        Tensor jointVPre = new Tensor(jointHeads, DType.F32);
        backend.Concat(jointVPre, new Tensor[] { txtV, imgV }, 1);
        txtV.Dispose(); imgV.Dispose();

        // ── 6. RoPE BEFORE the head permute: the pre-permute [B(=1), S, H, D] layout is exactly
        // WanRopeInterleaved's [S, heads·headDim] contract, so the whole joint Q/K rotates in one GPU-resident
        // op (was a full D2H→host-trig→H2D excursion per block per step).
        if (batch == 1)
            rope.ApplyGpu(backend, jointQPre, jointKPre, _numHeads);

        // ── 7. Permute [B, S, H, D] → [B, H, S, D] ──
        Tensor jointQ = new Tensor(jointMhShape, DType.F32);
        backend.Permute0213(jointQ, jointQPre, totalSeqLen, _numHeads, _headDim);
        jointQPre.Dispose();
        Tensor jointK = new Tensor(jointMhShape, DType.F32);
        backend.Permute0213(jointK, jointKPre, totalSeqLen, _numHeads, _headDim);
        jointKPre.Dispose();
        Tensor jointV = new Tensor(jointMhShape, DType.F32);
        backend.Permute0213(jointV, jointVPre, totalSeqLen, _numHeads, _headDim);
        jointVPre.Dispose();

        // Host RoPE fallback (batched inference only; B=1 runs the GPU path at 6) ──
        if (batch != 1)
            rope.Forward(jointQ, jointK, batch, _numHeads, totalSeqLen);

        // ── 8. Scaled dot-product attention ──
        Tensor jointAttnOut = new Tensor(jointMhShape, DType.F32);
        backend.ScaledDotProductAttention(jointAttnOut, jointQ, jointK, jointV, attnBias, scale, allowF16: true);   // QK RMS-norm bounds scores; enables the cuDNN fused path (bias rides fp32 in-engine)
        jointQ.Dispose();
        jointK.Dispose();
        jointV.Dispose();

        // ── 9. Permute back [B, H, S, D] → [B, S, hidden], then split [txt, img] ──
        Tensor jointAttnFlat = new Tensor(jointFlat, DType.F32);
        backend.Permute0213(jointAttnFlat, jointAttnOut, _numHeads, totalSeqLen, _headDim);
        jointAttnOut.Dispose();

        Tensor txtAttn, imgAttn;
        if (batch == 1)
        {
            // B=1: text rows then image rows are contiguous row-blocks → device SliceRows.
            txtAttn = new Tensor(txtShape, DType.F32);
            backend.SliceRows(txtAttn, jointAttnFlat, 0);
            imgAttn = new Tensor(imgShape, DType.F32);
            backend.SliceRows(imgAttn, jointAttnFlat, txtSeqLen);
        }
        else
        {
            (txtAttn, imgAttn) = DiTUtils.SplitAlongSeqDim(jointAttnFlat, txtSeqLen);   // host fallback, batched only
        }
        jointAttnFlat.Dispose();

        // ── 10. Output projections + gated residual (input + gate*value) ──
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

        // ── 11. Image MLP: LayerNorm + modulate + GELU MLP + gated residual ──
        Tensor imgMlpModulated = DiTUtils.NormModulate(backend, imgAfterAttn, imgMod[3], imgMod[4], imgShape);
        Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();
        Tensor imgFinal = new Tensor(imgShape, DType.F32);
        backend.GatedResidualLastDim(imgFinal, imgAfterAttn, imgMlpOut, imgMod[5]);
        imgMlpOut.Dispose();
        imgAfterAttn.Dispose();

        // ── 12. Text MLP: LayerNorm + modulate + GELU MLP + gated residual ──
        Tensor txtMlpModulated = DiTUtils.NormModulate(backend, txtAfterAttn, txtMod[3], txtMod[4], txtShape);
        Tensor txtMlpOut = _txtFfn.Forward(backend, txtMlpModulated, batch, txtSeqLen);
        txtMlpModulated.Dispose();
        Tensor txtFinal = new Tensor(txtShape, DType.F32);
        backend.GatedResidualLastDim(txtFinal, txtAfterAttn, txtMlpOut, txtMod[5]);
        txtMlpOut.Dispose();
        txtAfterAttn.Dispose();

        // Dispose modulation tensors
        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < txtMod.Length; i++) txtMod[i].Dispose();

        return (imgFinal, txtFinal);
    }
}

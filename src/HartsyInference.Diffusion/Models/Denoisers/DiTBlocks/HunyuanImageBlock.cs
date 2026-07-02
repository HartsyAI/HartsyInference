using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Hunyuan Image dual-stream MMDiT block (<c>HunyuanImageTransformerBlock</c>). Maintains separate image and text streams with independent AdaLN-Zero modulation (6 params each: <c>shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp</c>) — chunked from a single Linear that emits <c>6 * hidden</c>. Joint attention applies image-only RoPE before concatenating <c>[img, txt]</c> along the sequence dim, then SDPA, then splits back to per-stream output projections. FFN is GELU-approximate (tanh) with <c>ff.net.0.proj</c> + <c>ff.net.2</c> diffusers naming. Mirrors <c>diffusers/models/transformers/transformer_hunyuanimage.py:HunyuanImageTransformerBlock.forward</c> 1:1.</summary>
public sealed unsafe class HunyuanImageBlock : IStreamingBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _mlpDim;

    private readonly AdaLNModulation _imgModulation;
    private readonly AdaLNModulation _txtModulation;

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

    /// <summary>Creates a Hunyuan Image dual-stream block.</summary>
    /// <param name="hiddenSize">Model hidden dimension (3072 for V2.1).</param>
    /// <param name="numHeads">Number of attention heads (24 for V2.1).</param>
    /// <param name="headDim">Per-head dimension (128 for V2.1; <c>numHeads * headDim == hiddenSize</c>).</param>
    /// <param name="mlpDim">MLP inner dimension (4 * hiddenSize = 12288 for V2.1).</param>
    /// <param name="qkNormEps">QK-norm RMSNorm epsilon. Default 1e-6.</param>
    public HunyuanImageBlock(int hiddenSize, int numHeads, int headDim, int mlpDim, float qkNormEps = 1e-6f)
    {
        if (numHeads * headDim != hiddenSize)
            throw new ArgumentException($"numHeads * headDim ({numHeads} * {headDim}) must equal hiddenSize ({hiddenSize}).");

        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = headDim;
        _mlpDim = mlpDim;

        _imgModulation = new AdaLNModulation(hiddenSize, 6);
        _txtModulation = new AdaLNModulation(hiddenSize, 6);

        _normQ = new QkNorm(headDim, qkNormEps);
        _normK = new QkNorm(headDim, qkNormEps);
        _normAddedQ = new QkNorm(headDim, qkNormEps);
        _normAddedK = new QkNorm(headDim, qkNormEps);

        _imgFfn = new SwiGluFfn(hiddenSize, mlpDim);
        _txtFfn = new SwiGluFfn(hiddenSize, mlpDim);
    }

    /// <summary>Loads weights using diffusers naming under <c>transformer_blocks.{i}.*</c>. Modulation linears project temb to <c>6 * hidden</c>; attention is bias=True; QK-norm is RMSNorm; FFNs are GELU(approximate) with <c>ff.net.0.proj</c> + <c>ff.net.2</c> naming.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _imgModulation.LoadWeights(
            weights[$"{prefix}.norm1.linear.weight"],
            weights[$"{prefix}.norm1.linear.bias"]);
        _txtModulation.LoadWeights(
            weights[$"{prefix}.norm1_context.linear.weight"],
            weights[$"{prefix}.norm1_context.linear.bias"]);

        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _toOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];
        _toOutBias = weights[$"{prefix}.attn.to_out.0.bias"];
        weights.TryGetValue($"{prefix}.attn.to_q.bias", out _toQBias);
        weights.TryGetValue($"{prefix}.attn.to_k.bias", out _toKBias);
        weights.TryGetValue($"{prefix}.attn.to_v.bias", out _toVBias);

        _addQWeight = weights[$"{prefix}.attn.add_q_proj.weight"];
        _addKWeight = weights[$"{prefix}.attn.add_k_proj.weight"];
        _addVWeight = weights[$"{prefix}.attn.add_v_proj.weight"];
        _toAddOutWeight = weights[$"{prefix}.attn.to_add_out.weight"];
        _toAddOutBias = weights[$"{prefix}.attn.to_add_out.bias"];
        weights.TryGetValue($"{prefix}.attn.add_q_proj.bias", out _addQBias);
        weights.TryGetValue($"{prefix}.attn.add_k_proj.bias", out _addKBias);
        weights.TryGetValue($"{prefix}.attn.add_v_proj.bias", out _addVBias);

        _normQ.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);
        _normAddedQ.LoadWeights(weights[$"{prefix}.attn.norm_added_q.weight"]);
        _normAddedK.LoadWeights(weights[$"{prefix}.attn.norm_added_k.weight"]);

        _imgFfn.LoadGeluWeights(
            weights[$"{prefix}.ff.net.0.proj.weight"],
            weights[$"{prefix}.ff.net.0.proj.bias"],
            weights[$"{prefix}.ff.net.2.weight"],
            weights[$"{prefix}.ff.net.2.bias"]);

        _txtFfn.LoadGeluWeights(
            weights[$"{prefix}.ff_context.net.0.proj.weight"],
            weights[$"{prefix}.ff_context.net.0.proj.bias"],
            weights[$"{prefix}.ff_context.net.2.weight"],
            weights[$"{prefix}.ff_context.net.2.bias"]);
    }

    /// <summary>Sum of weight bytes in this streamable block (for the block-streaming budget heuristic).</summary>
    public long EstimatedWeightBytes
    {
        get { long t = 0; foreach (Tensor w in EnumerateWeights()) t += w.ElementCount * w.DType.SizeInBytes; return t; }
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

    /// <summary>Forward pass. Each stream's modulation produces 6 params; image-only RoPE is applied via <see cref="HunyuanImageRope"/> before <c>[img, txt]</c> joint-attention concat. Returns <c>(image, text)</c>.</summary>
    // GPU-residency rewrite (mirrors the verified QwenImageBlock): every glue op (LayerNorm / AdaLN modulation /
    // QK-norm / head reshape / joint concat / split / gated residual) runs as an IBackend op so the activation stays
    // device-resident across the whole block — no per-op DataPointer D2H sync barriers (the old DiTUtils / QkNorm /
    // AdaLNModulation host path D2H-synced every intermediate ~10 times per block × 60 blocks × steps). The only op
    // left on the CPU is RoPE (HunyuanImageRope): it is image-only here, applied to the standalone per-stream image
    // multi-head Q/K BEFORE the joint concat — one contained host excursion over a small tensor, exactly as before.
    // The head reshape is expressed as declaring Q/K/V directly as [B, S, H, D] (byte-identical to [B, S, hidden]) so
    // QK-norm runs over headDim with no reshape, then Permute0213 to [B, H, S, D]; the [img, txt] seq-dim concat is a
    // Concat at dim 2 (byte-identical to ConcatAlongSeqDimMultiHead), and the post-attention split is a permute-back
    // to [B, S, hidden] + SliceRows (B=1: image rows first, then text — see class note / QwenImageBlock).
    public (Tensor image, Tensor text) Forward(IBackend backend, Tensor image, Tensor text, Tensor temb,
        HunyuanImageRope rope, int imgPackedH, int imgPackedW, int imgPackedT = 1)
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
        // [B, S, H, D] views (byte-identical to [B, S, hidden]) so RmsNorm normalizes over headDim.
        TensorShape imgHeads = new TensorShape(batch, imgSeqLen, _numHeads, _headDim);
        TensorShape txtHeads = new TensorShape(batch, txtSeqLen, _numHeads, _headDim);
        TensorShape imgMhShape = new TensorShape(batch, _numHeads, imgSeqLen, _headDim);
        TensorShape txtMhShape = new TensorShape(batch, _numHeads, txtSeqLen, _headDim);
        TensorShape jointMhShape = new TensorShape(batch, _numHeads, totalSeqLen, _headDim);
        TensorShape jointFlat = new TensorShape(batch, totalSeqLen, _hiddenSize);

        // ── 1. LayerNorm (no affine) + AdaLN modulate: x*(1+scale)+shift ──
        Tensor imgModulated = DiTUtils.NormModulate(backend, image, imgMod[0], imgMod[1], imgShape);
        Tensor txtModulated = DiTUtils.NormModulate(backend, text, txtMod[0], txtMod[1], txtShape);

        // ── 2. Q/K/V projections (declared [B, S, H, D] so QK-norm + permute need no reshape view) ──
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

        // ── 4. Permute [B, S, H, D] → [B, H, S, D] ──
        Tensor imgQMh = new Tensor(imgMhShape, DType.F32);
        backend.Permute0213(imgQMh, imgQn, imgSeqLen, _numHeads, _headDim);
        imgQn.Dispose();
        Tensor imgKMh = new Tensor(imgMhShape, DType.F32);
        backend.Permute0213(imgKMh, imgKn, imgSeqLen, _numHeads, _headDim);
        imgKn.Dispose();
        Tensor imgVMh = new Tensor(imgMhShape, DType.F32);
        backend.Permute0213(imgVMh, imgV, imgSeqLen, _numHeads, _headDim);
        imgV.Dispose();

        Tensor txtQMh = new Tensor(txtMhShape, DType.F32);
        backend.Permute0213(txtQMh, txtQn, txtSeqLen, _numHeads, _headDim);
        txtQn.Dispose();
        Tensor txtKMh = new Tensor(txtMhShape, DType.F32);
        backend.Permute0213(txtKMh, txtKn, txtSeqLen, _numHeads, _headDim);
        txtKn.Dispose();
        Tensor txtVMh = new Tensor(txtMhShape, DType.F32);
        backend.Permute0213(txtVMh, txtV, txtSeqLen, _numHeads, _headDim);
        txtV.Dispose();

        // ── 5. Image-only RoPE (CPU; standalone [B, H, imgSeq, D] tensor — see class note) ──
        if (imgPackedT > 1)
        {
            Span<int> dims = stackalloc int[3] { imgPackedT, imgPackedH, imgPackedW };
            rope.ApplyJoint(imgQMh, imgKMh, batch, _numHeads, dims);
        }
        else
            rope.ApplyImage(imgQMh, imgKMh, batch, _numHeads, imgPackedH, imgPackedW);

        // ── 6. Concat [img, txt] along the seq dim of [B, H, S, D] ──
        Tensor jointQ = new Tensor(jointMhShape, DType.F32);
        backend.Concat(jointQ, new Tensor[] { imgQMh, txtQMh }, 2);
        Tensor jointK = new Tensor(jointMhShape, DType.F32);
        backend.Concat(jointK, new Tensor[] { imgKMh, txtKMh }, 2);
        Tensor jointV = new Tensor(jointMhShape, DType.F32);
        backend.Concat(jointV, new Tensor[] { imgVMh, txtVMh }, 2);
        imgQMh.Dispose(); txtQMh.Dispose();
        imgKMh.Dispose(); txtKMh.Dispose();
        imgVMh.Dispose(); txtVMh.Dispose();

        // ── 7. Joint scaled dot-product attention (no mask) ──
        Tensor jointAttnOut = new Tensor(jointMhShape, DType.F32);
        backend.ScaledDotProductAttention(jointAttnOut, jointQ, jointK, jointV, null, scale);
        jointQ.Dispose();
        jointK.Dispose();
        jointV.Dispose();

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
        Tensor imgMlpModulated = DiTUtils.NormModulate(backend, imgAfterAttn, imgMod[3], imgMod[4], imgShape);
        Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();
        Tensor imgFinal = new Tensor(imgShape, DType.F32);
        backend.GatedResidualLastDim(imgFinal, imgAfterAttn, imgMlpOut, imgMod[5]);
        imgMlpOut.Dispose();
        imgAfterAttn.Dispose();

        // ── 11. Text MLP path ──
        Tensor txtMlpModulated = DiTUtils.NormModulate(backend, txtAfterAttn, txtMod[3], txtMod[4], txtShape);
        Tensor txtMlpOut = _txtFfn.Forward(backend, txtMlpModulated, batch, txtSeqLen);
        txtMlpModulated.Dispose();
        Tensor txtFinal = new Tensor(txtShape, DType.F32);
        backend.GatedResidualLastDim(txtFinal, txtAfterAttn, txtMlpOut, txtMod[5]);
        txtMlpOut.Dispose();
        txtAfterAttn.Dispose();

        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < txtMod.Length; i++) txtMod[i].Dispose();

        return (imgFinal, txtFinal);
    }
}

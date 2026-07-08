using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Qwen-Image dual-stream MMDiT block (<c>QwenImageTransformerBlock</c>). Maintains separate image and text streams with independent AdaLN-Zero modulation (12 params each: <c>shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp</c> for both <c>mod1</c> and <c>mod2</c>), QK-norm, and GELU(approximate) FFNs. Joint attention concatenates <c>[txt, img]</c> along the sequence dim and applies precomputed RoPE separately to image and text (image-only RoPE for image tokens; zero-position RoPE rows for text tokens). Mirrors <c>diffusers/models/transformers/transformer_qwenimage.py:QwenImageTransformerBlock.forward</c> 1:1.</summary>
public sealed unsafe class QwenImageBlock
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

    /// <summary>Creates a Qwen-Image dual-stream block.</summary>
    /// <param name="hiddenSize">Model hidden dimension (3072 for V1).</param>
    /// <param name="numHeads">Number of attention heads (24 for V1).</param>
    /// <param name="headDim">Per-head dimension (128 for V1; <c>numHeads * headDim == hiddenSize</c>).</param>
    /// <param name="mlpDim">MLP inner dimension (4 * hiddenSize = 12288 for V1).</param>
    /// <param name="qkNormEps">QK-norm RMSNorm epsilon. Default 1e-6.</param>
    public QwenImageBlock(int hiddenSize, int numHeads, int headDim, int mlpDim, float qkNormEps = 1e-6f)
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

    /// <summary>Loads weights using diffusers naming under <c>transformer_blocks.{i}.*</c>. Modulation linears project temb to <c>2 * 6 * hidden</c> (mod1 + mod2 each with 6 params); attention is bias=True; QK-norm is RMSNorm; FFNs are GELU(approximate) with <c>ff.net.0.proj</c> + <c>ff.net.2</c> naming.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _imgModulation.LoadWeights(
            weights[$"{prefix}.img_mod.1.weight"],
            weights[$"{prefix}.img_mod.1.bias"]);
        _txtModulation.LoadWeights(
            weights[$"{prefix}.txt_mod.1.weight"],
            weights[$"{prefix}.txt_mod.1.bias"]);

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

        _imgFfn.LoadGeluWeights(
            weights[$"{prefix}.img_mlp.net.0.proj.weight"],
            weights[$"{prefix}.img_mlp.net.0.proj.bias"],
            weights[$"{prefix}.img_mlp.net.2.weight"],
            weights[$"{prefix}.img_mlp.net.2.bias"]);

        _txtFfn.LoadGeluWeights(
            weights[$"{prefix}.txt_mlp.net.0.proj.weight"],
            weights[$"{prefix}.txt_mlp.net.0.proj.bias"],
            weights[$"{prefix}.txt_mlp.net.2.weight"],
            weights[$"{prefix}.txt_mlp.net.2.bias"]);
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

    /// <summary>Forward pass. Each modulation produces <c>[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]</c>. Joint attention concats <c>[txt, img]</c>; QK rotated by <see cref="QwenImageRope"/> separately for image (per-token <c>[0, row, col]</c>) and text (offset by <paramref name="txtPositionStart"/>) before concat. Returns <c>(image, text)</c>.</summary>
    // GPU-residency rewrite (mirrors the verified ChromaDoubleStreamBlock): every glue op (LayerNorm / AdaLN
    // modulation / QK-norm / reshape-to-heads / joint concat / split / gated residual) runs as an IBackend GPU op so
    // the activation stays device-resident across the whole block — no per-op DataPointer reads / D2H sync barriers
    // (the old DiTUtils/QkNorm/AdaLNModulation CPU path D2H-synced every ~50 MB intermediate, ~16 times per block ×
    // 60 blocks × CFG, which was the 10+ min/gen cost). The only op left on the CPU is RoPE: QwenImageRope is
    // interleaved (GPT-J / complex-polar pairing), and the CUDA ApplyRope kernel is rotate-half (NEOX) — no match —
    // so the two per-stream rotations are fused into ONE joint pass (QwenImageRope.ApplyJoint, bit-identical since
    // RoPE is per-row independent) over the already-GPU-concatenated [txt, img] sequence. Batch is always 1 here
    // (QwenImagePipeline runs CFG as two batch-1 passes), which the seq-dim slice relies on.
    /// <param name="refPackedH">Packed-grid height of the trailing Qwen-Image-Edit reference-latent token section
    /// (0 = no ref tokens). Ref tokens ride at the END of the image stream and get their own RoPE positions
    /// (frame axis 1, centered on the ref grid) — everything else in the block treats them as ordinary image tokens.</param>
    /// <param name="refPackedW">Packed-grid width of the reference-latent section (0 = no ref tokens).</param>
    public (Tensor image, Tensor text) Forward(IBackend backend, Tensor image, Tensor text, Tensor temb,
        QwenImageRope rope, int imgPackedH, int imgPackedW, int txtPositionStart,
        int refPackedH = 0, int refPackedW = 0)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];
        int totalSeqLen = imgSeqLen + txtSeqLen;
        float scale = 1.0f / MathF.Sqrt(_headDim);

        // Modulation params: img_mod / txt_mod each → [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp].
        // .Forward runs Silu+Linear on the GPU; the per-row split it returns is tiny ([B, hidden] each) — negligible.
        Tensor[] imgMod = _imgModulation.Forward(backend, temb);
        Tensor[] txtMod = _txtModulation.Forward(backend, temb);

        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);
        // [B, S, H, D] views (byte-identical to [B, S, hidden]) so RmsNorm normalizes over headDim.
        TensorShape imgHeads = new TensorShape(batch, imgSeqLen, _numHeads, _headDim);
        TensorShape txtHeads = new TensorShape(batch, txtSeqLen, _numHeads, _headDim);
        TensorShape jointFlat = new TensorShape(batch, totalSeqLen, _hiddenSize);
        TensorShape jointMh = new TensorShape(batch, _numHeads, totalSeqLen, _headDim);

        // ── 1. LayerNorm (no affine, eps 1e-6) + AdaLN modulate: x*(1+scale)+shift ──
        Tensor imgModulated = NormModulate(backend, image, imgMod[0], imgMod[1], imgShape);
        Tensor txtModulated = NormModulate(backend, text, txtMod[0], txtMod[1], txtShape);

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

        // ── 4. Concat [txt, img] along the seq dim (contiguous row-concat in [B, S, hidden] layout) ──
        Tensor jointQf = new Tensor(jointFlat, DType.F32);
        backend.Concat(jointQf, new Tensor[] { txtQn, imgQn }, 1);
        Tensor jointKf = new Tensor(jointFlat, DType.F32);
        backend.Concat(jointKf, new Tensor[] { txtKn, imgKn }, 1);
        Tensor jointVf = new Tensor(jointFlat, DType.F32);
        backend.Concat(jointVf, new Tensor[] { txtV, imgV }, 1);
        txtQn.Dispose(); imgQn.Dispose();
        txtKn.Dispose(); imgKn.Dispose();
        txtV.Dispose(); imgV.Dispose();

        // ── 5. RoPE on the joint [txt, img(, ref)] Q,K — device kernel on the PRE-permute [B, S, H, D] layout
        // (rope rotates each (s, h) vector independently, so applying before the head permute is identical to
        // the old post-permute host pass). The cos/sin tables are position-only and cached across blocks/steps;
        // the host ApplyJoint remains as the batch>1 fallback and the tests' numerical reference. ──
        if (batch == 1)
        {
            (Tensor ropeCos, Tensor ropeSin) = rope.GetOrBuildJointTables(
                imgPackedH, imgPackedW, txtSeqLen, txtPositionStart, refPackedH, refPackedW);
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
        {
            rope.ApplyJoint(jointQ, jointK, batch, _numHeads, imgPackedH, imgPackedW, txtSeqLen, txtPositionStart,
                refPackedH, refPackedW);
        }

        // ── 7. Joint scaled dot-product attention (no mask) ──
        // allowF16: Q/K are per-head RMSNormed above, so scores are bounded — F16 SDPA is range-safe and
        // engages the cuDNN fused flash path (head_dim 128, no mask: the proven Krea2/Z-Image config).
        Tensor jointAttnOut = new Tensor(jointMh, DType.F32);
        backend.ScaledDotProductAttention(jointAttnOut, jointQ, jointK, jointV, null, scale, allowF16: true);
        jointQ.Dispose();
        jointK.Dispose();
        jointV.Dispose();

        // ── 8. Permute back [B, H, S, D] → [B, S, hidden], then split [txt, img] (B=1: contiguous rows) ──
        Tensor jointAttnFlat = new Tensor(jointFlat, DType.F32);
        backend.Permute0213(jointAttnFlat, jointAttnOut, _numHeads, totalSeqLen, _headDim);
        jointAttnOut.Dispose();

        Tensor txtAttn = new Tensor(txtShape, DType.F32);
        backend.SliceRows(txtAttn, jointAttnFlat, 0);
        Tensor imgAttn = new Tensor(imgShape, DType.F32);
        backend.SliceRows(imgAttn, jointAttnFlat, txtSeqLen);
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
        Tensor imgMlpModulated = NormModulate(backend, imgAfterAttn, imgMod[3], imgMod[4], imgShape);
        Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();
        Tensor imgFinal = new Tensor(imgShape, DType.F32);
        backend.GatedResidualLastDim(imgFinal, imgAfterAttn, imgMlpOut, imgMod[5]);
        imgMlpOut.Dispose();
        imgAfterAttn.Dispose();

        // ── 11. Text MLP path ──
        Tensor txtMlpModulated = NormModulate(backend, txtAfterAttn, txtMod[3], txtMod[4], txtShape);
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

    /// <summary>LayerNorm (no affine, eps 1e-6) followed by AdaLN modulation <c>out = x*(1+scale)+shift</c>, all on the
    /// GPU. <c>AffineBroadcastLastDim</c> computes <c>x*scale+shift</c>, so the scale tensor is pre-incremented by 1
    /// (<c>AddScalar</c>) to reproduce the <c>(1+scale)</c> factor — bit-identical to the old CPU
    /// <see cref="AdaLNModulation.ApplyModulation"/>. Mirrors ChromaDoubleStreamBlock.NormModulate.</summary>
    private static Tensor NormModulate(IBackend backend, Tensor x, Tensor shift, Tensor scale, TensorShape shape)
    {
        Tensor normed = new Tensor(shape, DType.F32);
        backend.LayerNormNoAffine(normed, x, 1e-6f);
        Tensor scalePlus1 = new Tensor(scale.Shape, DType.F32);
        backend.AddScalar(scalePlus1, scale, 1.0f);
        Tensor output = new Tensor(shape, DType.F32);
        backend.AffineBroadcastLastDim(output, normed, scalePlus1, shift);
        normed.Dispose();
        scalePlus1.Dispose();
        return output;
    }
}

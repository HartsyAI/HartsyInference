using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Chroma double-stream (joint-attention) block. Mirrors <see cref="FluxDoubleStreamBlock"/> but takes
/// the modulation parameters precomputed as a <c>[B, 12, hidden]</c> slice from the approximator's modulation table —
/// the per-block <c>norm1.linear</c> and <c>norm1_context.linear</c> projections are pruned (Chroma's defining
/// architectural change). Rows <c>0..6</c> are the image stream's <c>(shift_msa, scale_msa, gate_msa, shift_mlp,
/// scale_mlp, gate_mlp)</c>; rows <c>6..12</c> are the same for the text stream.
///
/// Reference: <c>diffusers/models/transformers/transformer_chroma.py:276-369</c>.
/// Also supports an optional attention mask <c>[B, totalSeqLen]</c> that gets expanded into the SDPA mask.</summary>
public sealed unsafe class ChromaDoubleStreamBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _mlpDim;

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

    // QK-norm (per-head RMSNorm with affine scale)
    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;
    private readonly QkNorm _normAddedQ;
    private readonly QkNorm _normAddedK;

    // Image MLP (GELU(tanh) approximate per diffusers FeedForward(activation_fn="gelu-approximate"))
    private readonly SwiGluFfn _imgFfn;

    // Text MLP (GELU(tanh) approximate)
    private readonly SwiGluFfn _txtFfn;

    /// <summary>Creates a ChromaDoubleStreamBlock.</summary>
    /// <param name="hiddenSize">Model hidden dimension (3072 for Chroma v1).</param>
    /// <param name="numHeads">Number of attention heads (24 for v1).</param>
    /// <param name="headDim">Per-head dimension (128 for v1; product = hiddenSize).</param>
    /// <param name="qkNormEps">QK-norm RMSNorm epsilon. Default 1e-6.</param>
    public ChromaDoubleStreamBlock(int hiddenSize, int numHeads, int headDim, float qkNormEps = 1e-6f)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = headDim;
        _mlpDim = hiddenSize * 4;

        _normQ = new QkNorm(_headDim, qkNormEps);
        _normK = new QkNorm(_headDim, qkNormEps);
        _normAddedQ = new QkNorm(_headDim, qkNormEps);
        _normAddedK = new QkNorm(_headDim, qkNormEps);

        _imgFfn = new SwiGluFfn(hiddenSize, _mlpDim);
        _txtFfn = new SwiGluFfn(hiddenSize, _mlpDim);
    }

    /// <summary>Loads weights using post-conversion diffusers-style naming under <c>transformer_blocks.{i}.*</c>.
    /// The <c>norm1</c> / <c>norm1_context</c> linears are intentionally absent (Chroma's pruned-AdaLN architecture).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        // Image Q/K/V (bias=True per FluxAttention)
        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _toOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];
        _toOutBias = weights[$"{prefix}.attn.to_out.0.bias"];
        _toQBias = weights[$"{prefix}.attn.to_q.bias"];
        _toKBias = weights[$"{prefix}.attn.to_k.bias"];
        _toVBias = weights[$"{prefix}.attn.to_v.bias"];

        // Text Q/K/V
        _addQWeight = weights[$"{prefix}.attn.add_q_proj.weight"];
        _addKWeight = weights[$"{prefix}.attn.add_k_proj.weight"];
        _addVWeight = weights[$"{prefix}.attn.add_v_proj.weight"];
        _toAddOutWeight = weights[$"{prefix}.attn.to_add_out.weight"];
        _toAddOutBias = weights[$"{prefix}.attn.to_add_out.bias"];
        _addQBias = weights[$"{prefix}.attn.add_q_proj.bias"];
        _addKBias = weights[$"{prefix}.attn.add_k_proj.bias"];
        _addVBias = weights[$"{prefix}.attn.add_v_proj.bias"];

        // QK-norm
        _normQ.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);
        _normAddedQ.LoadWeights(weights[$"{prefix}.attn.norm_added_q.weight"]);
        _normAddedK.LoadWeights(weights[$"{prefix}.attn.norm_added_k.weight"]);

        // Image MLP (GELU mode: net.0.proj → projection, net.2 → output)
        _imgFfn.LoadGeluWeights(
            weights[$"{prefix}.ff.net.0.proj.weight"],
            weights[$"{prefix}.ff.net.0.proj.bias"],
            weights[$"{prefix}.ff.net.2.weight"],
            weights[$"{prefix}.ff.net.2.bias"]);

        // Text MLP
        _txtFfn.LoadGeluWeights(
            weights[$"{prefix}.ff_context.net.0.proj.weight"],
            weights[$"{prefix}.ff_context.net.0.proj.bias"],
            weights[$"{prefix}.ff_context.net.2.weight"],
            weights[$"{prefix}.ff_context.net.2.bias"]);
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
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

    /// <summary>Forward pass. <paramref name="temb"/> is a <c>[B, 12, hidden]</c> slice of the global modulation
    /// table: rows 0..6 modulate the image stream, rows 6..12 modulate the text stream. Returns
    /// <c>(image, text)</c> in the diffusers' <c>(hidden_states, encoder_hidden_states)</c> argument order.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="image">Image tokens [B, N_img, hidden].</param>
    /// <param name="text">Text tokens [B, N_txt, hidden].</param>
    /// <param name="temb">Precomputed modulation rows [B, 12, hidden].</param>
    /// <param name="rope">Precomputed FluxRope.</param>
    /// <param name="attentionMask">Optional [B, totalSeqLen] mask (1=keep, 0=mask). Expanded to [B, 1, S, S] inside.</param>
    public (Tensor image, Tensor text) Forward(
        IBackend backend, Tensor image, Tensor text, Tensor temb, FluxRope rope, Tensor? attentionMask)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];
        int totalSeqLen = imgSeqLen + txtSeqLen;

        // ── 1. Slice the precomputed modulation rows ──
        // imgMod[0..6]: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp (image stream)
        // txtMod[0..6]: same for text stream
        Tensor[] imgMod = SliceModRows(temb, batch, rowStart: 0, rowCount: 6);
        Tensor[] txtMod = SliceModRows(temb, batch, rowStart: 6, rowCount: 6);

        // ── 2. LayerNorm (no affine, eps=1e-6) + modulate ──
        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        Tensor imgNormed = new Tensor(imgShape, DType.F32);
        DiTUtils.LayerNormNoAffine(imgNormed, image, batch, imgSeqLen, _hiddenSize);
        Tensor imgModulated = AdaLNModulation.ApplyModulation(imgNormed, imgMod[0], imgMod[1], batch, imgSeqLen, _hiddenSize);
        imgNormed.Dispose();

        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);
        Tensor txtNormed = new Tensor(txtShape, DType.F32);
        DiTUtils.LayerNormNoAffine(txtNormed, text, batch, txtSeqLen, _hiddenSize);
        Tensor txtModulated = AdaLNModulation.ApplyModulation(txtNormed, txtMod[0], txtMod[1], batch, txtSeqLen, _hiddenSize);
        txtNormed.Dispose();

        // ── 3. Q/K/V projections ──
        Tensor imgQ = new Tensor(imgShape, DType.F32);
        backend.Linear(imgQ, imgModulated, _toQWeight!, _toQBias);
        Tensor imgK = new Tensor(imgShape, DType.F32);
        backend.Linear(imgK, imgModulated, _toKWeight!, _toKBias);
        Tensor imgV = new Tensor(imgShape, DType.F32);
        backend.Linear(imgV, imgModulated, _toVWeight!, _toVBias);
        imgModulated.Dispose();

        Tensor txtQ = new Tensor(txtShape, DType.F32);
        backend.Linear(txtQ, txtModulated, _addQWeight!, _addQBias);
        Tensor txtK = new Tensor(txtShape, DType.F32);
        backend.Linear(txtK, txtModulated, _addKWeight!, _addKBias);
        Tensor txtV = new Tensor(txtShape, DType.F32);
        backend.Linear(txtV, txtModulated, _addVWeight!, _addVBias);
        txtModulated.Dispose();

        // ── 4. QK-Norm (per-head RMSNorm) ──
        int imgVectors = batch * imgSeqLen * _numHeads;
        int txtVectors = batch * txtSeqLen * _numHeads;

        Tensor imgQNormed = new Tensor(imgQ.Shape, DType.F32);
        Tensor imgKNormed = new Tensor(imgK.Shape, DType.F32);
        _normQ.Forward(imgQNormed, imgQ, imgVectors);
        _normK.Forward(imgKNormed, imgK, imgVectors);
        imgQ.Dispose();
        imgK.Dispose();

        Tensor txtQNormed = new Tensor(txtQ.Shape, DType.F32);
        Tensor txtKNormed = new Tensor(txtK.Shape, DType.F32);
        _normAddedQ.Forward(txtQNormed, txtQ, txtVectors);
        _normAddedK.Forward(txtKNormed, txtK, txtVectors);
        txtQ.Dispose();
        txtK.Dispose();

        // ── 5. Reshape to multi-head [B, H, S, D] ──
        TensorShape imgMhShape = new TensorShape(batch, _numHeads, imgSeqLen, _headDim);
        TensorShape txtMhShape = new TensorShape(batch, _numHeads, txtSeqLen, _headDim);
        TensorShape jointMhShape = new TensorShape(batch, _numHeads, totalSeqLen, _headDim);

        Tensor imgQMh = new Tensor(imgMhShape, DType.F32);
        Tensor imgKMh = new Tensor(imgMhShape, DType.F32);
        Tensor imgVMh = new Tensor(imgMhShape, DType.F32);
        DiTUtils.ReshapeToMultiHead(imgQMh, imgQNormed, batch, imgSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(imgKMh, imgKNormed, batch, imgSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(imgVMh, imgV, batch, imgSeqLen, _numHeads, _headDim);
        imgQNormed.Dispose();
        imgKNormed.Dispose();
        imgV.Dispose();

        Tensor txtQMh = new Tensor(txtMhShape, DType.F32);
        Tensor txtKMh = new Tensor(txtMhShape, DType.F32);
        Tensor txtVMh = new Tensor(txtMhShape, DType.F32);
        DiTUtils.ReshapeToMultiHead(txtQMh, txtQNormed, batch, txtSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(txtKMh, txtKNormed, batch, txtSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(txtVMh, txtV, batch, txtSeqLen, _numHeads, _headDim);
        txtQNormed.Dispose();
        txtKNormed.Dispose();
        txtV.Dispose();

        // ── 6. Concat [txt, img] for joint attention ──
        Tensor jointQ = new Tensor(jointMhShape, DType.F32);
        Tensor jointK = new Tensor(jointMhShape, DType.F32);
        Tensor jointV = new Tensor(jointMhShape, DType.F32);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointQ, txtQMh, imgQMh, batch, _numHeads, txtSeqLen, imgSeqLen, _headDim);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointK, txtKMh, imgKMh, batch, _numHeads, txtSeqLen, imgSeqLen, _headDim);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointV, txtVMh, imgVMh, batch, _numHeads, txtSeqLen, imgSeqLen, _headDim);
        txtQMh.Dispose(); imgQMh.Dispose();
        txtKMh.Dispose(); imgKMh.Dispose();
        txtVMh.Dispose(); imgVMh.Dispose();

        // ── 7. RoPE on concatenated Q and K ──
        rope.Forward(jointQ, jointK, batch, _numHeads, totalSeqLen);

        // ── 8. Build SDPA mask if requested: [B, S] -> [B, 1, S, S] via outer product ──
        Tensor? sdpaMask = attentionMask is not null
            ? BuildSdpaMask(attentionMask, batch, totalSeqLen, _numHeads)
            : null;

        // ── 9. Scaled dot-product attention ──
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor jointAttnOut = new Tensor(jointMhShape, DType.F32);
        backend.ScaledDotProductAttention(jointAttnOut, jointQ, jointK, jointV, sdpaMask, scale);
        jointQ.Dispose();
        jointK.Dispose();
        jointV.Dispose();
        sdpaMask?.Dispose();

        // ── 10. Split [txt, img] back ──
        Tensor txtAttnMh = new Tensor(txtMhShape, DType.F32);
        Tensor imgAttnMh = new Tensor(imgMhShape, DType.F32);
        DiTUtils.SplitAlongSeqDimMultiHead(txtAttnMh, imgAttnMh, jointAttnOut, batch, _numHeads, txtSeqLen, imgSeqLen, _headDim);
        jointAttnOut.Dispose();

        // ── 11. Reshape back to [B, S, hidden] ──
        Tensor imgAttn = new Tensor(imgShape, DType.F32);
        DiTUtils.ReshapeFromMultiHead(imgAttn, imgAttnMh, batch, imgSeqLen, _numHeads, _headDim);
        imgAttnMh.Dispose();

        Tensor txtAttn = new Tensor(txtShape, DType.F32);
        DiTUtils.ReshapeFromMultiHead(txtAttn, txtAttnMh, batch, txtSeqLen, _numHeads, _headDim);
        txtAttnMh.Dispose();

        // ── 12. Output projections + gated residual ──
        Tensor imgAttnProj = new Tensor(imgShape, DType.F32);
        backend.Linear(imgAttnProj, imgAttn, _toOutWeight!, _toOutBias);
        imgAttn.Dispose();
        Tensor imgAfterAttn = AdaLNModulation.ApplyGatedResidual(image, imgAttnProj, imgMod[2], batch, imgSeqLen, _hiddenSize);
        imgAttnProj.Dispose();

        Tensor txtAttnProj = new Tensor(txtShape, DType.F32);
        backend.Linear(txtAttnProj, txtAttn, _toAddOutWeight!, _toAddOutBias);
        txtAttn.Dispose();
        Tensor txtAfterAttn = AdaLNModulation.ApplyGatedResidual(text, txtAttnProj, txtMod[2], batch, txtSeqLen, _hiddenSize);
        txtAttnProj.Dispose();

        // ── 13. Image MLP path ──
        Tensor imgMlpNormed = new Tensor(imgShape, DType.F32);
        DiTUtils.LayerNormNoAffine(imgMlpNormed, imgAfterAttn, batch, imgSeqLen, _hiddenSize);
        Tensor imgMlpModulated = AdaLNModulation.ApplyModulation(imgMlpNormed, imgMod[3], imgMod[4], batch, imgSeqLen, _hiddenSize);
        imgMlpNormed.Dispose();
        Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();
        Tensor imgFinal = AdaLNModulation.ApplyGatedResidual(imgAfterAttn, imgMlpOut, imgMod[5], batch, imgSeqLen, _hiddenSize);
        imgMlpOut.Dispose();
        imgAfterAttn.Dispose();

        // ── 14. Text MLP path ──
        Tensor txtMlpNormed = new Tensor(txtShape, DType.F32);
        DiTUtils.LayerNormNoAffine(txtMlpNormed, txtAfterAttn, batch, txtSeqLen, _hiddenSize);
        Tensor txtMlpModulated = AdaLNModulation.ApplyModulation(txtMlpNormed, txtMod[3], txtMod[4], batch, txtSeqLen, _hiddenSize);
        txtMlpNormed.Dispose();
        Tensor txtMlpOut = _txtFfn.Forward(backend, txtMlpModulated, batch, txtSeqLen);
        txtMlpModulated.Dispose();
        Tensor txtFinal = AdaLNModulation.ApplyGatedResidual(txtAfterAttn, txtMlpOut, txtMod[5], batch, txtSeqLen, _hiddenSize);
        txtMlpOut.Dispose();
        txtAfterAttn.Dispose();

        // Dispose modulation slices
        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < txtMod.Length; i++) txtMod[i].Dispose();

        return (imgFinal, txtFinal);
    }

    /// <summary>Slices <paramref name="rowCount"/> consecutive rows out of a <c>[B, K, hidden]</c> modulation
    /// table starting at <paramref name="rowStart"/>, returning each row as its own <c>[B, hidden]</c> tensor.
    /// Lets <see cref="AdaLNModulation.ApplyModulation"/> / <see cref="AdaLNModulation.ApplyGatedResidual"/>
    /// consume them with the same per-batch broadcast semantics as Flux's modulation outputs.</summary>
    private Tensor[] SliceModRows(Tensor temb, int batch, int rowStart, int rowCount)
    {
        int totalRows = (int)temb.Shape[1];
        int hidden = (int)temb.Shape[2];
        Tensor[] rows = new Tensor[rowCount];
        float* tembPtr = (float*)temb.DataPointer;

        for (int r = 0; r < rowCount; r++)
        {
            TensorShape rowShape = new TensorShape(batch, hidden);
            Tensor row = new Tensor(rowShape, DType.F32);
            float* rowPtr = (float*)row.DataPointer;

            for (int b = 0; b < batch; b++)
            {
                long src = ((long)b * totalRows + (rowStart + r)) * hidden;
                long dst = (long)b * hidden;
                Buffer.MemoryCopy(tembPtr + src, rowPtr + dst, hidden * sizeof(float), hidden * sizeof(float));
            }
            rows[r] = row;
        }
        return rows;
    }

    /// <summary>Builds an additive SDPA mask <c>[B, 1, S, S]</c> from a per-token boolean mask <c>[B, S]</c>.
    /// Diffusers does <c>mask[:, None, None, :] * mask[:, None, :, None]</c> producing a 0/1 outer-product mask;
    /// we reproduce the same semantics by writing 0 where attention is allowed and a large negative value where
    /// it must be killed (additive-mask convention used by <see cref="IBackend.ScaledDotProductAttention"/>).</summary>
    private static Tensor BuildSdpaMask(Tensor mask, int batch, int seqLen, int numHeads)
    {
        // Mask is [B, S]; build [B, 1, S, S] (broadcast to all heads). The backend's SDPA expects an additive mask.
        TensorShape outShape = new TensorShape(batch, 1, seqLen, seqLen);
        Tensor outMask = new Tensor(outShape, DType.F32);

        float* mPtr = (float*)mask.DataPointer;
        float* outPtr = (float*)outMask.DataPointer;
        const float NegInf = -1.0e30f;

        for (int b = 0; b < batch; b++)
        {
            int maskOffset = b * seqLen;
            int outOffset = b * seqLen * seqLen;
            for (int q = 0; q < seqLen; q++)
            {
                float qKeep = mPtr[maskOffset + q];
                for (int k = 0; k < seqLen; k++)
                {
                    float kKeep = mPtr[maskOffset + k];
                    float allowed = qKeep * kKeep;
                    outPtr[outOffset + q * seqLen + k] = allowed > 0.5f ? 0.0f : NegInf;
                }
            }
        }
        return outMask;
    }
}

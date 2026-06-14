using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Flux DoubleStreamBlock: dual-stream joint attention between image and text tokens with RoPE. Similar to SD3 JointBlock but adds RoPE before attention and uses GELU(tanh) MLP.</summary>
public sealed unsafe class FluxDoubleStreamBlock
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
    public (Tensor image, Tensor text) Forward(IBackend backend, Tensor image, Tensor text, Tensor temb, FluxRope rope)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];
        int totalSeqLen = imgSeqLen + txtSeqLen;

        // ── 1. AdaLN modulation ──
        Tensor[] imgMod = _imgModulation.Forward(backend, temb);
        Tensor[] txtMod = _txtModulation.Forward(backend, temb);

        // ── 2. LayerNorm (no affine) + modulate ──
        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        Tensor imgNormed = new Tensor(imgShape, DType.F32);
        LayerNormNoAffine(imgNormed, image, batch, imgSeqLen, _hiddenSize);
        Tensor imgModulated = AdaLNModulation.ApplyModulation(imgNormed, imgMod[0], imgMod[1], batch, imgSeqLen, _hiddenSize);
        imgNormed.Dispose();

        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);
        Tensor txtNormed = new Tensor(txtShape, DType.F32);
        LayerNormNoAffine(txtNormed, text, batch, txtSeqLen, _hiddenSize);
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
        ReshapeToMultiHead(imgQMh, imgQNormed, batch, imgSeqLen, _numHeads, _headDim);
        ReshapeToMultiHead(imgKMh, imgKNormed, batch, imgSeqLen, _numHeads, _headDim);
        ReshapeToMultiHead(imgVMh, imgV, batch, imgSeqLen, _numHeads, _headDim);
        imgQNormed.Dispose();
        imgKNormed.Dispose();
        imgV.Dispose();

        Tensor txtQMh = new Tensor(txtMhShape, DType.F32);
        Tensor txtKMh = new Tensor(txtMhShape, DType.F32);
        Tensor txtVMh = new Tensor(txtMhShape, DType.F32);
        ReshapeToMultiHead(txtQMh, txtQNormed, batch, txtSeqLen, _numHeads, _headDim);
        ReshapeToMultiHead(txtKMh, txtKNormed, batch, txtSeqLen, _numHeads, _headDim);
        ReshapeToMultiHead(txtVMh, txtV, batch, txtSeqLen, _numHeads, _headDim);
        txtQNormed.Dispose();
        txtKNormed.Dispose();
        txtV.Dispose();

        // ── 6. Concatenate [txt, img] for joint attention ──
        Tensor jointQ = new Tensor(jointMhShape, DType.F32);
        Tensor jointK = new Tensor(jointMhShape, DType.F32);
        Tensor jointV = new Tensor(jointMhShape, DType.F32);
        ConcatAlongSeqDim(jointQ, txtQMh, imgQMh, batch, _numHeads, txtSeqLen, imgSeqLen, _headDim);
        ConcatAlongSeqDim(jointK, txtKMh, imgKMh, batch, _numHeads, txtSeqLen, imgSeqLen, _headDim);
        ConcatAlongSeqDim(jointV, txtVMh, imgVMh, batch, _numHeads, txtSeqLen, imgSeqLen, _headDim);
        txtQMh.Dispose(); imgQMh.Dispose();
        txtKMh.Dispose(); imgKMh.Dispose();
        txtVMh.Dispose(); imgVMh.Dispose();

        // ── 7. Apply RoPE to concatenated Q and K ──
        rope.Forward(jointQ, jointK, batch, _numHeads, totalSeqLen);

        // ── 8. Scaled dot-product attention ──
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor jointAttnOut = new Tensor(jointMhShape, DType.F32);
        backend.ScaledDotProductAttention(jointAttnOut, jointQ, jointK, jointV, null, scale);
        jointQ.Dispose();
        jointK.Dispose();
        jointV.Dispose();

        // ── 9. Split attention output back to [txt, img] ──
        Tensor txtAttnMh = new Tensor(txtMhShape, DType.F32);
        Tensor imgAttnMh = new Tensor(imgMhShape, DType.F32);
        SplitAlongSeqDim(txtAttnMh, imgAttnMh, jointAttnOut, batch, _numHeads, txtSeqLen, imgSeqLen, _headDim);
        jointAttnOut.Dispose();

        // ── 10. Reshape back to [B, S, hidden] ──
        Tensor imgAttn = new Tensor(imgShape, DType.F32);
        ReshapeFromMultiHead(imgAttn, imgAttnMh, batch, imgSeqLen, _numHeads, _headDim);
        imgAttnMh.Dispose();

        Tensor txtAttn = new Tensor(txtShape, DType.F32);
        ReshapeFromMultiHead(txtAttn, txtAttnMh, batch, txtSeqLen, _numHeads, _headDim);
        txtAttnMh.Dispose();

        // ── 11. Output projections + gated residual ──
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

        // ── 12. Image MLP: LayerNorm + modulate + GELU MLP + gated residual ──
        Tensor imgMlpNormed = new Tensor(imgShape, DType.F32);
        LayerNormNoAffine(imgMlpNormed, imgAfterAttn, batch, imgSeqLen, _hiddenSize);
        Tensor imgMlpModulated = AdaLNModulation.ApplyModulation(imgMlpNormed, imgMod[3], imgMod[4], batch, imgSeqLen, _hiddenSize);
        imgMlpNormed.Dispose();
        Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();
        Tensor imgFinal = AdaLNModulation.ApplyGatedResidual(imgAfterAttn, imgMlpOut, imgMod[5], batch, imgSeqLen, _hiddenSize);
        imgMlpOut.Dispose();
        imgAfterAttn.Dispose();

        // ── 13. Text MLP: LayerNorm + modulate + GELU MLP + gated residual ──
        Tensor txtMlpNormed = new Tensor(txtShape, DType.F32);
        LayerNormNoAffine(txtMlpNormed, txtAfterAttn, batch, txtSeqLen, _hiddenSize);
        Tensor txtMlpModulated = AdaLNModulation.ApplyModulation(txtMlpNormed, txtMod[3], txtMod[4], batch, txtSeqLen, _hiddenSize);
        txtMlpNormed.Dispose();
        Tensor txtMlpOut = _txtFfn.Forward(backend, txtMlpModulated, batch, txtSeqLen);
        txtMlpModulated.Dispose();
        Tensor txtFinal = AdaLNModulation.ApplyGatedResidual(txtAfterAttn, txtMlpOut, txtMod[5], batch, txtSeqLen, _hiddenSize);
        txtMlpOut.Dispose();
        txtAfterAttn.Dispose();

        // Dispose modulation tensors
        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < txtMod.Length; i++) txtMod[i].Dispose();

        return (imgFinal, txtFinal);
    }

    // ── Helper methods (same as JointBlock) ──

    private static void LayerNormNoAffine(Tensor output, Tensor input, int batch, int seqLen, int dim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int offset = (b * seqLen + s) * dim;

                float mean = 0f;
                for (int d = 0; d < dim; d++)
                    mean += inPtr[offset + d];
                mean /= dim;

                float variance = 0f;
                for (int d = 0; d < dim; d++)
                {
                    float diff = inPtr[offset + d] - mean;
                    variance += diff * diff;
                }
                variance /= dim;

                float invStd = 1.0f / MathF.Sqrt(variance + 1e-6f);
                for (int d = 0; d < dim; d++)
                    outPtr[offset + d] = (inPtr[offset + d] - mean) * invStd;
            }
        }
    }

    private static void ReshapeToMultiHead(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int inOffset = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    int outOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    Buffer.MemoryCopy(inPtr + inOffset, outPtr + outOffset, headDim * sizeof(float), headDim * sizeof(float));
                }
            }
        }
    }

    private static void ReshapeFromMultiHead(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int inOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    int outOffset = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    Buffer.MemoryCopy(inPtr + inOffset, outPtr + outOffset, headDim * sizeof(float), headDim * sizeof(float));
                }
            }
        }
    }

    private static void ConcatAlongSeqDim(Tensor output, Tensor first, Tensor second,
        int batch, int numHeads, int firstSeqLen, int secondSeqLen, int headDim)
    {
        float* firstPtr = (float*)first.DataPointer;
        float* secondPtr = (float*)second.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int totalSeqLen = firstSeqLen + secondSeqLen;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int outBase = (b * numHeads + h) * totalSeqLen * headDim;
                int firstBase = (b * numHeads + h) * firstSeqLen * headDim;
                int secondBase = (b * numHeads + h) * secondSeqLen * headDim;

                long firstBytes = (long)firstSeqLen * headDim * sizeof(float);
                Buffer.MemoryCopy(firstPtr + firstBase, outPtr + outBase, firstBytes, firstBytes);

                long secondBytes = (long)secondSeqLen * headDim * sizeof(float);
                Buffer.MemoryCopy(secondPtr + secondBase, outPtr + outBase + firstSeqLen * headDim, secondBytes, secondBytes);
            }
        }
    }

    private static void SplitAlongSeqDim(Tensor first, Tensor second, Tensor input,
        int batch, int numHeads, int firstSeqLen, int secondSeqLen, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* firstPtr = (float*)first.DataPointer;
        float* secondPtr = (float*)second.DataPointer;
        int totalSeqLen = firstSeqLen + secondSeqLen;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int inBase = (b * numHeads + h) * totalSeqLen * headDim;
                int firstBase = (b * numHeads + h) * firstSeqLen * headDim;
                int secondBase = (b * numHeads + h) * secondSeqLen * headDim;

                long firstBytes = (long)firstSeqLen * headDim * sizeof(float);
                Buffer.MemoryCopy(inPtr + inBase, firstPtr + firstBase, firstBytes, firstBytes);

                long secondBytes = (long)secondSeqLen * headDim * sizeof(float);
                Buffer.MemoryCopy(inPtr + inBase + firstSeqLen * headDim, secondPtr + secondBase, secondBytes, secondBytes);
            }
        }
    }
}

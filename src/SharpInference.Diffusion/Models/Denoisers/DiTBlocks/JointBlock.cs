using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>SD3 JointTransformerBlock: symmetric dual-stream attention between image and text tokens. Both streams share a single concatenated attention operation but maintain separate projection weights. The final block uses context_pre_only mode where context contributes Q/K/V but receives no output.</summary>
public sealed unsafe class JointBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _ffDim;
    private readonly bool _isPreOnly;
    private readonly bool _useQkNorm;

    // Image AdaLN-Zero: SiLU + Linear → 6 params (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
    private readonly AdaLNModulation _imgModulation;

    // Context AdaLN: 6 params for normal blocks, 2 params (shift, scale) for pre_only
    private readonly AdaLNModulation _ctxModulation;

    // Image attention projections
    private Tensor? _toQWeight, _toQBias;
    private Tensor? _toKWeight, _toKBias;
    private Tensor? _toVWeight, _toVBias;
    private Tensor? _toOutWeight, _toOutBias;

    // Context attention projections (separate from image)
    private Tensor? _addQWeight, _addQBias;
    private Tensor? _addKWeight, _addKBias;
    private Tensor? _addVWeight, _addVBias;
    private Tensor? _toAddOutWeight, _toAddOutBias;

    // Optional QK-norm (separate for image and context Q/K)
    private QkNorm? _normQ, _normK, _normAddedQ, _normAddedK;

    // Image MLP: LayerNorm (unparameterized in diffusers, may have weight/bias in some formats) + FFN
    private Tensor? _norm2Weight, _norm2Bias;
    private bool _norm2HasAffine;
    private readonly SwiGluFfn _imgFfn;

    // Context MLP: LayerNorm + FFN (null for pre_only)
    private Tensor? _norm2CtxWeight, _norm2CtxBias;
    private bool _norm2CtxHasAffine;
    private readonly SwiGluFfn? _ctxFfn;

    /// <summary>Creates a JointTransformerBlock.</summary>
    /// <param name="hiddenSize">Model hidden dimension (1536 for SD3 Medium).</param>
    /// <param name="numHeads">Number of attention heads (24 for SD3 Medium).</param>
    /// <param name="ffDim">Feed-forward inner dimension (typically 4 * hiddenSize).</param>
    /// <param name="isPreOnly">True for the final block where context receives no output.</param>
    /// <param name="useQkNorm">Whether to apply RMSNorm to Q/K before attention.</param>
    /// <param name="qkNormEps">QK-norm epsilon.</param>
    public JointBlock(int hiddenSize, int numHeads, int ffDim, bool isPreOnly, bool useQkNorm, float qkNormEps = 1e-6f)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = hiddenSize / numHeads;
        _ffDim = ffDim;
        _isPreOnly = isPreOnly;
        _useQkNorm = useQkNorm;

        _imgModulation = new AdaLNModulation(hiddenSize, 6);
        _ctxModulation = new AdaLNModulation(hiddenSize, isPreOnly ? 2 : 6);

        if (useQkNorm)
        {
            _normQ = new QkNorm(_headDim, qkNormEps);
            _normK = new QkNorm(_headDim, qkNormEps);
            _normAddedQ = new QkNorm(_headDim, qkNormEps);
            _normAddedK = new QkNorm(_headDim, qkNormEps);
        }

        _imgFfn = new SwiGluFfn(hiddenSize, ffDim);
        _ctxFfn = isPreOnly ? null : new SwiGluFfn(hiddenSize, ffDim);
    }

    /// <summary>Loads weights from named tensors using HuggingFace diffusers naming convention.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        // AdaLN modulation weights
        _imgModulation.LoadWeights(
            weights[$"{prefix}.norm1.linear.weight"],
            weights[$"{prefix}.norm1.linear.bias"]);

        _ctxModulation.LoadWeights(
            weights[$"{prefix}.norm1_context.linear.weight"],
            weights[$"{prefix}.norm1_context.linear.bias"]);

        // Image attention projections
        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toQBias = weights[$"{prefix}.attn.to_q.bias"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toKBias = weights[$"{prefix}.attn.to_k.bias"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _toVBias = weights[$"{prefix}.attn.to_v.bias"];
        _toOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];
        _toOutBias = weights[$"{prefix}.attn.to_out.0.bias"];

        // Context attention projections
        _addQWeight = weights[$"{prefix}.attn.add_q_proj.weight"];
        _addQBias = weights[$"{prefix}.attn.add_q_proj.bias"];
        _addKWeight = weights[$"{prefix}.attn.add_k_proj.weight"];
        _addKBias = weights[$"{prefix}.attn.add_k_proj.bias"];
        _addVWeight = weights[$"{prefix}.attn.add_v_proj.weight"];
        _addVBias = weights[$"{prefix}.attn.add_v_proj.bias"];
        _toAddOutWeight = weights[$"{prefix}.attn.to_add_out.weight"];
        _toAddOutBias = weights[$"{prefix}.attn.to_add_out.bias"];

        // QK-norm (optional)
        if (_useQkNorm)
        {
            _normQ!.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
            _normK!.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);
            _normAddedQ!.LoadWeights(weights[$"{prefix}.attn.norm_added_q.weight"]);
            _normAddedK!.LoadWeights(weights[$"{prefix}.attn.norm_added_k.weight"]);
        }

        // Image MLP norm (optional affine — diffusers SD3 uses elementwise_affine=False)
        _norm2HasAffine = weights.TryGetValue($"{prefix}.norm2.weight", out _norm2Weight)
                        & weights.TryGetValue($"{prefix}.norm2.bias", out _norm2Bias);

        // FFN: detect GELU vs SwiGLU from weight shapes
        Tensor ffProjWeight = weights[$"{prefix}.ff.net.0.proj.weight"];
        Tensor ffProjBias = weights[$"{prefix}.ff.net.0.proj.bias"];
        Tensor ffOutWeight = weights[$"{prefix}.ff.net.2.weight"];
        Tensor ffOutBias = weights[$"{prefix}.ff.net.2.bias"];
        _imgFfn.LoadGeluWeights(ffProjWeight, ffProjBias, ffOutWeight, ffOutBias);

        // Context MLP (absent for pre_only)
        if (!_isPreOnly)
        {
            _norm2CtxHasAffine = weights.TryGetValue($"{prefix}.norm2_context.weight", out _norm2CtxWeight)
                               & weights.TryGetValue($"{prefix}.norm2_context.bias", out _norm2CtxBias);

            Tensor ctxFfProjWeight = weights[$"{prefix}.ff_context.net.0.proj.weight"];
            Tensor ctxFfProjBias = weights[$"{prefix}.ff_context.net.0.proj.bias"];
            Tensor ctxFfOutWeight = weights[$"{prefix}.ff_context.net.2.weight"];
            Tensor ctxFfOutBias = weights[$"{prefix}.ff_context.net.2.bias"];
            _ctxFfn!.LoadGeluWeights(ctxFfProjWeight, ctxFfProjBias, ctxFfOutWeight, ctxFfOutBias);
        }
    }

    /// <summary>Forward pass through the joint block. Both streams share attention, then run independent MLPs.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="image">Image tokens [B, N_img, hidden].</param>
    /// <param name="context">Context tokens [B, N_ctx, hidden].</param>
    /// <param name="temb">Timestep embedding [B, hidden].</param>
    /// <returns>Updated (image, context) tensors. For pre_only blocks, context is returned unchanged.</returns>
    public (Tensor image, Tensor context) Forward(IBackend backend, Tensor image, Tensor context, Tensor temb)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int ctxSeqLen = (int)context.Shape[1];
        int totalSeqLen = imgSeqLen + ctxSeqLen;

        // ── 1. AdaLN modulation ──────────────────────────────────────────
        Tensor[] imgMod = _imgModulation.Forward(backend, temb);
        // imgMod: [shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp]

        Tensor[] ctxMod = _ctxModulation.Forward(backend, temb);
        // For non-pre_only: [shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp]
        // For pre_only: [shift_attn, scale_attn]

        // ── 2. LayerNorm (unparameterized) + modulate ────────────────────
        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        Tensor imgNormed = new Tensor(imgShape, DType.F32);
        LayerNormNoAffine(imgNormed, image, batch, imgSeqLen, _hiddenSize);
        Tensor imgModulated = AdaLNModulation.ApplyModulation(imgNormed, imgMod[0], imgMod[1], batch, imgSeqLen, _hiddenSize);
        imgNormed.Dispose();

        TensorShape ctxShape = new TensorShape(batch, ctxSeqLen, _hiddenSize);
        Tensor ctxNormed = new Tensor(ctxShape, DType.F32);
        LayerNormNoAffine(ctxNormed, context, batch, ctxSeqLen, _hiddenSize);
        Tensor ctxModulated = AdaLNModulation.ApplyModulation(ctxNormed, ctxMod[0], ctxMod[1], batch, ctxSeqLen, _hiddenSize);
        ctxNormed.Dispose();

        // ── 3. Q/K/V projections ─────────────────────────────────────────
        Tensor imgQ = LinearProject(imgModulated, _toQWeight!, _toQBias!, batch, imgSeqLen, _hiddenSize, _hiddenSize);
        Tensor imgK = LinearProject(imgModulated, _toKWeight!, _toKBias!, batch, imgSeqLen, _hiddenSize, _hiddenSize);
        Tensor imgV = LinearProject(imgModulated, _toVWeight!, _toVBias!, batch, imgSeqLen, _hiddenSize, _hiddenSize);
        imgModulated.Dispose();

        Tensor ctxQ = LinearProject(ctxModulated, _addQWeight!, _addQBias!, batch, ctxSeqLen, _hiddenSize, _hiddenSize);
        Tensor ctxK = LinearProject(ctxModulated, _addKWeight!, _addKBias!, batch, ctxSeqLen, _hiddenSize, _hiddenSize);
        Tensor ctxV = LinearProject(ctxModulated, _addVWeight!, _addVBias!, batch, ctxSeqLen, _hiddenSize, _hiddenSize);
        ctxModulated.Dispose();

        // ── 4. Reshape to multi-head [B, S, H*D] → [B, S, H, D] ────────
        // (keeping as flat [B*S*H, D] for QK-norm, then reshaping to [B, H, S, D] for SDPA)

        // ── 5. Optional QK-norm (per-head RMSNorm) ──────────────────────
        if (_useQkNorm)
        {
            int imgVectors = batch * imgSeqLen * _numHeads;
            int ctxVectors = batch * ctxSeqLen * _numHeads;

            Tensor imgQNormed = new Tensor(imgQ.Shape, DType.F32);
            Tensor imgKNormed = new Tensor(imgK.Shape, DType.F32);
            _normQ!.Forward(imgQNormed, imgQ, imgVectors);
            _normK!.Forward(imgKNormed, imgK, imgVectors);
            imgQ.Dispose();
            imgK.Dispose();
            imgQ = imgQNormed;
            imgK = imgKNormed;

            Tensor ctxQNormed = new Tensor(ctxQ.Shape, DType.F32);
            Tensor ctxKNormed = new Tensor(ctxK.Shape, DType.F32);
            _normAddedQ!.Forward(ctxQNormed, ctxQ, ctxVectors);
            _normAddedK!.Forward(ctxKNormed, ctxK, ctxVectors);
            ctxQ.Dispose();
            ctxK.Dispose();
            ctxQ = ctxQNormed;
            ctxK = ctxKNormed;
        }

        // ── 6. Reshape to multi-head 4D for SDPA ────────────────────────
        TensorShape imgMhShape = new TensorShape(batch, _numHeads, imgSeqLen, _headDim);
        TensorShape ctxMhShape = new TensorShape(batch, _numHeads, ctxSeqLen, _headDim);
        TensorShape jointMhShape = new TensorShape(batch, _numHeads, totalSeqLen, _headDim);

        Tensor imgQMh = new Tensor(imgMhShape, DType.F32);
        Tensor imgKMh = new Tensor(imgMhShape, DType.F32);
        Tensor imgVMh = new Tensor(imgMhShape, DType.F32);
        ReshapeToMultiHead(imgQMh, imgQ, batch, imgSeqLen, _numHeads, _headDim);
        ReshapeToMultiHead(imgKMh, imgK, batch, imgSeqLen, _numHeads, _headDim);
        ReshapeToMultiHead(imgVMh, imgV, batch, imgSeqLen, _numHeads, _headDim);
        imgQ.Dispose();
        imgK.Dispose();
        imgV.Dispose();

        Tensor ctxQMh = new Tensor(ctxMhShape, DType.F32);
        Tensor ctxKMh = new Tensor(ctxMhShape, DType.F32);
        Tensor ctxVMh = new Tensor(ctxMhShape, DType.F32);
        ReshapeToMultiHead(ctxQMh, ctxQ, batch, ctxSeqLen, _numHeads, _headDim);
        ReshapeToMultiHead(ctxKMh, ctxK, batch, ctxSeqLen, _numHeads, _headDim);
        ReshapeToMultiHead(ctxVMh, ctxV, batch, ctxSeqLen, _numHeads, _headDim);
        ctxQ.Dispose();
        ctxK.Dispose();
        ctxV.Dispose();

        // ── 7. Concatenate for joint attention: [ctx, img] along seq dim ─
        Tensor jointQ = new Tensor(jointMhShape, DType.F32);
        Tensor jointK = new Tensor(jointMhShape, DType.F32);
        Tensor jointV = new Tensor(jointMhShape, DType.F32);
        ConcatAlongSeqDim(jointQ, ctxQMh, imgQMh, batch, _numHeads, ctxSeqLen, imgSeqLen, _headDim);
        ConcatAlongSeqDim(jointK, ctxKMh, imgKMh, batch, _numHeads, ctxSeqLen, imgSeqLen, _headDim);
        ConcatAlongSeqDim(jointV, ctxVMh, imgVMh, batch, _numHeads, ctxSeqLen, imgSeqLen, _headDim);
        ctxQMh.Dispose(); imgQMh.Dispose();
        ctxKMh.Dispose(); imgKMh.Dispose();
        ctxVMh.Dispose(); imgVMh.Dispose();

        // ── 8. Scaled dot-product attention ──────────────────────────────
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor jointAttnOut = new Tensor(jointMhShape, DType.F32);
        backend.ScaledDotProductAttention(jointAttnOut, jointQ, jointK, jointV, null, scale);
        jointQ.Dispose();
        jointK.Dispose();
        jointV.Dispose();

        // ── 9. Split attention output back to [ctx, img] ────────────────
        Tensor ctxAttnMh = new Tensor(ctxMhShape, DType.F32);
        Tensor imgAttnMh = new Tensor(imgMhShape, DType.F32);
        SplitAlongSeqDim(ctxAttnMh, imgAttnMh, jointAttnOut, batch, _numHeads, ctxSeqLen, imgSeqLen, _headDim);
        jointAttnOut.Dispose();

        // ── 10. Reshape back to [B, S, hidden] ──────────────────────────
        Tensor imgAttn = new Tensor(imgShape, DType.F32);
        ReshapeFromMultiHead(imgAttn, imgAttnMh, batch, imgSeqLen, _numHeads, _headDim);
        imgAttnMh.Dispose();

        Tensor ctxAttn = new Tensor(ctxShape, DType.F32);
        ReshapeFromMultiHead(ctxAttn, ctxAttnMh, batch, ctxSeqLen, _numHeads, _headDim);
        ctxAttnMh.Dispose();

        // ── 11. Output projections ───────────────────────────────────────
        Tensor imgAttnProj = LinearProject(imgAttn, _toOutWeight!, _toOutBias!, batch, imgSeqLen, _hiddenSize, _hiddenSize);
        imgAttn.Dispose();

        // ── 12. Gated residual for image attention ───────────────────────
        Tensor imgAfterAttn = AdaLNModulation.ApplyGatedResidual(image, imgAttnProj, imgMod[2], batch, imgSeqLen, _hiddenSize);
        imgAttnProj.Dispose();

        // ── 13. Context attention output + gated residual ────────────────
        Tensor ctxAfterAttn;
        if (!_isPreOnly)
        {
            Tensor ctxAttnProj = LinearProject(ctxAttn, _toAddOutWeight!, _toAddOutBias!, batch, ctxSeqLen, _hiddenSize, _hiddenSize);
            ctxAttn.Dispose();
            ctxAfterAttn = AdaLNModulation.ApplyGatedResidual(context, ctxAttnProj, ctxMod[2], batch, ctxSeqLen, _hiddenSize);
            ctxAttnProj.Dispose();
        }
        else
        {
            ctxAttn.Dispose();
            ctxAfterAttn = context;
        }

        // ── 14. Image MLP ────────────────────────────────────────────────
        Tensor imgMlpNormed = new Tensor(imgShape, DType.F32);
        if (_norm2HasAffine)
            backend.LayerNorm(imgMlpNormed, imgAfterAttn, _norm2Weight!, _norm2Bias!, 1e-6f);
        else
            LayerNormNoAffine(imgMlpNormed, imgAfterAttn, batch, imgSeqLen, _hiddenSize);
        Tensor imgMlpModulated = AdaLNModulation.ApplyModulation(imgMlpNormed, imgMod[3], imgMod[4], batch, imgSeqLen, _hiddenSize);
        imgMlpNormed.Dispose();

        Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();

        Tensor imgFinal = AdaLNModulation.ApplyGatedResidual(imgAfterAttn, imgMlpOut, imgMod[5], batch, imgSeqLen, _hiddenSize);
        imgMlpOut.Dispose();
        imgAfterAttn.Dispose();

        // ── 15. Context MLP (skip for pre_only) ─────────────────────────
        Tensor ctxFinal;
        if (!_isPreOnly)
        {
            Tensor ctxMlpNormed = new Tensor(ctxShape, DType.F32);
            if (_norm2CtxHasAffine)
                backend.LayerNorm(ctxMlpNormed, ctxAfterAttn, _norm2CtxWeight!, _norm2CtxBias!, 1e-6f);
            else
                LayerNormNoAffine(ctxMlpNormed, ctxAfterAttn, batch, ctxSeqLen, _hiddenSize);
            Tensor ctxMlpModulated = AdaLNModulation.ApplyModulation(ctxMlpNormed, ctxMod[3], ctxMod[4], batch, ctxSeqLen, _hiddenSize);
            ctxMlpNormed.Dispose();

            Tensor ctxMlpOut = _ctxFfn!.Forward(backend, ctxMlpModulated, batch, ctxSeqLen);
            ctxMlpModulated.Dispose();

            ctxFinal = AdaLNModulation.ApplyGatedResidual(ctxAfterAttn, ctxMlpOut, ctxMod[5], batch, ctxSeqLen, _hiddenSize);
            ctxMlpOut.Dispose();
            ctxAfterAttn.Dispose();
        }
        else
        {
            ctxFinal = ctxAfterAttn;
        }

        // Dispose modulation tensors
        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < ctxMod.Length; i++) ctxMod[i].Dispose();

        return (imgFinal, ctxFinal);
    }

    // ── Helper methods ───────────────────────────────────────────────────

    /// <summary>Unparameterized LayerNorm: (x - mean) / sqrt(var + eps). No learnable weight/bias — modulation replaces the affine transform.</summary>
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
                {
                    outPtr[offset + d] = (inPtr[offset + d] - mean) * invStd;
                }
            }
        }
    }

    private static Tensor LinearProject(Tensor input, Tensor weight, Tensor bias, int batch, int seqLen, int inDim, int outDim)
    {
        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor output = new Tensor(outShape, DType.F32);

        float* inPtr = (float*)input.DataPointer;
        float* wPtr = (float*)weight.DataPointer;
        float* bPtr = (float*)bias.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int inOffset = (b * seqLen + s) * inDim;
                int outOffset = (b * seqLen + s) * outDim;
                for (int o = 0; o < outDim; o++)
                {
                    float sum = bPtr[o];
                    int wOffset = o * inDim;
                    for (int i = 0; i < inDim; i++)
                    {
                        sum += inPtr[inOffset + i] * wPtr[wOffset + i];
                    }
                    outPtr[outOffset + o] = sum;
                }
            }
        }

        return output;
    }

    /// <summary>Reshape [B, seqLen, numHeads * headDim] → [B, numHeads, seqLen, headDim].</summary>
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

    /// <summary>Reshape [B, numHeads, seqLen, headDim] → [B, seqLen, numHeads * headDim].</summary>
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

    /// <summary>Concatenates two tensors along the sequence dimension in [B, H, S, D] layout.</summary>
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

                // Copy first (context)
                long firstBytes = (long)firstSeqLen * headDim * sizeof(float);
                Buffer.MemoryCopy(firstPtr + firstBase / 1, outPtr + outBase / 1, firstBytes, firstBytes);

                // Copy second (image)
                long secondBytes = (long)secondSeqLen * headDim * sizeof(float);
                int outSecondOffset = outBase + firstSeqLen * headDim;
                Buffer.MemoryCopy(secondPtr + secondBase / 1, outPtr + outSecondOffset / 1, secondBytes, secondBytes);
            }
        }
    }

    /// <summary>Splits a tensor along the sequence dimension in [B, H, S, D] layout.</summary>
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

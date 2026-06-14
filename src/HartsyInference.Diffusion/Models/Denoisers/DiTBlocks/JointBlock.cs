using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>SD3 / SD3.5 JointTransformerBlock. Symmetric dual-stream attention between image and text tokens with separate per-modality projections sharing one concatenated SDPA. SD3.5 MMDiT-X variant adds an optional image-only second self-attention path (`attn2`) gated independently of the joint attention. The final block uses context_pre_only mode where context contributes Q/K/V but receives no output.</summary>
public sealed unsafe class JointBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _ffDim;
    private readonly bool _isPreOnly;
    private readonly bool _useQkNorm;
    private readonly bool _useDualAttention;

    // Image AdaLN: 6 params for normal blocks, 9 params for dual-attn (adds shift_msa2, scale_msa2, gate_msa2)
    private readonly AdaLNModulation _imgModulation;

    // Context AdaLN: 6 params for normal blocks, 2 params (shift, scale) for pre_only
    private readonly AdaLNModulation _ctxModulation;

    // Image attention (joint)
    private Tensor? _toQWeight, _toQBias;
    private Tensor? _toKWeight, _toKBias;
    private Tensor? _toVWeight, _toVBias;
    private Tensor? _toOutWeight, _toOutBias;

    // Context attention (joint)
    private Tensor? _addQWeight, _addQBias;
    private Tensor? _addKWeight, _addKBias;
    private Tensor? _addVWeight, _addVBias;
    private Tensor? _toAddOutWeight, _toAddOutBias;

    // QK-norm (separate for image and context)
    private readonly QkNorm? _normQ, _normK, _normAddedQ, _normAddedK;

    // Dual-attention (image-only second self-attn) — null unless useDualAttention
    private Tensor? _attn2QWeight, _attn2QBias;
    private Tensor? _attn2KWeight, _attn2KBias;
    private Tensor? _attn2VWeight, _attn2VBias;
    private Tensor? _attn2OutWeight, _attn2OutBias;
    private readonly QkNorm? _attn2NormQ, _attn2NormK;

    // Image MLP: optional affine LayerNorm (diffusers SD3 has elementwise_affine=False) + GELU FFN
    private Tensor? _norm2Weight, _norm2Bias;
    private bool _norm2HasAffine;
    private readonly SwiGluFfn _imgFfn;

    // Context MLP (null for pre_only)
    private Tensor? _norm2CtxWeight, _norm2CtxBias;
    private bool _norm2CtxHasAffine;
    private readonly SwiGluFfn? _ctxFfn;

    /// <summary>Creates a JointTransformerBlock.</summary>
    /// <param name="hiddenSize">Model hidden dimension (1536 for SD3 Medium, 2432 for SD3.5 Large).</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="ffDim">Feed-forward inner dimension (typically 4 * hiddenSize).</param>
    /// <param name="isPreOnly">True for the final block where context receives no output.</param>
    /// <param name="useQkNorm">Whether to apply RMSNorm to Q/K before attention. SD3.5 always true.</param>
    /// <param name="useDualAttention">SD3.5 MMDiT-X dual-attention layer: adds an image-only `attn2` path with its own Q/K/V/Out + qk-norm + 3 extra modulation params.</param>
    /// <param name="qkNormEps">QK-norm epsilon.</param>
    public JointBlock(int hiddenSize, int numHeads, int ffDim, bool isPreOnly, bool useQkNorm, bool useDualAttention = false, float qkNormEps = 1e-6f)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = hiddenSize / numHeads;
        _ffDim = ffDim;
        _isPreOnly = isPreOnly;
        _useQkNorm = useQkNorm;
        _useDualAttention = useDualAttention;

        int imgModParams = useDualAttention ? 9 : 6;
        _imgModulation = new AdaLNModulation(hiddenSize, imgModParams);
        _ctxModulation = new AdaLNModulation(hiddenSize, isPreOnly ? 2 : 6);

        if (useQkNorm)
        {
            _normQ = new QkNorm(_headDim, qkNormEps);
            _normK = new QkNorm(_headDim, qkNormEps);
            _normAddedQ = new QkNorm(_headDim, qkNormEps);
            _normAddedK = new QkNorm(_headDim, qkNormEps);
        }

        if (useDualAttention && useQkNorm)
        {
            _attn2NormQ = new QkNorm(_headDim, qkNormEps);
            _attn2NormK = new QkNorm(_headDim, qkNormEps);
        }

        _imgFfn = new SwiGluFfn(hiddenSize, ffDim);
        _ctxFfn = isPreOnly ? null : new SwiGluFfn(hiddenSize, ffDim);
    }

    /// <summary>Loads weights from named tensors using HuggingFace diffusers naming convention.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        // AdaLN modulation
        _imgModulation.LoadWeights(
            weights[$"{prefix}.norm1.linear.weight"],
            weights[$"{prefix}.norm1.linear.bias"]);

        _ctxModulation.LoadWeights(
            weights[$"{prefix}.norm1_context.linear.weight"],
            weights[$"{prefix}.norm1_context.linear.bias"]);

        // Image (joint) attention projections
        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toQBias = weights[$"{prefix}.attn.to_q.bias"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toKBias = weights[$"{prefix}.attn.to_k.bias"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _toVBias = weights[$"{prefix}.attn.to_v.bias"];
        _toOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];
        _toOutBias = weights[$"{prefix}.attn.to_out.0.bias"];

        // Context (joint) attention projections — context still contributes Q/K/V even on pre_only;
        // only the output projection (to_add_out) is omitted on the final block.
        _addQWeight = weights[$"{prefix}.attn.add_q_proj.weight"];
        _addQBias = weights[$"{prefix}.attn.add_q_proj.bias"];
        _addKWeight = weights[$"{prefix}.attn.add_k_proj.weight"];
        _addKBias = weights[$"{prefix}.attn.add_k_proj.bias"];
        _addVWeight = weights[$"{prefix}.attn.add_v_proj.weight"];
        _addVBias = weights[$"{prefix}.attn.add_v_proj.bias"];

        if (!_isPreOnly)
        {
            _toAddOutWeight = weights[$"{prefix}.attn.to_add_out.weight"];
            _toAddOutBias = weights[$"{prefix}.attn.to_add_out.bias"];
        }

        if (_useQkNorm)
        {
            _normQ!.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
            _normK!.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);
            _normAddedQ!.LoadWeights(weights[$"{prefix}.attn.norm_added_q.weight"]);
            _normAddedK!.LoadWeights(weights[$"{prefix}.attn.norm_added_k.weight"]);
        }

        // Dual-attention (SD3.5 MMDiT-X attn2)
        if (_useDualAttention)
        {
            _attn2QWeight = weights[$"{prefix}.attn2.to_q.weight"];
            _attn2QBias = weights[$"{prefix}.attn2.to_q.bias"];
            _attn2KWeight = weights[$"{prefix}.attn2.to_k.weight"];
            _attn2KBias = weights[$"{prefix}.attn2.to_k.bias"];
            _attn2VWeight = weights[$"{prefix}.attn2.to_v.weight"];
            _attn2VBias = weights[$"{prefix}.attn2.to_v.bias"];
            _attn2OutWeight = weights[$"{prefix}.attn2.to_out.0.weight"];
            _attn2OutBias = weights[$"{prefix}.attn2.to_out.0.bias"];

            if (_useQkNorm)
            {
                _attn2NormQ!.LoadWeights(weights[$"{prefix}.attn2.norm_q.weight"]);
                _attn2NormK!.LoadWeights(weights[$"{prefix}.attn2.norm_k.weight"]);
            }
        }

        // Image MLP norm (optional affine — SD3 diffusers uses elementwise_affine=False)
        _norm2HasAffine = weights.TryGetValue($"{prefix}.norm2.weight", out _norm2Weight)
                        & weights.TryGetValue($"{prefix}.norm2.bias", out _norm2Bias);

        // Image FFN: GELU(approximate=tanh) — diffusers SD3 uses gelu, not SwiGLU
        _imgFfn.LoadGeluWeights(
            weights[$"{prefix}.ff.net.0.proj.weight"],
            weights[$"{prefix}.ff.net.0.proj.bias"],
            weights[$"{prefix}.ff.net.2.weight"],
            weights[$"{prefix}.ff.net.2.bias"]);

        // Context MLP (skip for pre_only)
        if (!_isPreOnly)
        {
            _norm2CtxHasAffine = weights.TryGetValue($"{prefix}.norm2_context.weight", out _norm2CtxWeight)
                               & weights.TryGetValue($"{prefix}.norm2_context.bias", out _norm2CtxBias);

            _ctxFfn!.LoadGeluWeights(
                weights[$"{prefix}.ff_context.net.0.proj.weight"],
                weights[$"{prefix}.ff_context.net.0.proj.bias"],
                weights[$"{prefix}.ff_context.net.2.weight"],
                weights[$"{prefix}.ff_context.net.2.bias"]);
        }
    }

    /// <summary>Yields all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in _imgModulation.EnumerateWeights()) yield return w;
        foreach (Tensor w in _ctxModulation.EnumerateWeights()) yield return w;

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

        if (_normQ is not null) foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        if (_normK is not null) foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
        if (_normAddedQ is not null) foreach (Tensor w in _normAddedQ.EnumerateWeights()) yield return w;
        if (_normAddedK is not null) foreach (Tensor w in _normAddedK.EnumerateWeights()) yield return w;

        if (_attn2QWeight is not null) yield return _attn2QWeight;
        if (_attn2QBias is not null) yield return _attn2QBias;
        if (_attn2KWeight is not null) yield return _attn2KWeight;
        if (_attn2KBias is not null) yield return _attn2KBias;
        if (_attn2VWeight is not null) yield return _attn2VWeight;
        if (_attn2VBias is not null) yield return _attn2VBias;
        if (_attn2OutWeight is not null) yield return _attn2OutWeight;
        if (_attn2OutBias is not null) yield return _attn2OutBias;
        if (_attn2NormQ is not null) foreach (Tensor w in _attn2NormQ.EnumerateWeights()) yield return w;
        if (_attn2NormK is not null) foreach (Tensor w in _attn2NormK.EnumerateWeights()) yield return w;

        if (_norm2Weight is not null) yield return _norm2Weight;
        if (_norm2Bias is not null) yield return _norm2Bias;
        foreach (Tensor w in _imgFfn.EnumerateWeights()) yield return w;

        if (_norm2CtxWeight is not null) yield return _norm2CtxWeight;
        if (_norm2CtxBias is not null) yield return _norm2CtxBias;
        if (_ctxFfn is not null) foreach (Tensor w in _ctxFfn.EnumerateWeights()) yield return w;
    }

    /// <summary>Forward pass through the joint block. Image and context streams share one joint attention; in dual-attention layers the image gets a parallel image-only second attention. Then independent MLPs.</summary>
    public (Tensor image, Tensor context) Forward(IBackend backend, Tensor image, Tensor context, Tensor temb)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int ctxSeqLen = (int)context.Shape[1];
        int totalSeqLen = imgSeqLen + ctxSeqLen;

        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        TensorShape ctxShape = new TensorShape(batch, ctxSeqLen, _hiddenSize);
        TensorShape imgMhShape = new TensorShape(batch, _numHeads, imgSeqLen, _headDim);
        TensorShape ctxMhShape = new TensorShape(batch, _numHeads, ctxSeqLen, _headDim);
        TensorShape jointMhShape = new TensorShape(batch, _numHeads, totalSeqLen, _headDim);

        // ── 1. AdaLN modulation ──
        Tensor[] imgMod = _imgModulation.Forward(backend, temb);
        // 6: [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
        // 9: append [shift_msa2, scale_msa2, gate_msa2]
        Tensor[] ctxMod = _ctxModulation.Forward(backend, temb);
        // 6: [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
        // 2 (pre_only): [shift_msa, scale_msa]

        // ── 2. LayerNorm (no affine) on image and context ──
        // The image-stream norm is shared between joint attn and (optional) dual attn2 paths;
        // each path then applies its own modulation.
        Tensor imgNormed = new Tensor(imgShape, DType.F32);
        DiTUtils.LayerNormNoAffine(imgNormed, image, batch, imgSeqLen, _hiddenSize);
        Tensor imgModulated = AdaLNModulation.ApplyModulation(imgNormed, imgMod[0], imgMod[1], batch, imgSeqLen, _hiddenSize);

        Tensor ctxNormed = new Tensor(ctxShape, DType.F32);
        DiTUtils.LayerNormNoAffine(ctxNormed, context, batch, ctxSeqLen, _hiddenSize);
        Tensor ctxModulated = AdaLNModulation.ApplyModulation(ctxNormed, ctxMod[0], ctxMod[1], batch, ctxSeqLen, _hiddenSize);
        ctxNormed.Dispose();


        // ── 3. Joint Q/K/V projections (GPU-routed) ──
        Tensor imgQ = new Tensor(imgShape, DType.F32);
        backend.Linear(imgQ, imgModulated, _toQWeight!, _toQBias);
        Tensor imgK = new Tensor(imgShape, DType.F32);
        backend.Linear(imgK, imgModulated, _toKWeight!, _toKBias);
        Tensor imgV = new Tensor(imgShape, DType.F32);
        backend.Linear(imgV, imgModulated, _toVWeight!, _toVBias);
        imgModulated.Dispose();

        Tensor ctxQ = new Tensor(ctxShape, DType.F32);
        backend.Linear(ctxQ, ctxModulated, _addQWeight!, _addQBias);
        Tensor ctxK = new Tensor(ctxShape, DType.F32);
        backend.Linear(ctxK, ctxModulated, _addKWeight!, _addKBias);
        Tensor ctxV = new Tensor(ctxShape, DType.F32);
        backend.Linear(ctxV, ctxModulated, _addVWeight!, _addVBias);
        ctxModulated.Dispose();

        // ── 4. Optional QK-norm (per-head RMSNorm) ──
        if (_useQkNorm)
        {
            int imgVectors = batch * imgSeqLen * _numHeads;
            int ctxVectors = batch * ctxSeqLen * _numHeads;

            Tensor imgQNormed = new Tensor(imgQ.Shape, DType.F32);
            Tensor imgKNormed = new Tensor(imgK.Shape, DType.F32);
            _normQ!.Forward(imgQNormed, imgQ, imgVectors);
            _normK!.Forward(imgKNormed, imgK, imgVectors);
            imgQ.Dispose(); imgK.Dispose();
            imgQ = imgQNormed; imgK = imgKNormed;

            Tensor ctxQNormed = new Tensor(ctxQ.Shape, DType.F32);
            Tensor ctxKNormed = new Tensor(ctxK.Shape, DType.F32);
            _normAddedQ!.Forward(ctxQNormed, ctxQ, ctxVectors);
            _normAddedK!.Forward(ctxKNormed, ctxK, ctxVectors);
            ctxQ.Dispose(); ctxK.Dispose();
            ctxQ = ctxQNormed; ctxK = ctxKNormed;
        }

        // ── 5. Reshape to multi-head [B, H, S, D] ──
        Tensor imgQMh = new Tensor(imgMhShape, DType.F32);
        Tensor imgKMh = new Tensor(imgMhShape, DType.F32);
        Tensor imgVMh = new Tensor(imgMhShape, DType.F32);
        DiTUtils.ReshapeToMultiHead(imgQMh, imgQ, batch, imgSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(imgKMh, imgK, batch, imgSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(imgVMh, imgV, batch, imgSeqLen, _numHeads, _headDim);
        imgQ.Dispose(); imgK.Dispose(); imgV.Dispose();

        Tensor ctxQMh = new Tensor(ctxMhShape, DType.F32);
        Tensor ctxKMh = new Tensor(ctxMhShape, DType.F32);
        Tensor ctxVMh = new Tensor(ctxMhShape, DType.F32);
        DiTUtils.ReshapeToMultiHead(ctxQMh, ctxQ, batch, ctxSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(ctxKMh, ctxK, batch, ctxSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(ctxVMh, ctxV, batch, ctxSeqLen, _numHeads, _headDim);
        ctxQ.Dispose(); ctxK.Dispose(); ctxV.Dispose();

        // ── 6. Concatenate [ctx, img] for joint attention ──
        Tensor jointQ = new Tensor(jointMhShape, DType.F32);
        Tensor jointK = new Tensor(jointMhShape, DType.F32);
        Tensor jointV = new Tensor(jointMhShape, DType.F32);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointQ, ctxQMh, imgQMh, batch, _numHeads, ctxSeqLen, imgSeqLen, _headDim);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointK, ctxKMh, imgKMh, batch, _numHeads, ctxSeqLen, imgSeqLen, _headDim);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointV, ctxVMh, imgVMh, batch, _numHeads, ctxSeqLen, imgSeqLen, _headDim);
        ctxQMh.Dispose(); imgQMh.Dispose();
        ctxKMh.Dispose(); imgKMh.Dispose();
        ctxVMh.Dispose(); imgVMh.Dispose();

        // ── 7. Joint scaled dot-product attention ──
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor jointAttnOut = new Tensor(jointMhShape, DType.F32);
        backend.ScaledDotProductAttention(jointAttnOut, jointQ, jointK, jointV, null, scale);
        jointQ.Dispose(); jointK.Dispose(); jointV.Dispose();

        // ── 8. Split + reshape back to [B, S, hidden] ──
        Tensor ctxAttnMh = new Tensor(ctxMhShape, DType.F32);
        Tensor imgAttnMh = new Tensor(imgMhShape, DType.F32);
        DiTUtils.SplitAlongSeqDimMultiHead(ctxAttnMh, imgAttnMh, jointAttnOut, batch, _numHeads, ctxSeqLen, imgSeqLen, _headDim);
        jointAttnOut.Dispose();

        Tensor imgAttn = new Tensor(imgShape, DType.F32);
        DiTUtils.ReshapeFromMultiHead(imgAttn, imgAttnMh, batch, imgSeqLen, _numHeads, _headDim);
        imgAttnMh.Dispose();

        Tensor ctxAttn = new Tensor(ctxShape, DType.F32);
        DiTUtils.ReshapeFromMultiHead(ctxAttn, ctxAttnMh, batch, ctxSeqLen, _numHeads, _headDim);
        ctxAttnMh.Dispose();

        // ── 9. Output projections ──
        Tensor imgAttnProj = new Tensor(imgShape, DType.F32);
        backend.Linear(imgAttnProj, imgAttn, _toOutWeight!, _toOutBias);
        imgAttn.Dispose();

        // ── 10. Optional dual-attention (SD3.5 MMDiT-X) ──
        Tensor? attn2Proj = null;
        if (_useDualAttention)
        {
            // Re-modulate the SAME imgNormed with shift_msa2/scale_msa2 (independent from joint)
            Tensor imgModulated2 = AdaLNModulation.ApplyModulation(imgNormed, imgMod[6], imgMod[7], batch, imgSeqLen, _hiddenSize);

            // attn2 Q/K/V (image-only)
            Tensor a2Q = new Tensor(imgShape, DType.F32);
            backend.Linear(a2Q, imgModulated2, _attn2QWeight!, _attn2QBias);
            Tensor a2K = new Tensor(imgShape, DType.F32);
            backend.Linear(a2K, imgModulated2, _attn2KWeight!, _attn2KBias);
            Tensor a2V = new Tensor(imgShape, DType.F32);
            backend.Linear(a2V, imgModulated2, _attn2VWeight!, _attn2VBias);
            imgModulated2.Dispose();

            if (_useQkNorm)
            {
                int imgVectors = batch * imgSeqLen * _numHeads;
                Tensor a2QNormed = new Tensor(a2Q.Shape, DType.F32);
                Tensor a2KNormed = new Tensor(a2K.Shape, DType.F32);
                _attn2NormQ!.Forward(a2QNormed, a2Q, imgVectors);
                _attn2NormK!.Forward(a2KNormed, a2K, imgVectors);
                a2Q.Dispose(); a2K.Dispose();
                a2Q = a2QNormed; a2K = a2KNormed;
            }

            Tensor a2QMh = new Tensor(imgMhShape, DType.F32);
            Tensor a2KMh = new Tensor(imgMhShape, DType.F32);
            Tensor a2VMh = new Tensor(imgMhShape, DType.F32);
            DiTUtils.ReshapeToMultiHead(a2QMh, a2Q, batch, imgSeqLen, _numHeads, _headDim);
            DiTUtils.ReshapeToMultiHead(a2KMh, a2K, batch, imgSeqLen, _numHeads, _headDim);
            DiTUtils.ReshapeToMultiHead(a2VMh, a2V, batch, imgSeqLen, _numHeads, _headDim);
            a2Q.Dispose(); a2K.Dispose(); a2V.Dispose();

            Tensor a2OutMh = new Tensor(imgMhShape, DType.F32);
            backend.ScaledDotProductAttention(a2OutMh, a2QMh, a2KMh, a2VMh, null, scale);
            a2QMh.Dispose(); a2KMh.Dispose(); a2VMh.Dispose();

            Tensor a2Out = new Tensor(imgShape, DType.F32);
            DiTUtils.ReshapeFromMultiHead(a2Out, a2OutMh, batch, imgSeqLen, _numHeads, _headDim);
            a2OutMh.Dispose();

            attn2Proj = new Tensor(imgShape, DType.F32);
            backend.Linear(attn2Proj, a2Out, _attn2OutWeight!, _attn2OutBias);
            a2Out.Dispose();
        }

        imgNormed.Dispose();

        // ── 11. Image gated residual: image + gate_msa * imgAttnProj (+ gate_msa2 * attn2Proj if dual) ──
        Tensor imgAfterAttn = AdaLNModulation.ApplyGatedResidual(image, imgAttnProj, imgMod[2], batch, imgSeqLen, _hiddenSize);
        imgAttnProj.Dispose();

        if (attn2Proj is not null)
        {
            Tensor imgAfterDual = AdaLNModulation.ApplyGatedResidual(imgAfterAttn, attn2Proj, imgMod[8], batch, imgSeqLen, _hiddenSize);
            attn2Proj.Dispose();
            imgAfterAttn.Dispose();
            imgAfterAttn = imgAfterDual;
        }

        // ── 12. Context gated residual (skip for pre_only) ──
        Tensor ctxAfterAttn;
        if (!_isPreOnly)
        {
            Tensor ctxAttnProj = new Tensor(ctxShape, DType.F32);
            backend.Linear(ctxAttnProj, ctxAttn, _toAddOutWeight!, _toAddOutBias);
            ctxAttn.Dispose();
            ctxAfterAttn = AdaLNModulation.ApplyGatedResidual(context, ctxAttnProj, ctxMod[2], batch, ctxSeqLen, _hiddenSize);
            ctxAttnProj.Dispose();
        }
        else
        {
            ctxAttn.Dispose();
            ctxAfterAttn = context;
        }

        // ── 13. Image MLP: norm + modulate + GELU FFN + gated residual ──
        Tensor imgMlpNormed = new Tensor(imgShape, DType.F32);
        if (_norm2HasAffine)
            backend.LayerNorm(imgMlpNormed, imgAfterAttn, _norm2Weight!, _norm2Bias!, 1e-6f);
        else
            DiTUtils.LayerNormNoAffine(imgMlpNormed, imgAfterAttn, batch, imgSeqLen, _hiddenSize);
        Tensor imgMlpModulated = AdaLNModulation.ApplyModulation(imgMlpNormed, imgMod[3], imgMod[4], batch, imgSeqLen, _hiddenSize);
        imgMlpNormed.Dispose();

        Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();

        Tensor imgFinal = AdaLNModulation.ApplyGatedResidual(imgAfterAttn, imgMlpOut, imgMod[5], batch, imgSeqLen, _hiddenSize);
        imgMlpOut.Dispose();
        imgAfterAttn.Dispose();

        // ── 14. Context MLP (skip for pre_only) ──
        Tensor ctxFinal;
        if (!_isPreOnly)
        {
            Tensor ctxMlpNormed = new Tensor(ctxShape, DType.F32);
            if (_norm2CtxHasAffine)
                backend.LayerNorm(ctxMlpNormed, ctxAfterAttn, _norm2CtxWeight!, _norm2CtxBias!, 1e-6f);
            else
                DiTUtils.LayerNormNoAffine(ctxMlpNormed, ctxAfterAttn, batch, ctxSeqLen, _hiddenSize);
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
}

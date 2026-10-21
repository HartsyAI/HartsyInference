using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>SD3 / SD3.5 JointTransformerBlock. Symmetric dual-stream attention between image and text tokens with separate per-modality projections sharing one concatenated SDPA. SD3.5 MMDiT-X variant adds an optional image-only second self-attention path (`attn2`) gated independently of the joint attention. The final block uses context_pre_only mode where context contributes Q/K/V but receives no output.</summary>
public sealed class JointBlock
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

    private JointAttention? _jointAttention;
    private AttentionLayer? _dualAttention;

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

        // A pre-only block's context stream has no output projection: its attention rows are dropped.
        _jointAttention = new JointAttention(
            new JointAttentionConfig { Heads = _numHeads, HeadDim = _headDim, ConcatOrder = [1, 0], AllowF16 = false },
            [Stream(_toQWeight!, _toQBias, _toKWeight!, _toKBias, _toVWeight!, _toVBias, _toOutWeight, _toOutBias, _normQ, _normK),
             Stream(_addQWeight!, _addQBias, _addKWeight!, _addKBias, _addVWeight!, _addVBias, _toAddOutWeight, _toAddOutBias,
                 _normAddedQ, _normAddedK)]);
        _dualAttention = _useDualAttention
            ? Stream(_attn2QWeight!, _attn2QBias, _attn2KWeight!, _attn2KBias, _attn2VWeight!, _attn2VBias, _attn2OutWeight,
                _attn2OutBias, _attn2NormQ, _attn2NormK)
            : null;
    }

    private AttentionLayer Stream(Tensor q, Tensor? qBias, Tensor k, Tensor? kBias, Tensor v, Tensor? vBias, Tensor? output,
        Tensor? outputBias, QkNorm? normQ, QkNorm? normK) => new(
        new AttentionLayerConfig
        {
            Heads = _numHeads, KvHeads = _numHeads, HeadDim = _headDim,
            QkNorm = _useQkNorm ? AttentionQkNorm.PerHead : AttentionQkNorm.None, ComputeDType = DType.F32,
        },
        new AttentionLayerWeights
        {
            Query = q, QueryBias = qBias, Key = k, KeyBias = kBias, Value = v, ValueBias = vBias, Output = output,
            OutputBias = outputBias, QueryNorm = normQ?.Weight, KeyNorm = normK?.Weight, QueryNormEps = normQ?.Eps ?? 0f,
            KeyNormEps = normK?.Eps ?? 0f,
        });

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

        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        TensorShape ctxShape = new TensorShape(batch, ctxSeqLen, _hiddenSize);

        // ── 1. AdaLN modulation ──
        Tensor[] imgMod = _imgModulation.Forward(backend, temb);
        // 6: [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
        // 9: append [shift_msa2, scale_msa2, gate_msa2]
        Tensor[] ctxMod = _ctxModulation.Forward(backend, temb);
        // 6: [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
        // 2 (pre_only): [shift_msa, scale_msa]

        // ── 2. LayerNorm (no affine, eps 1e-6) on image and context + modulate x*(1+scale)+shift ──
        // The image-stream norm is shared between joint attn and (optional) dual attn2 paths;
        // each path then applies its own modulation.
        Tensor imgNormed = new Tensor(imgShape, DType.F32);
        backend.LayerNormNoAffine(imgNormed, image, 1e-6f);
        Tensor imgModulated = DiTUtils.Modulate(backend, imgNormed, imgMod[0], imgMod[1], imgShape);

        Tensor ctxModulated = DiTUtils.NormModulate(backend, context, ctxMod[0], ctxMod[1], ctxShape);

        // ── 3-8. Joint attention over [ctx, img], then the image output projection ──
        Span<Tensor?> attended = [null, null];
        _jointAttention!.Attend(backend, [imgModulated, ctxModulated], attended, new JointCallContext { ReleaseInputs = true });
        Tensor imgAttnProj = _jointAttention.ProjectOutput(backend, 0, attended[0]!);
        Tensor ctxAttn = attended[1]!;

        // ── 9. Optional dual-attention (SD3.5 MMDiT-X) ──
        Tensor? attn2Proj = null;
        if (_useDualAttention)
        {
            // Re-modulate the SAME imgNormed with shift_msa2/scale_msa2 (independent from joint)
            Tensor imgModulated2 = DiTUtils.Modulate(backend, imgNormed, imgMod[6], imgMod[7], imgShape);

            attn2Proj = _dualAttention!.Forward(backend, imgModulated2, imgModulated2, new AttentionCallContext { ReleaseInputs = true });
        }

        imgNormed.Dispose();

        // ── 10. Image gated residual: image + gate_msa * imgAttnProj (+ gate_msa2 * attn2Proj if dual) ──
        Tensor imgAfterAttn = new Tensor(imgShape, DType.F32);
        backend.GatedResidualLastDim(imgAfterAttn, image, imgAttnProj, imgMod[2]);
        imgAttnProj.Dispose();

        if (attn2Proj is not null)
        {
            Tensor imgAfterDual = new Tensor(imgShape, DType.F32);
            backend.GatedResidualLastDim(imgAfterDual, imgAfterAttn, attn2Proj, imgMod[8]);
            attn2Proj.Dispose();
            imgAfterAttn.Dispose();
            imgAfterAttn = imgAfterDual;
        }

        // ── 11. Context gated residual (skip for pre_only) ──
        Tensor ctxAfterAttn;
        if (!_isPreOnly)
        {
            Tensor ctxAttnProj = _jointAttention.ProjectOutput(backend, 1, ctxAttn);
            ctxAfterAttn = new Tensor(ctxShape, DType.F32);
            backend.GatedResidualLastDim(ctxAfterAttn, context, ctxAttnProj, ctxMod[2]);
            ctxAttnProj.Dispose();
        }
        else
        {
            ctxAttn.Dispose();
            ctxAfterAttn = context;
        }

        // ── 12. Image MLP: norm + modulate + GELU FFN + gated residual ──
        Tensor imgMlpModulated;
        if (_norm2HasAffine)
        {
            Tensor imgMlpNormed = new Tensor(imgShape, DType.F32);
            backend.LayerNorm(imgMlpNormed, imgAfterAttn, _norm2Weight!, _norm2Bias!, 1e-6f);
            imgMlpModulated = DiTUtils.Modulate(backend, imgMlpNormed, imgMod[3], imgMod[4], imgShape);
            imgMlpNormed.Dispose();
        }
        else
        {
            imgMlpModulated = DiTUtils.NormModulate(backend, imgAfterAttn, imgMod[3], imgMod[4], imgShape);
        }

        Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();

        Tensor imgFinal = new Tensor(imgShape, DType.F32);
        backend.GatedResidualLastDim(imgFinal, imgAfterAttn, imgMlpOut, imgMod[5]);
        imgMlpOut.Dispose();
        imgAfterAttn.Dispose();

        // ── 13. Context MLP (skip for pre_only) ──
        Tensor ctxFinal;
        if (!_isPreOnly)
        {
            Tensor ctxMlpModulated;
            if (_norm2CtxHasAffine)
            {
                Tensor ctxMlpNormed = new Tensor(ctxShape, DType.F32);
                backend.LayerNorm(ctxMlpNormed, ctxAfterAttn, _norm2CtxWeight!, _norm2CtxBias!, 1e-6f);
                ctxMlpModulated = DiTUtils.Modulate(backend, ctxMlpNormed, ctxMod[3], ctxMod[4], ctxShape);
                ctxMlpNormed.Dispose();
            }
            else
            {
                ctxMlpModulated = DiTUtils.NormModulate(backend, ctxAfterAttn, ctxMod[3], ctxMod[4], ctxShape);
            }

            Tensor ctxMlpOut = _ctxFfn!.Forward(backend, ctxMlpModulated, batch, ctxSeqLen);
            ctxMlpModulated.Dispose();

            ctxFinal = new Tensor(ctxShape, DType.F32);
            backend.GatedResidualLastDim(ctxFinal, ctxAfterAttn, ctxMlpOut, ctxMod[5]);
            ctxMlpOut.Dispose();
            ctxAfterAttn.Dispose();
        }
        else
        {
            ctxFinal = ctxAfterAttn;
        }

        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < ctxMod.Length; i++) ctxMod[i].Dispose();

        return (imgFinal, ctxFinal);
    }
}

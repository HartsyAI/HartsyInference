using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
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

    private JointAttention? _attention;

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
    /// <param name="branchDamp">Residual-stream damp for the F16 activation path (the exact Chroma recipe — see <see cref="ChromaF16"/>): applied to every branch-OUTPUT projection so the token stream rides at damp scale while all branch inputs stay baseline via the scale-invariant no-affine LayerNorm. 1.0 = off.</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix, float branchDamp = 1.0f)
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

        Tensor imgFfnOutWeight = weights[$"{prefix}.ff.net.2.weight"];
        Tensor imgFfnOutBias = weights[$"{prefix}.ff.net.2.bias"];
        Tensor txtFfnOutWeight = weights[$"{prefix}.ff_context.net.2.weight"];
        Tensor txtFfnOutBias = weights[$"{prefix}.ff_context.net.2.bias"];
        if (branchDamp != 1.0f)
        {
            // Weight damp rides the GEMM alpha (any dtype); biases need value-scaled copies (the epilogue
            // adds bias after alpha). Runs once per transformer instance at load.
            _toOutWeight.Fp8ScaleFactor *= branchDamp;
            _toOutBias = ChromaF16.DampBias(_toOutBias, branchDamp);
            _toAddOutWeight.Fp8ScaleFactor *= branchDamp;
            _toAddOutBias = ChromaF16.DampBias(_toAddOutBias, branchDamp);
            imgFfnOutWeight.Fp8ScaleFactor *= branchDamp;
            imgFfnOutBias = ChromaF16.DampBias(imgFfnOutBias, branchDamp);
            txtFfnOutWeight.Fp8ScaleFactor *= branchDamp;
            txtFfnOutBias = ChromaF16.DampBias(txtFfnOutBias, branchDamp);
        }

        // Image MLP (GELU mode): net.0.proj → projection, net.2 → output
        _imgFfn.LoadGeluWeights(
            weights[$"{prefix}.ff.net.0.proj.weight"],
            weights[$"{prefix}.ff.net.0.proj.bias"],
            imgFfnOutWeight,
            imgFfnOutBias);

        // Text MLP (GELU mode)
        _txtFfn.LoadGeluWeights(
            weights[$"{prefix}.ff_context.net.0.proj.weight"],
            weights[$"{prefix}.ff_context.net.0.proj.bias"],
            txtFfnOutWeight,
            txtFfnOutBias);

        // Built after the damp, which replaces the output biases. QK RMS-norm bounds the scores, so the call may narrow to F16.
        _attention = new JointAttention(
            new JointAttentionConfig { Heads = _numHeads, HeadDim = _headDim, ConcatOrder = [1, 0], AllowF16 = true },
            [Stream(_toQWeight!, _toQBias, _toKWeight!, _toKBias, _toVWeight!, _toVBias, _toOutWeight!, _toOutBias, _normQ, _normK),
             Stream(_addQWeight!, _addQBias, _addKWeight!, _addKBias, _addVWeight!, _addVBias, _toAddOutWeight!, _toAddOutBias,
                 _normAddedQ, _normAddedK)]);
    }

    private AttentionLayer Stream(Tensor q, Tensor? qBias, Tensor k, Tensor? kBias, Tensor v, Tensor? vBias, Tensor output,
        Tensor? outputBias, QkNorm normQ, QkNorm normK) => new(
        new AttentionLayerConfig { Heads = _numHeads, KvHeads = _numHeads, HeadDim = _headDim, QkNorm = AttentionQkNorm.PerHead },
        new AttentionLayerWeights
        {
            Query = q, QueryBias = qBias, Key = k, KeyBias = kBias, Value = v, ValueBias = vBias, Output = output,
            OutputBias = outputBias, QueryNorm = normQ.Weight, KeyNorm = normK.Weight, QueryNormEps = normQ.Eps,
            KeyNormEps = normK.Eps,
        });

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
    public (Tensor image, Tensor text) Forward(IBackend backend, Tensor image, Tensor text, Tensor temb, FluxRope rope, Tensor? attnBias = null)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];
        // Activation dtype follows the INPUT (the Chroma/Krea2 pattern): FluxTransformer casts the token
        // streams to F16 once before the block loop on the numerics.ditF16 path; the AdaLN modulation vectors
        // stay F32 (F16 kernels take an F16 activation + F32 params). The F32 path is byte-identical.
        DType act = image.DType;

        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);

        // ── 1. AdaLN modulation ──
        Tensor[] imgMod = _imgModulation.Forward(backend, temb);
        Tensor[] txtMod = _txtModulation.Forward(backend, temb);

        // ── 2. LayerNorm (no affine, eps 1e-6) + modulate x*(1+scale)+shift ──
        Tensor imgModulated = DiTUtils.NormModulate(backend, image, imgMod[0], imgMod[1], imgShape);
        Tensor txtModulated = DiTUtils.NormModulate(backend, text, txtMod[0], txtMod[1], txtShape);

        // ── 3-10. Joint attention over [txt, img], output projections, gated residuals ──
        AttentionCallContext joint = batch == 1
            ? new AttentionCallContext { Rotation = rope, Bias = attnBias }
            : new AttentionCallContext { HeadMajorRotation = rope, Bias = attnBias };
        Span<Tensor?> attended = [null, null];
        _attention!.Attend(backend, [imgModulated, txtModulated], attended, new JointCallContext { Joint = joint, ReleaseInputs = true });
        Tensor imgAttnProj = _attention.ProjectOutput(backend, 0, attended[0]!);
        Tensor imgAfterAttn = new Tensor(imgShape, act);
        backend.GatedResidualLastDim(imgAfterAttn, image, imgAttnProj, imgMod[2]);
        imgAttnProj.Dispose();

        Tensor txtAttnProj = _attention.ProjectOutput(backend, 1, attended[1]!);
        Tensor txtAfterAttn = new Tensor(txtShape, act);
        backend.GatedResidualLastDim(txtAfterAttn, text, txtAttnProj, txtMod[2]);
        txtAttnProj.Dispose();

        // ── 11. Image MLP: LayerNorm + modulate + GELU MLP + gated residual ──
        Tensor imgMlpModulated = DiTUtils.NormModulate(backend, imgAfterAttn, imgMod[3], imgMod[4], imgShape);
        Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();
        Tensor imgFinal = new Tensor(imgShape, act);
        backend.GatedResidualLastDim(imgFinal, imgAfterAttn, imgMlpOut, imgMod[5]);
        imgMlpOut.Dispose();
        imgAfterAttn.Dispose();

        // ── 12. Text MLP: LayerNorm + modulate + GELU MLP + gated residual ──
        Tensor txtMlpModulated = DiTUtils.NormModulate(backend, txtAfterAttn, txtMod[3], txtMod[4], txtShape);
        Tensor txtMlpOut = _txtFfn.Forward(backend, txtMlpModulated, batch, txtSeqLen);
        txtMlpModulated.Dispose();
        Tensor txtFinal = new Tensor(txtShape, act);
        backend.GatedResidualLastDim(txtFinal, txtAfterAttn, txtMlpOut, txtMod[5]);
        txtMlpOut.Dispose();
        txtAfterAttn.Dispose();

        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < txtMod.Length; i++) txtMod[i].Dispose();

        return (imgFinal, txtFinal);
    }
}

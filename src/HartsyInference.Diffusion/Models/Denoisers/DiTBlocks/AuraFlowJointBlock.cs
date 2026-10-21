using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>AuraFlow joint MMDiT block (<c>AuraFlowJointTransformerBlock</c>). Dual-stream image+text attention with AdaLN-Zero modulation, FP32 QK-norm, no biases on attention/FFN linears, and SwiGLU FFN. Differs from SD3 <see cref="JointBlock"/> mainly by the bias-free linears, FP32 LayerNorm, SwiGLU (vs GELU), and the post-attention norm-then-modulate ordering (<c>norm2(residual + gate * attn)</c>). Concat order in joint attention is <c>[txt, img]</c>, matching <c>AuraFlowAttnProcessor2_0</c>.</summary>
public sealed unsafe class AuraFlowJointBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;

    private readonly AdaLNModulation _imgModulation;
    private readonly AdaLNModulation _txtModulation;

    // Image (sample) attention projections — bias=False everywhere.
    private Tensor? _toQWeight;
    private Tensor? _toKWeight;
    private Tensor? _toVWeight;
    private Tensor? _toOutWeight;

    // Context (added) attention projections — bias=False everywhere.
    private Tensor? _addQWeight;
    private Tensor? _addKWeight;
    private Tensor? _addVWeight;
    private Tensor? _toAddOutWeight;

    // FP32 QK-norm — always present in AuraFlow.
    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;
    private readonly QkNorm _normAddedQ;
    private readonly QkNorm _normAddedK;

    private readonly SwiGluFfn _imgFfn;
    private readonly SwiGluFfn _txtFfn;

    private JointAttention? _attention;

    /// <summary>Creates an AuraFlow joint block. <paramref name="headDim"/> is explicit (256 for v0.3); <c>numHeads * headDim</c> must equal <paramref name="hiddenSize"/>.</summary>
    public AuraFlowJointBlock(int hiddenSize, int numHeads, int headDim, int mlpDim, float qkNormEps = 1e-5f)
    {
        if (numHeads * headDim != hiddenSize)
            throw new ArgumentException($"numHeads * headDim ({numHeads} * {headDim} = {numHeads * headDim}) must equal hiddenSize ({hiddenSize}).");

        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = headDim;

        _imgModulation = new AdaLNModulation(hiddenSize, 6);
        _txtModulation = new AdaLNModulation(hiddenSize, 6);

        _normQ = new QkNorm(headDim, qkNormEps);
        _normK = new QkNorm(headDim, qkNormEps);
        _normAddedQ = new QkNorm(headDim, qkNormEps);
        _normAddedK = new QkNorm(headDim, qkNormEps);

        _imgFfn = new SwiGluFfn(hiddenSize, mlpDim);
        _txtFfn = new SwiGluFfn(hiddenSize, mlpDim);
    }

    /// <summary>Loads weights using HuggingFace diffusers naming (<c>joint_transformer_blocks.{i}.*</c>). All AdaLN/attention/FFN linears are bias=False.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _imgModulation.LoadWeights(weights[$"{prefix}.norm1.linear.weight"], null);
        _txtModulation.LoadWeights(weights[$"{prefix}.norm1_context.linear.weight"], null);

        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _toOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];

        _addQWeight = weights[$"{prefix}.attn.add_q_proj.weight"];
        _addKWeight = weights[$"{prefix}.attn.add_k_proj.weight"];
        _addVWeight = weights[$"{prefix}.attn.add_v_proj.weight"];
        _toAddOutWeight = weights[$"{prefix}.attn.to_add_out.weight"];

        // AuraFlow uses `qk_norm="fp32_layer_norm"` with `elementwise_affine=False, bias=False` — i.e.,
        // **no learnable weights** (auraflow_transformer_2d.py lines 161, 226, plus diffusers Attention.__init__).
        // The checkpoint therefore does not ship `attn.norm_q.weight` etc. We synthesize a unit-scale weight
        // (all 1s) so our `QkNorm` (which is RMSNorm-style with a mandatory learnable scale) becomes
        // mathematically `(x / RMS(x)) * 1`. **Caveat:** AuraFlow's QK-norm is *LayerNorm* not *RMSNorm* —
        // it subtracts the mean as well. With this shim we apply RMSNorm instead, which differs by
        // `(x - mean) / std` vs `x / rms`. For zero-mean Q/K this matches; for non-zero-mean it'll be a
        // small but compounding drift across all 4 joint blocks. Fixing this requires a no-affine
        // LayerNorm-style QK-norm — see tracking note in PHASE_4_MODEL_BREADTH.md `### AuraFlow`.
        _normQ.LoadWeights(GetOrFakeOnes(weights, $"{prefix}.attn.norm_q.weight", _headDim));
        _normK.LoadWeights(GetOrFakeOnes(weights, $"{prefix}.attn.norm_k.weight", _headDim));
        _normAddedQ.LoadWeights(GetOrFakeOnes(weights, $"{prefix}.attn.norm_added_q.weight", _headDim));
        _normAddedK.LoadWeights(GetOrFakeOnes(weights, $"{prefix}.attn.norm_added_k.weight", _headDim));

        // AuraFlowFeedForward: linear_1 = gate (SiLU path), linear_2 = linear path, out_projection = output.
        // Maps onto SwiGluFfn's (w1=gate, w3=linear, w2=output) with all biases null.
        _imgFfn.LoadSwiGluWeights(
            weights[$"{prefix}.ff.linear_1.weight"], null,
            weights[$"{prefix}.ff.linear_2.weight"], null,
            weights[$"{prefix}.ff.out_projection.weight"], null);
        _txtFfn.LoadSwiGluWeights(
            weights[$"{prefix}.ff_context.linear_1.weight"], null,
            weights[$"{prefix}.ff_context.linear_2.weight"], null,
            weights[$"{prefix}.ff_context.out_projection.weight"], null);

        // QK-RMS-normed, so the joint call may narrow to F16 (the cuDNN fused config at D=256).
        _attention = new JointAttention(
            new JointAttentionConfig { Heads = _numHeads, HeadDim = _headDim, ConcatOrder = [1, 0], AllowF16 = true },
            [Stream(_toQWeight!, _toKWeight!, _toVWeight!, _toOutWeight!, _normQ, _normK),
             Stream(_addQWeight!, _addKWeight!, _addVWeight!, _toAddOutWeight!, _normAddedQ, _normAddedK)]);
    }

    private AttentionLayer Stream(Tensor q, Tensor k, Tensor v, Tensor output, QkNorm normQ, QkNorm normK) => new(
        new AttentionLayerConfig
        {
            Heads = _numHeads, KvHeads = _numHeads, HeadDim = _headDim, QkNorm = AttentionQkNorm.PerHead, ComputeDType = DType.F32,
        },
        new AttentionLayerWeights
        {
            Query = q, Key = k, Value = v, Output = output, QueryNorm = normQ.Weight, KeyNorm = normK.Weight,
            QueryNormEps = normQ.Eps, KeyNormEps = normK.Eps,
        });

    /// <summary>Forward pass through the dual-stream block. Returns <c>(image, text)</c> matching SD3 C# convention; diffusers reference returns <c>(text, image)</c> from <c>AuraFlowJointTransformerBlock.forward</c>.</summary>
    // GPU-residency rewrite (mirrors ChromaDoubleStreamBlock): every glue op — LayerNorm+AdaLN modulation, per-head
    // FP32 QK-norm, head split/merge, the joint [txt, img] concat/split, gated residuals — runs as an IBackend op so
    // the activation stays device-resident across the block (no per-op DataPointer D2H sync barriers around every
    // GEMM). Q/K/V are declared [B, S, H, D] (byte-identical to [B, S, hidden]) so RmsNorm normalizes over headDim
    // with no reshape; the [txt, img] concat happens in [B, S, hidden] layout then a single Permute0213 lifts to
    // [B, H, S, D] for SDPA (equivalent to the old per-head ConcatAlongSeqDimMultiHead). AuraFlow has no RoPE, so
    // nothing is left on the host. Numerics are bit-identical to the old host helpers: DiTUtils.NormModulate
    // reproduces LayerNormNoAffine + x*(1+scale)+shift; backend.RmsNorm reproduces QkNorm.Forward;
    // backend.GatedResidualLastDim reproduces AdaLNModulation.ApplyGatedResidual.
    public (Tensor image, Tensor text) Forward(IBackend backend, Tensor image, Tensor text, Tensor temb)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];

        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);

        // ── 1. AdaLN modulation: 6 params each [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] ──
        Tensor[] imgMod = _imgModulation.Forward(backend, temb);
        Tensor[] txtMod = _txtModulation.Forward(backend, temb);

        // ── 2. LayerNorm (no affine, eps 1e-6) + modulate x*(1+scale)+shift on both streams ──
        Tensor imgModulated = DiTUtils.NormModulate(backend, image, imgMod[0], imgMod[1], imgShape);
        Tensor txtModulated = DiTUtils.NormModulate(backend, text, txtMod[0], txtMod[1], txtShape);

        // ── 3-9. Joint attention over [txt, img]: bias-free projections, FP32 per-head QK-norm, output projections ──
        Span<Tensor?> attended = [null, null];
        _attention!.Attend(backend, [imgModulated, txtModulated], attended, new JointCallContext { ReleaseInputs = true });
        Tensor imgAttnProj = _attention.ProjectOutput(backend, 0, attended[0]!);
        Tensor txtAttnProj = _attention.ProjectOutput(backend, 1, attended[1]!);

        // ── 10. Image post-attention path (lines 264-267):
        //   h_post = norm2(residual + gate_msa * attn)            <- norm AFTER residual
        //   h_mod  = h_post * (1 + scale_mlp) + shift_mlp
        //   out    = residual + gate_mlp * ff(h_mod)              <- residual is the ORIGINAL `image`
        Tensor imgPreNorm = new Tensor(imgShape, DType.F32);
        backend.GatedResidualLastDim(imgPreNorm, image, imgAttnProj, imgMod[2]);
        imgAttnProj.Dispose();

        Tensor imgMlpModulated = DiTUtils.NormModulate(backend, imgPreNorm, imgMod[3], imgMod[4], imgShape);
        imgPreNorm.Dispose();

        Tensor imgFfOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();

        Tensor imgFinal = new Tensor(imgShape, DType.F32);
        backend.GatedResidualLastDim(imgFinal, image, imgFfOut, imgMod[5]);
        imgFfOut.Dispose();

        // ── 11. Text post-attention path (mirrors image, lines 270-273) ──
        Tensor txtPreNorm = new Tensor(txtShape, DType.F32);
        backend.GatedResidualLastDim(txtPreNorm, text, txtAttnProj, txtMod[2]);
        txtAttnProj.Dispose();

        Tensor txtMlpModulated = DiTUtils.NormModulate(backend, txtPreNorm, txtMod[3], txtMod[4], txtShape);
        txtPreNorm.Dispose();

        Tensor txtFfOut = _txtFfn.Forward(backend, txtMlpModulated, batch, txtSeqLen);
        txtMlpModulated.Dispose();

        Tensor txtFinal = new Tensor(txtShape, DType.F32);
        backend.GatedResidualLastDim(txtFinal, text, txtFfOut, txtMod[5]);
        txtFfOut.Dispose();

        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < txtMod.Length; i++) txtMod[i].Dispose();

        return (imgFinal, txtFinal);
    }

    /// <summary>Returns the named tensor or a fresh F32 [headDim] all-ones tensor when missing.
    /// Used for AuraFlow's non-affine QK-norm (no learnable scale in the checkpoint). The synthetic
    /// tensor lives for the lifetime of the block — small enough that we don't bother to cache.</summary>
    internal static Tensor GetOrFakeOnes(IReadOnlyDictionary<string, Tensor> weights, string key, int headDim)
    {
        if (weights.TryGetValue(key, out Tensor? t) && t is not null)
            return t.DType != DType.F32 ? t.CastTo(DType.F32) : t;

        Tensor ones = new(new TensorShape(headDim), DType.F32);
        unsafe
        {
            float* p = (float*)ones.DataPointer;
            for (int i = 0; i < headDim; i++) p[i] = 1.0f;
        }
        return ones;
    }

    /// <summary>Yields all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _imgModulation.EnumerateWeights()) yield return t;
        foreach (Tensor t in _txtModulation.EnumerateWeights()) yield return t;

        if (_toQWeight is not null) yield return _toQWeight;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toOutWeight is not null) yield return _toOutWeight;

        if (_addQWeight is not null) yield return _addQWeight;
        if (_addKWeight is not null) yield return _addKWeight;
        if (_addVWeight is not null) yield return _addVWeight;
        if (_toAddOutWeight is not null) yield return _toAddOutWeight;

        foreach (Tensor t in _normQ.EnumerateWeights()) yield return t;
        foreach (Tensor t in _normK.EnumerateWeights()) yield return t;
        foreach (Tensor t in _normAddedQ.EnumerateWeights()) yield return t;
        foreach (Tensor t in _normAddedK.EnumerateWeights()) yield return t;

        foreach (Tensor t in _imgFfn.EnumerateWeights()) yield return t;
        foreach (Tensor t in _txtFfn.EnumerateWeights()) yield return t;
    }
}

using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>AuraFlow single-stream block (<c>AuraFlowSingleTransformerBlock</c>). Operates on the already-concatenated <c>[txt, img]</c> token stream produced by the transformer wrapper. Self-attention only (no <c>add_kv</c> proj), AdaLN-Zero modulation, FP32 QK-norm, bias-free linears, and a SwiGLU FFN. Differs from a standard DiT block by the post-attention norm-then-modulate ordering: <c>norm2(residual + gate * attn)</c>.</summary>
public sealed unsafe class AuraFlowSingleBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;

    private readonly AdaLNModulation _modulation;

    private Tensor? _toQWeight;
    private Tensor? _toKWeight;
    private Tensor? _toVWeight;
    private Tensor? _toOutWeight;

    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;

    private readonly SwiGluFfn _ffn;

    private AttentionLayer? _attention;

    /// <summary>Creates an AuraFlow single-stream block. <paramref name="headDim"/> is explicit (256 for v0.3); <c>numHeads * headDim</c> must equal <paramref name="hiddenSize"/>.</summary>
    public AuraFlowSingleBlock(int hiddenSize, int numHeads, int headDim, int mlpDim, float qkNormEps = 1e-5f)
    {
        if (numHeads * headDim != hiddenSize)
            throw new ArgumentException($"numHeads * headDim ({numHeads} * {headDim} = {numHeads * headDim}) must equal hiddenSize ({hiddenSize}).");

        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = headDim;

        _modulation = new AdaLNModulation(hiddenSize, 6);

        _normQ = new QkNorm(headDim, qkNormEps);
        _normK = new QkNorm(headDim, qkNormEps);

        _ffn = new SwiGluFfn(hiddenSize, mlpDim);
    }

    /// <summary>Loads weights using HuggingFace diffusers naming (<c>single_transformer_blocks.{i}.*</c>). All linears are bias=False.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _modulation.LoadWeights(weights[$"{prefix}.norm1.linear.weight"], null);

        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _toOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];

        // Same shim as AuraFlowJointBlock — AuraFlow's `qk_norm="fp32_layer_norm"` is non-affine,
        // so the checkpoint doesn't ship these weights. Synthesize unit-scale.
        _normQ.LoadWeights(AuraFlowJointBlock.GetOrFakeOnes(weights, $"{prefix}.attn.norm_q.weight", _headDim));
        _normK.LoadWeights(AuraFlowJointBlock.GetOrFakeOnes(weights, $"{prefix}.attn.norm_k.weight", _headDim));

        _ffn.LoadSwiGluWeights(
            weights[$"{prefix}.ff.linear_1.weight"], null,
            weights[$"{prefix}.ff.linear_2.weight"], null,
            weights[$"{prefix}.ff.out_projection.weight"], null);

        // QK-RMS-normed, so the call may narrow to F16 (the cuDNN fused config at D=256).
        _attention = new AttentionLayer(
            new AttentionLayerConfig
            {
                Heads = _numHeads, KvHeads = _numHeads, HeadDim = _headDim, QkNorm = AttentionQkNorm.PerHead,
                ComputeDType = DType.F32, AllowF16 = true,
            },
            new AttentionLayerWeights
            {
                Query = _toQWeight, Key = _toKWeight, Value = _toVWeight, Output = _toOutWeight, QueryNorm = _normQ.Weight,
                KeyNorm = _normK.Weight, QueryNormEps = _normQ.Eps, KeyNormEps = _normK.Eps,
            });
    }

    /// <summary>Forward pass on the joint <c>[txt, img]</c> token stream. Per <c>AuraFlowSingleTransformerBlock.forward</c> lines 171-191.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor temb)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];

        TensorShape hidShape = new TensorShape(batch, seqLen, _hiddenSize);

        // ── 1. AdaLN-Zero (norm + modulate via shift_msa, scale_msa) returning gate_msa, shift_mlp, scale_mlp, gate_mlp ──
        // The reference's `norm1(hidden, emb=temb)` computes `norm(x) * (1 + scale_msa) + shift_msa` internally
        // and returns (modulated_x, gate_msa, shift_mlp, scale_mlp, gate_mlp). We replicate this with our 6-param
        // AdaLNModulation: mod[0]=shift_msa, mod[1]=scale_msa; mod[2]=gate_msa, mod[3]=shift_mlp, mod[4]=scale_mlp,
        // mod[5]=gate_mlp.
        Tensor[] mod = _modulation.Forward(backend, temb);

        // ── 2. LayerNorm (no affine, eps 1e-6) + modulate: x*(1+scale_msa)+shift_msa ──
        Tensor normHidden = DiTUtils.NormModulate(backend, hidden, mod[0], mod[1], hidShape);

        // ── 3-7. Self-attention: bias-free projections, FP32 per-head QK-norm, output projection ──
        Tensor attnProj = _attention!.Forward(backend, normHidden, normHidden, new AttentionCallContext { ReleaseInputs = true });

        // ── 8. Post-attention path (lines 187-191):
        //   h_post  = norm2(residual + gate_msa * attn)
        //   h_mod   = h_post * (1 + scale_mlp) + shift_mlp
        //   out     = residual + gate_mlp * ff(h_mod)
        Tensor preNorm = new Tensor(hidShape, DType.F32);
        backend.GatedResidualLastDim(preNorm, hidden, attnProj, mod[2]);
        attnProj.Dispose();

        Tensor mlpModulated = DiTUtils.NormModulate(backend, preNorm, mod[3], mod[4], hidShape);
        preNorm.Dispose();

        Tensor ffOut = _ffn.Forward(backend, mlpModulated, batch, seqLen);
        mlpModulated.Dispose();

        Tensor result = new Tensor(hidShape, DType.F32);
        backend.GatedResidualLastDim(result, hidden, ffOut, mod[5]);
        ffOut.Dispose();

        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();

        return result;
    }

    /// <summary>Yields all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _modulation.EnumerateWeights()) yield return t;

        if (_toQWeight is not null) yield return _toQWeight;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toOutWeight is not null) yield return _toOutWeight;

        foreach (Tensor t in _normQ.EnumerateWeights()) yield return t;
        foreach (Tensor t in _normK.EnumerateWeights()) yield return t;

        foreach (Tensor t in _ffn.EnumerateWeights()) yield return t;
    }
}

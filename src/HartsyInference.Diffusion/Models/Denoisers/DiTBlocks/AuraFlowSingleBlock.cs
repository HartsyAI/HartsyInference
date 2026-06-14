using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>AuraFlow single-stream block (<c>AuraFlowSingleTransformerBlock</c>). Operates on the already-concatenated <c>[txt, img]</c> token stream produced by the transformer wrapper. Self-attention only (no <c>add_kv</c> proj), AdaLN-Zero modulation, FP32 QK-norm, bias-free linears, and a SwiGLU FFN. Differs from a standard DiT block by the post-attention norm-then-modulate ordering: <c>norm2(residual + gate * attn)</c>.</summary>
public sealed unsafe class AuraFlowSingleBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _innerDim;
    private readonly int _mlpDim;

    private readonly AdaLNModulation _modulation;

    private Tensor? _toQWeight;
    private Tensor? _toKWeight;
    private Tensor? _toVWeight;
    private Tensor? _toOutWeight;

    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;

    private readonly SwiGluFfn _ffn;

    /// <summary>Creates an AuraFlow single-stream block. <paramref name="headDim"/> is explicit (256 for v0.3); <c>numHeads * headDim</c> must equal <paramref name="hiddenSize"/>.</summary>
    public AuraFlowSingleBlock(int hiddenSize, int numHeads, int headDim, int mlpDim, float qkNormEps = 1e-5f)
    {
        if (numHeads * headDim != hiddenSize)
            throw new ArgumentException($"numHeads * headDim ({numHeads} * {headDim} = {numHeads * headDim}) must equal hiddenSize ({hiddenSize}).");

        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = headDim;
        _innerDim = numHeads * headDim;
        _mlpDim = mlpDim;

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
        _normQ.LoadWeights(GetOrFakeOnes(weights, $"{prefix}.attn.norm_q.weight", _headDim));
        _normK.LoadWeights(GetOrFakeOnes(weights, $"{prefix}.attn.norm_k.weight", _headDim));

        _ffn.LoadSwiGluWeights(
            weights[$"{prefix}.ff.linear_1.weight"], null,
            weights[$"{prefix}.ff.linear_2.weight"], null,
            weights[$"{prefix}.ff.out_projection.weight"], null);
    }

    /// <summary>Forward pass on the joint <c>[txt, img]</c> token stream. Per <c>AuraFlowSingleTransformerBlock.forward</c> lines 171-191.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor temb)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];

        TensorShape hidShape = new TensorShape(batch, seqLen, _hiddenSize);
        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);

        // ── 1. AdaLN-Zero (norm + modulate via shift_msa, scale_msa) returning gate_msa, shift_mlp, scale_mlp, gate_mlp ──
        // The reference's `norm1(hidden, emb=temb)` computes `norm(x) * (1 + scale_msa) + shift_msa` internally
        // and returns (modulated_x, gate_msa, shift_mlp, scale_mlp, gate_mlp). We replicate this with our 6-param
        // AdaLNModulation: ApplyModulation uses mod[0]=shift_msa, mod[1]=scale_msa; mod[2]=gate_msa,
        // mod[3]=shift_mlp, mod[4]=scale_mlp, mod[5]=gate_mlp.
        Tensor[] mod = _modulation.Forward(backend, temb);

        Tensor normed = new Tensor(hidShape, DType.F32);
        DiTUtils.LayerNormNoAffine(normed, hidden, batch, seqLen, _hiddenSize);
        Tensor normHidden = AdaLNModulation.ApplyModulation(normed, mod[0], mod[1], batch, seqLen, _hiddenSize);
        normed.Dispose();

        // ── 2. Self-attention Q/K/V (bias-free) ──
        Tensor q = new Tensor(hidShape, DType.F32);
        backend.Linear(q, normHidden, _toQWeight!, null);
        Tensor k = new Tensor(hidShape, DType.F32);
        backend.Linear(k, normHidden, _toKWeight!, null);
        Tensor v = new Tensor(hidShape, DType.F32);
        backend.Linear(v, normHidden, _toVWeight!, null);
        normHidden.Dispose();

        // ── 3. FP32 QK-norm ──
        int numVectors = batch * seqLen * _numHeads;
        Tensor qNormed = new Tensor(q.Shape, DType.F32);
        Tensor kNormed = new Tensor(k.Shape, DType.F32);
        _normQ.Forward(qNormed, q, numVectors);
        _normK.Forward(kNormed, k, numVectors);
        q.Dispose();
        k.Dispose();
        q = qNormed;
        k = kNormed;

        // ── 4. Reshape to multi-head and run SDPA ──
        Tensor qMh = new Tensor(mhShape, DType.F32);
        Tensor kMh = new Tensor(mhShape, DType.F32);
        Tensor vMh = new Tensor(mhShape, DType.F32);
        DiTUtils.ReshapeToMultiHead(qMh, q, batch, seqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(kMh, k, batch, seqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(vMh, v, batch, seqLen, _numHeads, _headDim);
        q.Dispose(); k.Dispose(); v.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnMh = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnMh, qMh, kMh, vMh, null, scale);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor attn = new Tensor(hidShape, DType.F32);
        DiTUtils.ReshapeFromMultiHead(attn, attnMh, batch, seqLen, _numHeads, _headDim);
        attnMh.Dispose();

        // ── 5. Output projection ──
        Tensor attnProj = new Tensor(hidShape, DType.F32);
        backend.Linear(attnProj, attn, _toOutWeight!, null);
        attn.Dispose();

        // ── 6. Post-attention path (lines 187-191):
        //   h_post  = norm2(residual + gate_msa * attn)
        //   h_mod   = h_post * (1 + scale_mlp) + shift_mlp
        //   out     = residual + gate_mlp * ff(h_mod)
        Tensor preNorm = AdaLNModulation.ApplyGatedResidual(hidden, attnProj, mod[2], batch, seqLen, _hiddenSize);
        attnProj.Dispose();

        Tensor postNorm = new Tensor(hidShape, DType.F32);
        DiTUtils.LayerNormNoAffine(postNorm, preNorm, batch, seqLen, _hiddenSize);
        preNorm.Dispose();

        Tensor mlpModulated = AdaLNModulation.ApplyModulation(postNorm, mod[3], mod[4], batch, seqLen, _hiddenSize);
        postNorm.Dispose();

        Tensor ffOut = _ffn.Forward(backend, mlpModulated, batch, seqLen);
        mlpModulated.Dispose();

        Tensor result = AdaLNModulation.ApplyGatedResidual(hidden, ffOut, mod[5], batch, seqLen, _hiddenSize);
        ffOut.Dispose();

        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();

        return result;
    }

    /// <summary>Yields all weight tensors for GPU preloading.</summary>
    /// <summary>See note on <c>AuraFlowJointBlock.GetOrFakeOnes</c>.</summary>
    private static Tensor GetOrFakeOnes(IReadOnlyDictionary<string, Tensor> weights, string key, int headDim)
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

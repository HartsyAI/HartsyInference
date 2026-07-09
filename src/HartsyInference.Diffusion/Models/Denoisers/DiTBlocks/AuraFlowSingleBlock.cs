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
    // GPU-residency rewrite (mirrors ChromaSingleStreamBlock / Krea2Attention): every glue op — LayerNorm+AdaLN
    // modulation, per-head FP32 QK-norm, head split/merge, gated residual — runs as an IBackend op so the activation
    // stays device-resident across the block (no per-op DataPointer D2H sync barriers around every GEMM). Head split =
    // declaring Q/K/V directly as [B, S, H, D] (byte-identical to [B, S, hidden]) so RmsNorm normalizes over headDim
    // with no reshape, then Permute0213 to/from [B, H, S, D]. AuraFlow has no RoPE, so nothing is left on the host.
    // Numerics are bit-identical to the old host helpers: DiTUtils.NormModulate reproduces LayerNormNoAffine +
    // x*(1+scale)+shift; backend.RmsNorm reproduces QkNorm.Forward; backend.GatedResidualLastDim reproduces
    // AdaLNModulation.ApplyGatedResidual.
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor temb)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];

        TensorShape hidShape = new TensorShape(batch, seqLen, _hiddenSize);
        // [B, S, H, D] view (byte-identical to [B, S, hidden]) so RmsNorm normalizes over headDim.
        TensorShape heads = new TensorShape(batch, seqLen, _numHeads, _headDim);
        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);

        // ── 1. AdaLN-Zero (norm + modulate via shift_msa, scale_msa) returning gate_msa, shift_mlp, scale_mlp, gate_mlp ──
        // The reference's `norm1(hidden, emb=temb)` computes `norm(x) * (1 + scale_msa) + shift_msa` internally
        // and returns (modulated_x, gate_msa, shift_mlp, scale_mlp, gate_mlp). We replicate this with our 6-param
        // AdaLNModulation: mod[0]=shift_msa, mod[1]=scale_msa; mod[2]=gate_msa, mod[3]=shift_mlp, mod[4]=scale_mlp,
        // mod[5]=gate_mlp.
        Tensor[] mod = _modulation.Forward(backend, temb);

        // ── 2. LayerNorm (no affine, eps 1e-6) + modulate: x*(1+scale_msa)+shift_msa ──
        Tensor normHidden = DiTUtils.NormModulate(backend, hidden, mod[0], mod[1], hidShape);

        // ── 3. Self-attention Q/K/V (bias-free), declared [B, S, H, D] for per-head RMSNorm ──
        Tensor q = new Tensor(heads, DType.F32);
        backend.Linear(q, normHidden, _toQWeight!, null);
        Tensor k = new Tensor(heads, DType.F32);
        backend.Linear(k, normHidden, _toKWeight!, null);
        Tensor v = new Tensor(heads, DType.F32);
        backend.Linear(v, normHidden, _toVWeight!, null);
        normHidden.Dispose();

        // ── 4. FP32 QK-norm (per-head RMSNorm over the last dim = headDim) ──
        Tensor qNormed = new Tensor(heads, DType.F32);
        backend.RmsNorm(qNormed, q, _normQ.Weight, _normQ.Eps);
        q.Dispose();
        Tensor kNormed = new Tensor(heads, DType.F32);
        backend.RmsNorm(kNormed, k, _normK.Weight, _normK.Eps);
        k.Dispose();

        // ── 5. Permute [B, S, H, D] → [B, H, S, D] then run SDPA ──
        Tensor qMh = new Tensor(mhShape, DType.F32);
        backend.Permute0213(qMh, qNormed, seqLen, _numHeads, _headDim);
        qNormed.Dispose();
        Tensor kMh = new Tensor(mhShape, DType.F32);
        backend.Permute0213(kMh, kNormed, seqLen, _numHeads, _headDim);
        kNormed.Dispose();
        Tensor vMh = new Tensor(mhShape, DType.F32);
        backend.Permute0213(vMh, v, seqLen, _numHeads, _headDim);
        v.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnMh = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnMh, qMh, kMh, vMh, null, scale, allowF16: true);   // QK-RMS-normed, mask-null, D=256 — the proven cuDNN fused config
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        // ── 6. Permute back [B, H, S, D] → [B, S, hidden] ──
        Tensor attn = new Tensor(hidShape, DType.F32);
        backend.Permute0213(attn, attnMh, _numHeads, seqLen, _headDim);
        attnMh.Dispose();

        // ── 7. Output projection ──
        Tensor attnProj = new Tensor(hidShape, DType.F32);
        backend.Linear(attnProj, attn, _toOutWeight!, null);
        attn.Dispose();

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

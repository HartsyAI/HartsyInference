using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Flux SingleStreamBlock: parallel attention + MLP on a concatenated image+text sequence. Uses a fused linear1 for Q/K/V + MLP input, then combines attention output with GELU(MLP) via linear2. No SD3 equivalent exists.</summary>
public sealed class FluxSingleStreamBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _mlpDim;

    // Modulation: SiLU + Linear → 3 params (shift, scale, gate)
    private readonly AdaLNModulation _modulation;

    // QK-norm
    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;

    // Separate Q/K/V projections (diffusers format) + MLP projection
    private Tensor? _toQWeight, _toQBias;
    private Tensor? _toKWeight, _toKBias;
    private Tensor? _toVWeight, _toVBias;
    private Tensor? _projMlpWeight, _projMlpBias;

    // Output: proj_out combines attention output + MLP output → hidden
    private Tensor? _projOutWeight, _projOutBias;

    /// <summary>Creates a FluxSingleStreamBlock.</summary>
    /// <param name="hiddenSize">Model hidden dimension (3072 for Flux.1).</param>
    /// <param name="numHeads">Number of attention heads (24 for Flux.1).</param>
    /// <param name="mlpDim">MLP inner dimension (4 * hiddenSize = 12288).</param>
    /// <param name="qkNormEps">QK-norm RMSNorm epsilon.</param>
    public FluxSingleStreamBlock(int hiddenSize, int numHeads, int mlpDim, float qkNormEps = 1e-6f)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = hiddenSize / numHeads;
        _mlpDim = mlpDim;

        _modulation = new AdaLNModulation(hiddenSize, 3);
        _normQ = new QkNorm(_headDim, qkNormEps);
        _normK = new QkNorm(_headDim, qkNormEps);
    }

    /// <summary>Loads weights from named tensors using diffusers naming: single_transformer_blocks.{i}.* </summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _modulation.LoadWeights(
            weights[$"{prefix}.norm.linear.weight"],
            weights[$"{prefix}.norm.linear.bias"]);

        _normQ.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);

        // Separate Q/K/V in diffusers format
        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];

        // Bias is optional (Flux.1 has bias, Flux.2 does not)
        weights.TryGetValue($"{prefix}.attn.to_q.bias", out _toQBias);
        weights.TryGetValue($"{prefix}.attn.to_k.bias", out _toKBias);
        weights.TryGetValue($"{prefix}.attn.to_v.bias", out _toVBias);

        // MLP projection
        _projMlpWeight = weights[$"{prefix}.proj_mlp.weight"];
        weights.TryGetValue($"{prefix}.proj_mlp.bias", out _projMlpBias);

        // Output projection: [hiddenSize + mlpDim, hiddenSize]
        _projOutWeight = weights[$"{prefix}.proj_out.weight"];
        weights.TryGetValue($"{prefix}.proj_out.bias", out _projOutBias);
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in _modulation.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
        if (_toQWeight is not null) yield return _toQWeight;
        if (_toQBias is not null) yield return _toQBias;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toKBias is not null) yield return _toKBias;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toVBias is not null) yield return _toVBias;
        if (_projMlpWeight is not null) yield return _projMlpWeight;
        if (_projMlpBias is not null) yield return _projMlpBias;
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
    }

    /// <summary>Forward pass on concatenated [text, image] sequence. Parallel attention + MLP.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="x">Input [B, totalSeqLen, hidden] (text + image concatenated).</param>
    /// <param name="temb">Timestep embedding [B, hidden].</param>
    /// <param name="rope">Precomputed FluxRope.</param>
    /// <returns>Updated x tensor [B, totalSeqLen, hidden].</returns>
    // GPU-residency rewrite (mirrors the verified HunyuanImageSingleBlock): every glue op (LayerNorm / AdaLN
    // modulation / QK-norm / head reshape / feature-dim concat / gated residual) runs as an IBackend op so the
    // activation stays device-resident — no per-op DataPointer D2H sync barriers (the old LayerNormNoAffine /
    // AdaLNModulation / QkNorm.Forward / ReshapeTo(From)MultiHead / ConcatAlongFeatureDim host loops D2H-synced
    // every intermediate around every GEMM). Head reshape = declaring Q/K/V directly as [B, S, H, D]
    // (byte-identical to [B, S, hidden]) so QK-norm's RmsNorm runs over headDim with no reshape, then Permute0213
    // to/from [B, H, S, D]. RoPE runs pre-permute via FluxRope.ApplyGpu (same interleaved-pair rotation, identity
    // on the zero-position text rows, bit-identical to the host path); B>1 keeps the host rope on the permuted
    // layout, exactly as before.
    public Tensor Forward(IBackend backend, Tensor x, Tensor temb, FluxRope rope, Tensor? attnBias = null)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];

        TensorShape shape = new TensorShape(batch, seqLen, _hiddenSize);
        // [B, S, H, D] view (byte-identical to [B, S, hidden]) so RmsNorm normalizes over headDim.
        TensorShape headsShape = new TensorShape(batch, seqLen, _numHeads, _headDim);
        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        TensorShape mlpShape = new TensorShape(batch, seqLen, _mlpDim);

        // ── 1. Modulation: 3 params (shift, scale, gate) ──
        Tensor[] mod = _modulation.Forward(backend, temb);

        // ── 2. LayerNorm (no affine, eps 1e-6) + modulate x*(1+scale)+shift ──
        Tensor modulated = DiTUtils.NormModulate(backend, x, mod[0], mod[1], shape);

        // ── 3. Q/K/V projections + MLP projection (parallel) ──
        Tensor q = new Tensor(headsShape, DType.F32);
        backend.Linear(q, modulated, _toQWeight!, _toQBias);
        Tensor k = new Tensor(headsShape, DType.F32);
        backend.Linear(k, modulated, _toKWeight!, _toKBias);
        Tensor v = new Tensor(headsShape, DType.F32);
        backend.Linear(v, modulated, _toVWeight!, _toVBias);
        // mlpInput / mlpActivated / concatted run at F16. At 1024x1024 these are the three
        // largest activations in the block (213/213/267 MB at F32) and dominate VRAM peak.
        // F16 halves them and lets the next Linear skip its input cast since the joint
        // dtype is already F16. The proj_mlp Linear writes F16 directly into mlpInput;
        // cuBLAS still accumulates in F32 internally so accuracy is preserved.
        Tensor mlpInput = new Tensor(mlpShape, DType.F16);
        backend.Linear(mlpInput, modulated, _projMlpWeight!, _projMlpBias);
        modulated.Dispose();

        // ── 4. QK-Norm (per-head RMSNorm over the last dim = headDim) ──
        Tensor qNormed = new Tensor(headsShape, DType.F32);
        backend.RmsNorm(qNormed, q, _normQ.Weight, _normQ.Eps);
        q.Dispose();
        Tensor kNormed = new Tensor(headsShape, DType.F32);
        backend.RmsNorm(kNormed, k, _normK.Weight, _normK.Eps);
        k.Dispose();

        // ── 5. RoPE BEFORE the head permute: the pre-permute [B(=1), S, H, D] layout is exactly
        // WanRopeInterleaved's [S, heads·headDim] contract, so the whole joint Q/K rotates in one GPU-resident
        // op (was a full D2H→host-trig→H2D excursion per block per step).
        if (batch == 1)
            rope.ApplyGpu(backend, qNormed, kNormed, _numHeads);

        // ── 6. Permute [B, S, H, D] → [B, H, S, D] ──
        Tensor qMh = new Tensor(mhShape, DType.F32);
        backend.Permute0213(qMh, qNormed, seqLen, _numHeads, _headDim);
        qNormed.Dispose();
        Tensor kMh = new Tensor(mhShape, DType.F32);
        backend.Permute0213(kMh, kNormed, seqLen, _numHeads, _headDim);
        kNormed.Dispose();
        Tensor vMh = new Tensor(mhShape, DType.F32);
        backend.Permute0213(vMh, v, seqLen, _numHeads, _headDim);
        v.Dispose();

        // Host RoPE fallback (batched inference only; B=1 runs the GPU path at 5) ──
        if (batch != 1)
            rope.Forward(qMh, kMh, batch, _numHeads, seqLen);

        // ── 7. SDPA ──
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, attnBias, scale, allowF16: true);   // QK RMS-norm bounds scores; enables the cuDNN fused path (bias rides fp32 in-engine)
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();

        // ── 8. Permute attention output back [B, H, S, D] → [B, S, hidden] ──
        Tensor attnFlat = new Tensor(shape, DType.F32);
        backend.Permute0213(attnFlat, attnOut, _numHeads, seqLen, _headDim);
        attnOut.Dispose();

        // ── 9. GELU(tanh) on MLP input ── (F16, matches mlpInput; CudaBackend.Gelu has an F16 path)
        Tensor mlpActivated = new Tensor(mlpShape, DType.F16);
        backend.Gelu(mlpActivated, mlpInput);
        mlpInput.Dispose();

        // ── 10. Concatenate [attn_out, gelu(mlp)] → proj_out ──
        // attnFlat is F32 (the attention path stays F32 — SDPA is fine in F32 and the
        // attention activations are 4x smaller than the MLP ones). Cast it down to F16
        // so concat operands match. The cast is small (53 MB at 1024x1024), a worthwhile
        // tradeoff for a 134 MB concatted buffer at F16 instead of 267 MB at F32.
        // backend.Concat at dim 2 is a raw byte-copy, dtype-agnostic — byte-identical to
        // the old host ConcatAlongFeatureDim.
        Tensor attnFlatF16 = Utilities.DtypeCastHelper.EnsureDtype(backend, attnFlat, DType.F16);

        int concatDim = _hiddenSize + _mlpDim;
        TensorShape concatShape = new TensorShape(batch, seqLen, concatDim);
        Tensor concatted = new Tensor(concatShape, DType.F16);
        backend.Concat(concatted, new Tensor[] { attnFlatF16, mlpActivated }, 2);
        attnFlatF16.Dispose();
        mlpActivated.Dispose();

        // proj_out's input is F16 (concatted) and its weight is fp8/F16 — joint resolution
        // picks F16 with no input cast needed. Output stays F32 because the gated residual
        // at the end of the block adds it to x (F32).
        Tensor output = new Tensor(shape, DType.F32);
        backend.Linear(output, concatted, _projOutWeight!, _projOutBias);
        concatted.Dispose();

        // ── 11. Gated residual: x = x + gate * output ──
        Tensor result = new Tensor(shape, DType.F32);
        backend.GatedResidualLastDim(result, x, output, mod[2]);
        output.Dispose();

        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();

        return result;
    }
}

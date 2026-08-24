using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Chroma single-stream (parallel attention + MLP) block. Mirrors <see cref="FluxSingleStreamBlock"/>
/// but takes a precomputed modulation slice <c>[B, 3, hidden]</c> from the approximator's table — the per-block
/// <c>norm.linear</c> projection is pruned (Chroma's defining architectural change).
///
/// The 3 rows are <c>(shift_msa, scale_msa, gate_msa)</c>. Input is the already-concatenated <c>[txt; img]</c>
/// sequence — the parent transformer does the concat once before the loop and slices the image tail off after.
///
/// Reference: <c>diffusers/models/transformers/transformer_chroma.py:204-273</c>.</summary>
public sealed unsafe class ChromaSingleStreamBlock : IStreamingBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _mlpDim;

    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;

    private Tensor? _toQWeight, _toQBias;
    private Tensor? _toKWeight, _toKBias;
    private Tensor? _toVWeight, _toVBias;
    private Tensor? _projMlpWeight, _projMlpBias;

    // Fused BFL linear1 [3*hidden + mlp, hidden] (HARTSY_CHROMA_FUSED_QKV: the converter kept it whole).
    // When set, the split projections above stay null and the forward runs one GEMM + SliceLastDim×2 +
    // QkvSplitNorm (the Hunyuan3DFluxBlocks recipe).
    private Tensor? _lin1Weight, _lin1Bias;

    private Tensor? _projOutWeight, _projOutBias;

    /// <summary>Creates a ChromaSingleStreamBlock.</summary>
    /// <param name="hiddenSize">Model hidden dimension (3072 for Chroma v1).</param>
    /// <param name="numHeads">Number of attention heads (24 for v1).</param>
    /// <param name="headDim">Per-head dimension (128 for v1).</param>
    /// <param name="qkNormEps">QK-norm RMSNorm epsilon.</param>
    public ChromaSingleStreamBlock(int hiddenSize, int numHeads, int headDim, float qkNormEps = 1e-6f)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = headDim;
        _mlpDim = hiddenSize * 4;

        _normQ = new QkNorm(_headDim, qkNormEps);
        _normK = new QkNorm(_headDim, qkNormEps);
    }

    /// <summary>Loads weights from named tensors using diffusers naming
    /// (<c>single_transformer_blocks.{i}.*</c>). The <c>norm.linear</c> entry is intentionally absent.</summary>
    /// <param name="branchDamp">Residual-stream damp for the F16 activation path (see <see cref="ChromaF16"/>):
    /// applied to the block's single branch-output projection (<c>proj_out</c>). 1.0 = off.</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix, float branchDamp = 1.0f)
    {
        _normQ.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);

        if (weights.TryGetValue($"{prefix}.linear1.weight", out Tensor? lin1W))
        {
            _lin1Weight = lin1W;
            _lin1Bias = weights[$"{prefix}.linear1.bias"];
        }
        else
        {
            _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
            _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
            _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
            _toQBias = weights[$"{prefix}.attn.to_q.bias"];
            _toKBias = weights[$"{prefix}.attn.to_k.bias"];
            _toVBias = weights[$"{prefix}.attn.to_v.bias"];

            _projMlpWeight = weights[$"{prefix}.proj_mlp.weight"];
            _projMlpBias = weights[$"{prefix}.proj_mlp.bias"];
        }

        _projOutWeight = weights[$"{prefix}.proj_out.weight"];
        _projOutBias = weights[$"{prefix}.proj_out.bias"];
        if (branchDamp != 1.0f)
        {
            _projOutWeight.Fp8ScaleFactor *= branchDamp;
            _projOutBias = ChromaF16.DampBias(_projOutBias, branchDamp);
        }
    }

    /// <inheritdoc/>
    /// <remarks>Via <see cref="DType.ComputeByteCount"/>, not <c>ElementCount * SizeInBytes</c>: block-quantized
    /// dtypes report <c>SizeInBytes == 0</c>, so the naive product totals to zero and silently disables streaming.</remarks>
    public long EstimatedWeightBytes
    {
        get
        {
            long total = 0;
            foreach (Tensor w in EnumerateWeights()) total += w.DType.ComputeByteCount(w.ElementCount);
            return total;
        }
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
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
        if (_lin1Weight is not null) yield return _lin1Weight;
        if (_lin1Bias is not null) yield return _lin1Bias;
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
    }

    /// <summary>Forward pass on the already-concatenated <c>[txt; img]</c> sequence.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="x">Input [B, totalSeqLen, hidden].</param>
    /// <param name="temb">Modulation slice [B, 3, hidden] (shift, scale, gate).</param>
    /// <param name="rope">Precomputed FluxRope for the joint sequence.</param>
    /// <param name="sdpaMask">Optional additive [B, 1, S, S] SDPA mask, pre-built once per forward by ChromaTransformer (shared; not disposed here).</param>
    // GPU-residency rewrite (Chroma-only): all glue runs as IBackend GPU ops; only RoPE stays on the CPU
    // (interleaved/GPT-J convention, no CUDA kernel — ~5s/run). Mirrors ChromaDoubleStreamBlock; batch is always 1
    // for Chroma.
    public Tensor Forward(IBackend backend, Tensor x, Tensor temb, FluxRope rope, Tensor? sdpaMask)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        float scale = 1.0f / MathF.Sqrt(_headDim);
        // Activation dtype follows the INPUT (see ChromaDoubleStreamBlock.Forward): F16 on the
        // HARTSY_DIT_F16 path; modulation vectors and the SDPA mask stay F32.
        DType act = x.DType;

        // ── 1. Slice modulation rows into tiny [B, hidden] tensors: shift, scale, gate. Device slices
        // (B=1): the old host path read the device-produced temb's DataPointer — a full pipeline drain
        // per block (the temb stream-stall). ──
        Tensor[] mod = batch == 1 ? ChromaDoubleStreamBlock.SliceModRowsDevice(backend, temb, rowStart: 0, rowCount: 3)
            : ChromaDoubleStreamBlock.SliceModRows(temb, batch, rowStart: 0, rowCount: 3);

        TensorShape shape = new TensorShape(batch, seqLen, _hiddenSize);
        TensorShape heads = new TensorShape(batch, seqLen, _numHeads, _headDim);
        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        TensorShape mlpShape = new TensorShape(batch, seqLen, _mlpDim);

        // ── 2. LayerNorm (no affine) + modulate: x*(1+scale)+shift ──
        Tensor modulated = DiTUtils.NormModulate(backend, x, mod[0], mod[1], shape);

        // ── 3+4. Q/K/V + MLP projection + QK-Norm. Fused path (HARTSY_CHROMA_FUSED_QKV): the BFL
        //         linear1 [3*hidden + mlp, hidden] runs as ONE GEMM; qkv + mlp slice off the activation
        //         and QkvSplitNorm does split + per-head RMSNorm + v copy in one kernel (4 GEMMs +
        //         2 RmsNorm passes → 1 GEMM + 3 kernels, the Hunyuan3DFluxBlocks recipe). Split path:
        //         4 parallel Linears + 2 separate RmsNorm passes (Q/K/V declared [B, S, H, D]). ──
        Tensor qNormed, kNormed, v, mlpInput;
        if (_lin1Weight is not null)
        {
            Tensor lin1 = new Tensor(new TensorShape(batch, seqLen, 3 * _hiddenSize + _mlpDim), act);
            backend.Linear(lin1, modulated, _lin1Weight, _lin1Bias);
            modulated.Dispose();
            Tensor qkv = new Tensor(new TensorShape(batch, seqLen, 3 * _hiddenSize), act);
            backend.SliceLastDim(qkv, lin1, 0);
            mlpInput = new Tensor(mlpShape, act);
            backend.SliceLastDim(mlpInput, lin1, 3 * _hiddenSize);
            lin1.Dispose();
            qNormed = new Tensor(heads, act);
            kNormed = new Tensor(heads, act);
            v = new Tensor(heads, act);
            backend.QkvSplitNorm(qNormed, kNormed, v, qkv, _normQ.Weight, _normK.Weight, _normQ.Eps);
            qkv.Dispose();
        }
        else
        {
            Tensor q = new Tensor(heads, act);
            backend.Linear(q, modulated, _toQWeight!, _toQBias);
            Tensor k = new Tensor(heads, act);
            backend.Linear(k, modulated, _toKWeight!, _toKBias);
            v = new Tensor(heads, act);
            backend.Linear(v, modulated, _toVWeight!, _toVBias);
            mlpInput = new Tensor(mlpShape, act);
            backend.Linear(mlpInput, modulated, _projMlpWeight!, _projMlpBias);
            modulated.Dispose();

            qNormed = new Tensor(heads, act);
            backend.RmsNorm(qNormed, q, _normQ.Weight, _normQ.Eps);
            q.Dispose();
            kNormed = new Tensor(heads, act);
            backend.RmsNorm(kNormed, k, _normK.Weight, _normK.Eps);
            k.Dispose();
        }

        // ── 5. RoPE. B=1 (pipeline case): GPU-resident on the pre-permute [B, S, H, D] layout
        //       (device WanRopeInterleaved, no D2H). B>1: host Forward post-permute. ──
        bool gpuRope = batch == 1;
        if (gpuRope)
        {
            rope.ApplyGpu(backend, qNormed, kNormed, _numHeads);
        }

        // ── 6. Permute [B, S, H, D] → [B, H, S, D] ──
        Tensor qMh = new Tensor(mhShape, act);
        backend.Permute0213(qMh, qNormed, seqLen, _numHeads, _headDim);
        qNormed.Dispose();
        Tensor kMh = new Tensor(mhShape, act);
        backend.Permute0213(kMh, kNormed, seqLen, _numHeads, _headDim);
        kNormed.Dispose();
        Tensor vMh = new Tensor(mhShape, act);
        backend.Permute0213(vMh, v, seqLen, _numHeads, _headDim);
        v.Dispose();

        if (!gpuRope)
        {
            rope.Forward(qMh, kMh, batch, _numHeads, seqLen);
        }

        // ── 7. SDPA. The additive [B,1,S,S] mask is built ONCE per forward by ChromaTransformer and shared
        //       across all blocks — use it directly, do NOT dispose here.
        //       allowF16 is safe: Q/K are RMS-normed (QkNorm) so scores are bounded; the mask rides the cuDNN
        //       fused engine as an fp32 bias score-modifier, never rounded through F16. ──
        Tensor attnOut = new Tensor(mhShape, act);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, sdpaMask, scale, allowF16: true);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();

        // ── 8. Permute attention back [B, H, S, D] → [B, S, hidden] ──
        Tensor attnFlat = new Tensor(shape, act);
        backend.Permute0213(attnFlat, attnOut, _numHeads, seqLen, _headDim);
        attnOut.Dispose();

        // ── 9. GELU(tanh) on MLP input ──
        Tensor mlpActivated = new Tensor(mlpShape, act);
        backend.Gelu(mlpActivated, mlpInput);
        mlpInput.Dispose();
        ChromaF16.Probe("sgl-mlpAct", mlpActivated);

        // ── 10. Concat [attn, gelu(mlp)] along the feature dim → proj_out ──
        int concatDim = _hiddenSize + _mlpDim;
        TensorShape concatShape = new TensorShape(batch, seqLen, concatDim);
        Tensor concatted = new Tensor(concatShape, act);
        backend.Concat(concatted, new Tensor[] { attnFlat, mlpActivated }, 2);
        attnFlat.Dispose();
        mlpActivated.Dispose();

        Tensor projected = new Tensor(shape, act);
        backend.Linear(projected, concatted, _projOutWeight!, _projOutBias);
        concatted.Dispose();

        // ── 11. Gated residual: x = x + gate * proj_out ──
        Tensor result = new Tensor(shape, act);
        backend.GatedResidualLastDim(result, x, projected, mod[2]);
        projected.Dispose();

        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();

        ChromaF16.Probe("sgl-out", result);
        return result;
    }
}

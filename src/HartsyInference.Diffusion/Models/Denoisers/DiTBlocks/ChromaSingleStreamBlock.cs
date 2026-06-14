using HartsyInference.Core.Backends;
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
public sealed unsafe class ChromaSingleStreamBlock
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
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _normQ.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);

        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _toQBias = weights[$"{prefix}.attn.to_q.bias"];
        _toKBias = weights[$"{prefix}.attn.to_k.bias"];
        _toVBias = weights[$"{prefix}.attn.to_v.bias"];

        _projMlpWeight = weights[$"{prefix}.proj_mlp.weight"];
        _projMlpBias = weights[$"{prefix}.proj_mlp.bias"];

        _projOutWeight = weights[$"{prefix}.proj_out.weight"];
        _projOutBias = weights[$"{prefix}.proj_out.bias"];
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
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
    }

    /// <summary>Forward pass on the already-concatenated <c>[txt; img]</c> sequence.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="x">Input [B, totalSeqLen, hidden].</param>
    /// <param name="temb">Modulation slice [B, 3, hidden] (shift, scale, gate).</param>
    /// <param name="rope">Precomputed FluxRope for the joint sequence.</param>
    /// <param name="attentionMask">Optional [B, totalSeqLen] mask (1=keep, 0=mask).</param>
    public Tensor Forward(IBackend backend, Tensor x, Tensor temb, FluxRope rope, Tensor? attentionMask)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];

        // ── 1. Slice modulation rows ──
        Tensor[] mod = SliceModRows(temb, batch, rowCount: 3);

        // ── 2. LayerNorm (no affine) + modulate ──
        TensorShape shape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor normed = new Tensor(shape, DType.F32);
        DiTUtils.LayerNormNoAffine(normed, x, batch, seqLen, _hiddenSize);
        Tensor modulated = AdaLNModulation.ApplyModulation(normed, mod[0], mod[1], batch, seqLen, _hiddenSize);
        normed.Dispose();

        // ── 3. Q/K/V + MLP projection (parallel) ──
        Tensor q = new Tensor(shape, DType.F32);
        backend.Linear(q, modulated, _toQWeight!, _toQBias);
        Tensor k = new Tensor(shape, DType.F32);
        backend.Linear(k, modulated, _toKWeight!, _toKBias);
        Tensor v = new Tensor(shape, DType.F32);
        backend.Linear(v, modulated, _toVWeight!, _toVBias);
        TensorShape mlpShape = new TensorShape(batch, seqLen, _mlpDim);
        Tensor mlpInput = new Tensor(mlpShape, DType.F32);
        backend.Linear(mlpInput, modulated, _projMlpWeight!, _projMlpBias);
        modulated.Dispose();

        // ── 4. QK-Norm ──
        int totalVectors = batch * seqLen * _numHeads;
        Tensor qNormed = new Tensor(q.Shape, DType.F32);
        Tensor kNormed = new Tensor(k.Shape, DType.F32);
        _normQ.Forward(qNormed, q, totalVectors);
        _normK.Forward(kNormed, k, totalVectors);
        q.Dispose();
        k.Dispose();

        // ── 5. Reshape to multi-head ──
        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        Tensor qMh = new Tensor(mhShape, DType.F32);
        Tensor kMh = new Tensor(mhShape, DType.F32);
        Tensor vMh = new Tensor(mhShape, DType.F32);
        DiTUtils.ReshapeToMultiHead(qMh, qNormed, batch, seqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(kMh, kNormed, batch, seqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(vMh, v, batch, seqLen, _numHeads, _headDim);
        qNormed.Dispose();
        kNormed.Dispose();
        v.Dispose();

        // ── 6. RoPE ──
        rope.Forward(qMh, kMh, batch, _numHeads, seqLen);

        // ── 7. SDPA (with optional mask) ──
        Tensor? sdpaMask = attentionMask is not null
            ? BuildSdpaMask(attentionMask, batch, seqLen)
            : null;

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, sdpaMask, scale);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();
        sdpaMask?.Dispose();

        // ── 8. Reshape attention back to [B, S, hidden] ──
        Tensor attnFlat = new Tensor(shape, DType.F32);
        DiTUtils.ReshapeFromMultiHead(attnFlat, attnOut, batch, seqLen, _numHeads, _headDim);
        attnOut.Dispose();

        // ── 9. GELU(tanh) on MLP input ──
        Tensor mlpActivated = new Tensor(mlpShape, DType.F32);
        backend.Gelu(mlpActivated, mlpInput);
        mlpInput.Dispose();

        // ── 10. Concat [attn, gelu(mlp)] → proj_out ──
        int concatDim = _hiddenSize + _mlpDim;
        TensorShape concatShape = new TensorShape(batch, seqLen, concatDim);
        Tensor concatted = new Tensor(concatShape, DType.F32);
        ConcatAlongFeatureDim(concatted, attnFlat, mlpActivated, batch, seqLen, _hiddenSize, _mlpDim);
        attnFlat.Dispose();
        mlpActivated.Dispose();

        Tensor projected = new Tensor(shape, DType.F32);
        backend.Linear(projected, concatted, _projOutWeight!, _projOutBias);
        concatted.Dispose();

        // ── 11. Gated residual: x = x + gate * proj_out ──
        Tensor result = AdaLNModulation.ApplyGatedResidual(x, projected, mod[2], batch, seqLen, _hiddenSize);
        projected.Dispose();

        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();

        return result;
    }

    private static Tensor[] SliceModRows(Tensor temb, int batch, int rowCount)
    {
        int totalRows = (int)temb.Shape[1];
        int hidden = (int)temb.Shape[2];
        Tensor[] rows = new Tensor[rowCount];
        float* tembPtr = (float*)temb.DataPointer;

        for (int r = 0; r < rowCount; r++)
        {
            TensorShape rowShape = new TensorShape(batch, hidden);
            Tensor row = new Tensor(rowShape, DType.F32);
            float* rowPtr = (float*)row.DataPointer;

            for (int b = 0; b < batch; b++)
            {
                long src = ((long)b * totalRows + r) * hidden;
                long dst = (long)b * hidden;
                Buffer.MemoryCopy(tembPtr + src, rowPtr + dst, hidden * sizeof(float), hidden * sizeof(float));
            }
            rows[r] = row;
        }
        return rows;
    }

    private static Tensor BuildSdpaMask(Tensor mask, int batch, int seqLen)
    {
        TensorShape outShape = new TensorShape(batch, 1, seqLen, seqLen);
        Tensor outMask = new Tensor(outShape, DType.F32);

        float* mPtr = (float*)mask.DataPointer;
        float* outPtr = (float*)outMask.DataPointer;
        const float NegInf = -1.0e30f;

        for (int b = 0; b < batch; b++)
        {
            int maskOffset = b * seqLen;
            int outOffset = b * seqLen * seqLen;
            for (int q = 0; q < seqLen; q++)
            {
                float qKeep = mPtr[maskOffset + q];
                for (int kk = 0; kk < seqLen; kk++)
                {
                    float kKeep = mPtr[maskOffset + kk];
                    float allowed = qKeep * kKeep;
                    outPtr[outOffset + q * seqLen + kk] = allowed > 0.5f ? 0.0f : NegInf;
                }
            }
        }
        return outMask;
    }

    private static void ConcatAlongFeatureDim(Tensor output, Tensor first, Tensor second,
        int batch, int seqLen, int firstDim, int secondDim)
    {
        float* firstPtr = (float*)first.DataPointer;
        float* secondPtr = (float*)second.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int totalDim = firstDim + secondDim;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int outOffset = (b * seqLen + s) * totalDim;
                int firstOffset = (b * seqLen + s) * firstDim;
                int secondOffset = (b * seqLen + s) * secondDim;

                Buffer.MemoryCopy(firstPtr + firstOffset, outPtr + outOffset,
                    firstDim * sizeof(float), firstDim * sizeof(float));
                Buffer.MemoryCopy(secondPtr + secondOffset, outPtr + outOffset + firstDim,
                    secondDim * sizeof(float), secondDim * sizeof(float));
            }
        }
    }
}

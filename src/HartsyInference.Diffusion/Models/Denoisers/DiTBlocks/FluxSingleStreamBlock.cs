using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Flux SingleStreamBlock: parallel attention + MLP on a concatenated image+text sequence. Uses a fused linear1 for Q/K/V + MLP input, then combines attention output with GELU(MLP) via linear2. No SD3 equivalent exists.</summary>
public sealed unsafe class FluxSingleStreamBlock
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
    public Tensor Forward(IBackend backend, Tensor x, Tensor temb, FluxRope rope, Tensor? attnBias = null)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];

        // ── 1. Modulation: 3 params (shift, scale, gate) ──
        Tensor[] mod = _modulation.Forward(backend, temb);

        // ── 2. LayerNorm (no affine) + modulate ──
        TensorShape shape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor normed = new Tensor(shape, DType.F32);
        LayerNormNoAffine(normed, x, batch, seqLen, _hiddenSize);
        Tensor modulated = AdaLNModulation.ApplyModulation(normed, mod[0], mod[1], batch, seqLen, _hiddenSize);
        normed.Dispose();

        // ── 3. Q/K/V projections + MLP projection (parallel) ──
        Tensor q = new Tensor(shape, DType.F32);
        backend.Linear(q, modulated, _toQWeight!, _toQBias);
        Tensor k = new Tensor(shape, DType.F32);
        backend.Linear(k, modulated, _toKWeight!, _toKBias);
        Tensor v = new Tensor(shape, DType.F32);
        backend.Linear(v, modulated, _toVWeight!, _toVBias);
        // mlpInput / mlpActivated / concatted run at F16. At 1024x1024 these are the three
        // largest activations in the block (213/213/267 MB at F32) and dominate VRAM peak.
        // F16 halves them and lets the next Linear skip its input cast since the joint
        // dtype is already F16. The proj_mlp Linear writes F16 directly into mlpInput;
        // cuBLAS still accumulates in F32 internally so accuracy is preserved.
        TensorShape mlpProjShape = new TensorShape(batch, seqLen, _mlpDim);
        Tensor mlpInput = new Tensor(mlpProjShape, DType.F16);
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

        // ── 5. Reshape to multi-head [B, H, S, D] ──
        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        Tensor qMh = new Tensor(mhShape, DType.F32);
        Tensor kMh = new Tensor(mhShape, DType.F32);
        Tensor vMh = new Tensor(mhShape, DType.F32);
        ReshapeToMultiHead(qMh, qNormed, batch, seqLen, _numHeads, _headDim);
        ReshapeToMultiHead(kMh, kNormed, batch, seqLen, _numHeads, _headDim);
        ReshapeToMultiHead(vMh, v, batch, seqLen, _numHeads, _headDim);
        qNormed.Dispose();
        kNormed.Dispose();
        v.Dispose();

        // ── 6. Apply RoPE ──
        rope.Forward(qMh, kMh, batch, _numHeads, seqLen);

        // ── 7. SDPA ──
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, attnBias, scale);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();

        // ── 8. Reshape attention output back to [B, S, hidden] ──
        Tensor attnFlat = new Tensor(shape, DType.F32);
        ReshapeFromMultiHead(attnFlat, attnOut, batch, seqLen, _numHeads, _headDim);
        attnOut.Dispose();

        // ── 9. GELU(tanh) on MLP input ── (F16, matches mlpInput; CudaBackend.Gelu has an F16 path)
        TensorShape mlpShape = new TensorShape(batch, seqLen, _mlpDim);
        Tensor mlpActivated = new Tensor(mlpShape, DType.F16);
        backend.Gelu(mlpActivated, mlpInput);
        mlpInput.Dispose();

        // ── 10. Concatenate [attn_out, gelu(mlp)] → proj_out ──
        // attnFlat is F32 (the attention path stays F32 — SDPA is fine in F32 and the
        // attention activations are 4x smaller than the MLP ones). Cast it down to F16
        // so concat operands match. The cast is small (53 MB at 1024x1024), a worthwhile
        // tradeoff for a 134 MB concatted buffer at F16 instead of 267 MB at F32.
        Tensor attnFlatF16 = Utilities.DtypeCastHelper.EnsureDtype(backend, attnFlat, DType.F16);

        int concatDim = _hiddenSize + _mlpDim;
        TensorShape concatShape = new TensorShape(batch, seqLen, concatDim);
        Tensor concatted = new Tensor(concatShape, DType.F16);
        ConcatAlongFeatureDim(concatted, attnFlatF16, mlpActivated, batch, seqLen, _hiddenSize, _mlpDim);
        attnFlatF16.Dispose();
        mlpActivated.Dispose();

        // proj_out's input is F16 (concatted) and its weight is fp8/F16 — joint resolution
        // picks F16 with no input cast needed. Output stays F32 because the gated residual
        // at the end of the block adds it to x (F32).
        Tensor output = new Tensor(shape, DType.F32);
        backend.Linear(output, concatted, _projOutWeight!, _projOutBias);
        concatted.Dispose();

        // ── 11. Gated residual: x = x + gate * output ──
        Tensor result = AdaLNModulation.ApplyGatedResidual(x, output, mod[2], batch, seqLen, _hiddenSize);
        output.Dispose();

        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();

        return result;
    }

    // ── Helper methods ──

    private static void LayerNormNoAffine(Tensor output, Tensor input, int batch, int seqLen, int dim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int offset = (b * seqLen + s) * dim;

                float mean = 0f;
                for (int d = 0; d < dim; d++)
                    mean += inPtr[offset + d];
                mean /= dim;

                float variance = 0f;
                for (int d = 0; d < dim; d++)
                {
                    float diff = inPtr[offset + d] - mean;
                    variance += diff * diff;
                }
                variance /= dim;

                float invStd = 1.0f / MathF.Sqrt(variance + 1e-6f);
                for (int d = 0; d < dim; d++)
                    outPtr[offset + d] = (inPtr[offset + d] - mean) * invStd;
            }
        }
    }

    private static void ReshapeToMultiHead(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int inOffset = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    int outOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    Buffer.MemoryCopy(inPtr + inOffset, outPtr + outOffset, headDim * sizeof(float), headDim * sizeof(float));
                }
            }
        }
    }

    private static void ReshapeFromMultiHead(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int inOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    int outOffset = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    Buffer.MemoryCopy(inPtr + inOffset, outPtr + outOffset, headDim * sizeof(float), headDim * sizeof(float));
                }
            }
        }
    }

    /// <summary>Concatenates two tensors along the feature dimension: [B, S, D1] + [B, S, D2] → [B, S, D1+D2].
    /// Operates on raw bytes via Tensor.DataPointer — all three tensors must share the same element
    /// size (validated at the top). Works for F32 and F16 alike.</summary>
    private static void ConcatAlongFeatureDim(Tensor output, Tensor first, Tensor second,
        int batch, int seqLen, int firstDim, int secondDim)
    {
        int elemSize = first.DType.SizeInBytes;
        if (second.DType.SizeInBytes != elemSize || output.DType.SizeInBytes != elemSize)
        {
            throw new InvalidOperationException(
                $"ConcatAlongFeatureDim requires identical element size; got first={first.DType}, second={second.DType}, output={output.DType}");
        }
        byte* firstPtr = (byte*)first.DataPointer;
        byte* secondPtr = (byte*)second.DataPointer;
        byte* outPtr = (byte*)output.DataPointer;
        int totalDim = firstDim + secondDim;
        long firstRowBytes = (long)firstDim * elemSize;
        long secondRowBytes = (long)secondDim * elemSize;
        long totalRowBytes = (long)totalDim * elemSize;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                long row = (long)b * seqLen + s;
                long outOffset = row * totalRowBytes;
                long firstOffset = row * firstRowBytes;
                long secondOffset = row * secondRowBytes;

                Buffer.MemoryCopy(firstPtr + firstOffset, outPtr + outOffset, firstRowBytes, firstRowBytes);
                Buffer.MemoryCopy(secondPtr + secondOffset, outPtr + outOffset + firstRowBytes, secondRowBytes, secondRowBytes);
            }
        }
    }
}

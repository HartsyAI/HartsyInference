using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

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

    /// <summary>Forward pass on concatenated [text, image] sequence. Parallel attention + MLP.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="x">Input [B, totalSeqLen, hidden] (text + image concatenated).</param>
    /// <param name="temb">Timestep embedding [B, hidden].</param>
    /// <param name="rope">Precomputed FluxRope.</param>
    /// <returns>Updated x tensor [B, totalSeqLen, hidden].</returns>
    public Tensor Forward(IBackend backend, Tensor x, Tensor temb, FluxRope rope)
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
        Tensor q = LinearProject(modulated, _toQWeight!, _toQBias, batch, seqLen, _hiddenSize, _hiddenSize);
        Tensor k = LinearProject(modulated, _toKWeight!, _toKBias, batch, seqLen, _hiddenSize, _hiddenSize);
        Tensor v = LinearProject(modulated, _toVWeight!, _toVBias, batch, seqLen, _hiddenSize, _hiddenSize);
        Tensor mlpInput = LinearProject(modulated, _projMlpWeight!, _projMlpBias, batch, seqLen, _hiddenSize, _mlpDim);
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
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, null, scale);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();

        // ── 8. Reshape attention output back to [B, S, hidden] ──
        Tensor attnFlat = new Tensor(shape, DType.F32);
        ReshapeFromMultiHead(attnFlat, attnOut, batch, seqLen, _numHeads, _headDim);
        attnOut.Dispose();

        // ── 9. GELU(tanh) on MLP input ──
        TensorShape mlpShape = new TensorShape(batch, seqLen, _mlpDim);
        Tensor mlpActivated = new Tensor(mlpShape, DType.F32);
        backend.Gelu(mlpActivated, mlpInput);
        mlpInput.Dispose();

        // ── 10. Concatenate [attn_out, gelu(mlp)] → proj_out ──
        int concatDim = _hiddenSize + _mlpDim;
        TensorShape concatShape = new TensorShape(batch, seqLen, concatDim);
        Tensor concatted = new Tensor(concatShape, DType.F32);
        ConcatAlongFeatureDim(concatted, attnFlat, mlpActivated, batch, seqLen, _hiddenSize, _mlpDim);
        attnFlat.Dispose();
        mlpActivated.Dispose();

        Tensor output = LinearProject(concatted, _projOutWeight!, _projOutBias, batch, seqLen, concatDim, _hiddenSize);
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

    private static Tensor LinearProject(Tensor input, Tensor weight, Tensor? bias, int batch, int seqLen, int inDim, int outDim)
    {
        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor output = new Tensor(outShape, DType.F32);

        float* inPtr = (float*)input.DataPointer;
        float* wPtr = (float*)weight.DataPointer;
        float* bPtr = bias != null ? (float*)bias.DataPointer : null;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int inOffset = (b * seqLen + s) * inDim;
                int outOffset = (b * seqLen + s) * outDim;
                for (int o = 0; o < outDim; o++)
                {
                    float sum = bPtr != null ? bPtr[o] : 0f;
                    int wOffset = o * inDim;
                    for (int i = 0; i < inDim; i++)
                        sum += inPtr[inOffset + i] * wPtr[wOffset + i];
                    outPtr[outOffset + o] = sum;
                }
            }
        }

        return output;
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

    /// <summary>Concatenates two tensors along the feature dimension: [B, S, D1] + [B, S, D2] → [B, S, D1+D2].</summary>
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

using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.TextEncoders;

/// <summary>Single T5 encoder block: RMSNorm → SelfAttention → Residual → RMSNorm → GEGLU FFN → Residual. All linear layers are bias-free (T5 v1.1 convention).</summary>
public sealed unsafe class T5Block
{
    private readonly int _dModel;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly float _eps;

    // Self-attention sublayer
    private Tensor? _attnNormWeight;
    private Tensor? _qWeight;
    private Tensor? _kWeight;
    private Tensor? _vWeight;
    private Tensor? _oWeight;

    // GEGLU FFN sublayer
    private Tensor? _ffnNormWeight;
    private Tensor? _wi0Weight;  // gate projection (goes through GeLU)
    private Tensor? _wi1Weight;  // linear projection
    private Tensor? _woWeight;   // output projection

    public T5Block(int dModel, int numHeads, int headDim, float eps)
    {
        _dModel = dModel;
        _numHeads = numHeads;
        _headDim = headDim;
        _eps = eps;
    }

    /// <summary>Loads weights for this block from named tensors using T5 HuggingFace naming.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        // Attention sublayer (layer.0)
        // Norm weights must be F32 — they go through backend.RmsNorm with F32 input
        _attnNormWeight = EnsureF32(weights[$"{prefix}.layer.0.layer_norm.weight"]);
        _qWeight = weights[$"{prefix}.layer.0.SelfAttention.q.weight"];
        _kWeight = weights[$"{prefix}.layer.0.SelfAttention.k.weight"];
        _vWeight = weights[$"{prefix}.layer.0.SelfAttention.v.weight"];
        _oWeight = weights[$"{prefix}.layer.0.SelfAttention.o.weight"];

        // FFN sublayer (layer.1) — named DenseReluDense despite using GEGLU in v1.1
        _ffnNormWeight = EnsureF32(weights[$"{prefix}.layer.1.layer_norm.weight"]);
        _wi0Weight = weights[$"{prefix}.layer.1.DenseReluDense.wi_0.weight"];
        _wi1Weight = weights[$"{prefix}.layer.1.DenseReluDense.wi_1.weight"];
        _woWeight = weights[$"{prefix}.layer.1.DenseReluDense.wo.weight"];
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_attnNormWeight is not null) yield return _attnNormWeight;
        if (_qWeight is not null) yield return _qWeight;
        if (_kWeight is not null) yield return _kWeight;
        if (_vWeight is not null) yield return _vWeight;
        if (_oWeight is not null) yield return _oWeight;
        if (_ffnNormWeight is not null) yield return _ffnNormWeight;
        if (_wi0Weight is not null) yield return _wi0Weight;
        if (_wi1Weight is not null) yield return _wi1Weight;
        if (_woWeight is not null) yield return _woWeight;
    }

    /// <summary>Forward pass: input [B, seqLen, dModel] + positionBias [1, numHeads, seqLen, seqLen] + optional attentionMask → output [B, seqLen, dModel].</summary>
    public Tensor Forward(IBackend backend, Tensor input, Tensor positionBias, Tensor? attentionMask)
    {
        int batch = (int)input.Shape[0];
        int seqLen = (int)input.Shape[1];
        TensorShape hidShape = new TensorShape(batch, seqLen, _dModel);

        // --- Self-Attention sublayer ---

        // RMSNorm
        Tensor normed = new Tensor(hidShape, DType.F32);
        backend.RmsNorm(normed, input, _attnNormWeight!, _eps);

        // Q, K, V projections (no bias) — weight shape: [dModel, dModel]
        Tensor query = new Tensor(hidShape, DType.F32);
        backend.Linear(query, normed, _qWeight!, null);
        Tensor key = new Tensor(hidShape, DType.F32);
        backend.Linear(key, normed, _kWeight!, null);
        Tensor value = new Tensor(hidShape, DType.F32);
        backend.Linear(value, normed, _vWeight!, null);
        normed.Dispose();

        // Reshape to multi-head: [B, seqLen, dModel] → [B, numHeads, seqLen, headDim]
        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        Tensor queryMh = new Tensor(mhShape, DType.F32);
        Tensor keyMh = new Tensor(mhShape, DType.F32);
        Tensor valueMh = new Tensor(mhShape, DType.F32);

        ReshapeToMultiHead(queryMh, query, batch, seqLen, _numHeads, _headDim);
        ReshapeToMultiHead(keyMh, key, batch, seqLen, _numHeads, _headDim);
        ReshapeToMultiHead(valueMh, value, batch, seqLen, _numHeads, _headDim);
        query.Dispose();
        key.Dispose();
        value.Dispose();

        // Attention: scores = Q @ K^T / sqrt(headDim) + positionBias [+ attentionMask]
        // Build combined mask: positionBias + attentionMask
        Tensor? combinedMask = BuildAttentionMask(positionBias, attentionMask, batch, seqLen);

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, queryMh, keyMh, valueMh, combinedMask, scale);
        queryMh.Dispose();
        keyMh.Dispose();
        valueMh.Dispose();
        combinedMask?.Dispose();

        // Reshape back: [B, numHeads, seqLen, headDim] → [B, seqLen, dModel]
        Tensor merged = new Tensor(hidShape, DType.F32);
        ReshapeFromMultiHead(merged, attnOut, batch, seqLen, _numHeads, _headDim);
        attnOut.Dispose();

        // Output projection (no bias)
        Tensor attnProjected = new Tensor(hidShape, DType.F32);
        backend.Linear(attnProjected, merged, _oWeight!, null);
        merged.Dispose();

        // Residual connection
        Tensor attnResidual = new Tensor(hidShape, DType.F32);
        backend.Add(attnResidual, input, attnProjected);
        attnProjected.Dispose();

        // --- GEGLU FFN sublayer ---

        // RMSNorm
        Tensor ffnNormed = new Tensor(hidShape, DType.F32);
        backend.RmsNorm(ffnNormed, attnResidual, _ffnNormWeight!, _eps);

        // GEGLU: gate = GeLU(x @ wi_0^T), linear = x @ wi_1^T, output = (gate * linear) @ wo^T
        int dFf = (int)_wi0Weight!.Shape[0]; // wi_0 is [dFf, dModel]
        TensorShape ffShape = new TensorShape(batch, seqLen, dFf);

        Tensor gateProj = new Tensor(ffShape, DType.F32);
        backend.Linear(gateProj, ffnNormed, _wi0Weight!, null);
        Tensor linearProj = new Tensor(ffShape, DType.F32);
        backend.Linear(linearProj, ffnNormed, _wi1Weight!, null);
        ffnNormed.Dispose();

        // Apply GeLU to gate projection
        Tensor gateActivated = new Tensor(ffShape, DType.F32);
        backend.Gelu(gateActivated, gateProj);
        gateProj.Dispose();

        // Element-wise multiply: gated = gelu(gate) * linear
        Tensor gated = new Tensor(ffShape, DType.F32);
        backend.Mul(gated, gateActivated, linearProj);
        gateActivated.Dispose();
        linearProj.Dispose();

        // Output projection
        Tensor ffnOutput = new Tensor(hidShape, DType.F32);
        backend.Linear(ffnOutput, gated, _woWeight!, null);
        gated.Dispose();

        // Residual connection
        Tensor output = new Tensor(hidShape, DType.F32);
        backend.Add(output, attnResidual, ffnOutput);
        attnResidual.Dispose();
        ffnOutput.Dispose();

        return output;
    }

    /// <summary>Builds combined attention mask from position bias and optional padding mask. Position bias is additive to attention scores. Padding mask uses large negative values for masked positions.</summary>
    private Tensor? BuildAttentionMask(Tensor positionBias, Tensor? attentionMask, int batch, int seqLen)
    {
        if (attentionMask is null)
        {
            // Just broadcast position bias to batch size if needed
            if (batch == 1)
                return positionBias;

            // Expand [1, H, S, S] to [B, H, S, S] by copying
            TensorShape maskShape = new TensorShape(batch, _numHeads, seqLen, seqLen);
            Tensor expanded = new Tensor(maskShape, DType.F32);
            float* srcPtr = (float*)positionBias.DataPointer;
            float* dstPtr = (float*)expanded.DataPointer;
            int singleBatchSize = _numHeads * seqLen * seqLen;

            for (int b = 0; b < batch; b++)
            {
                Buffer.MemoryCopy(srcPtr, dstPtr + b * singleBatchSize, singleBatchSize * sizeof(float), singleBatchSize * sizeof(float));
            }
            return expanded;
        }

        // Combine position bias with attention mask
        // attentionMask is [B, seqLen] with 1=attend, 0=mask
        // Convert to [B, 1, 1, seqLen] with 0=attend, -1e9=mask, then add to position bias
        TensorShape combinedShape = new TensorShape(batch, _numHeads, seqLen, seqLen);
        Tensor combined = new Tensor(combinedShape, DType.F32);

        float* biasPtr = (float*)positionBias.DataPointer;
        float* maskPtr = (float*)attentionMask.DataPointer;
        float* outPtr = (float*)combined.DataPointer;
        int headSeqSq = seqLen * seqLen;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < _numHeads; h++)
            {
                for (int q = 0; q < seqLen; q++)
                {
                    for (int k = 0; k < seqLen; k++)
                    {
                        // Position bias is [1, H, S, S]
                        float bias = biasPtr[h * headSeqSq + q * seqLen + k];
                        // Attention mask is [B, seqLen] — mask the key positions
                        float maskVal = maskPtr[b * seqLen + k];
                        float maskAdditional = maskVal > 0.5f ? 0.0f : -1e9f;
                        outPtr[(b * _numHeads + h) * headSeqSq + q * seqLen + k] = bias + maskAdditional;
                    }
                }
            }
        }

        return combined;
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
                    for (int d = 0; d < headDim; d++)
                    {
                        outPtr[outOffset + d] = inPtr[inOffset + d];
                    }
                }
            }
        }
    }

    private static Tensor EnsureF32(Tensor tensor) =>
        tensor.DType != DType.F32 ? tensor.CastTo(DType.F32) : tensor;

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
                    for (int d = 0; d < headDim; d++)
                    {
                        outPtr[outOffset + d] = inPtr[inOffset + d];
                    }
                }
            }
        }
    }
}

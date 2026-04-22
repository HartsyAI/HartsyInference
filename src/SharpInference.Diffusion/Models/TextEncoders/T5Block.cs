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
        _attnNormWeight = weights[$"{prefix}.layer.0.layer_norm.weight"];
        _qWeight = weights[$"{prefix}.layer.0.SelfAttention.q.weight"];
        _kWeight = weights[$"{prefix}.layer.0.SelfAttention.k.weight"];
        _vWeight = weights[$"{prefix}.layer.0.SelfAttention.v.weight"];
        _oWeight = weights[$"{prefix}.layer.0.SelfAttention.o.weight"];

        // FFN sublayer (layer.1) — named DenseReluDense despite using GEGLU in v1.1
        _ffnNormWeight = weights[$"{prefix}.layer.1.layer_norm.weight"];
        _wi0Weight = weights[$"{prefix}.layer.1.DenseReluDense.wi_0.weight"];
        _wi1Weight = weights[$"{prefix}.layer.1.DenseReluDense.wi_1.weight"];
        _woWeight = weights[$"{prefix}.layer.1.DenseReluDense.wo.weight"];
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
        Tensor query = LinearNoBias(normed, _qWeight!, batch, seqLen, _dModel, _dModel);
        Tensor key = LinearNoBias(normed, _kWeight!, batch, seqLen, _dModel, _dModel);
        Tensor value = LinearNoBias(normed, _vWeight!, batch, seqLen, _dModel, _dModel);
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
        Tensor attnProjected = LinearNoBias(merged, _oWeight!, batch, seqLen, _dModel, _dModel);
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

        Tensor gateProj = LinearNoBias(ffnNormed, _wi0Weight!, batch, seqLen, _dModel, dFf);
        Tensor linearProj = LinearNoBias(ffnNormed, _wi1Weight!, batch, seqLen, _dModel, dFf);
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
        Tensor ffnOutput = LinearNoBias(gated, _woWeight!, batch, seqLen, dFf, _dModel);
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

    /// <summary>Matrix multiply without bias: output = input @ weight^T. Weight shape: [outDim, inDim].</summary>
    private static Tensor LinearNoBias(Tensor input, Tensor weight, int batch, int seqLen, int inDim, int outDim)
    {
        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor output = new Tensor(outShape, DType.F32);

        float* inPtr = (float*)input.DataPointer;
        float* wPtr = (float*)weight.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int inOffset = (b * seqLen + s) * inDim;
                int outOffset = (b * seqLen + s) * outDim;
                for (int o = 0; o < outDim; o++)
                {
                    float sum = 0f;
                    int wOffset = o * inDim;
                    for (int i = 0; i < inDim; i++)
                    {
                        sum += inPtr[inOffset + i] * wPtr[wOffset + i];
                    }
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
                    for (int d = 0; d < headDim; d++)
                    {
                        outPtr[outOffset + d] = inPtr[inOffset + d];
                    }
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
                    for (int d = 0; d < headDim; d++)
                    {
                        outPtr[outOffset + d] = inPtr[inOffset + d];
                    }
                }
            }
        }
    }
}

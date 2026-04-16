using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.UNetBlocks;

/// <summary>Transformer block for UNet cross-attention: LayerNorm→SelfAttn→Residual→LayerNorm→CrossAttn→Residual→LayerNorm→FFN→Residual. Matches diffusers BasicTransformerBlock.</summary>
public sealed unsafe class CrossAttentionBlock
{
    private readonly int _channels;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _crossAttentionDim;

    // Input projection: GroupNorm + linear proj_in (or conv 1x1)
    private Tensor? _normWeight;
    private Tensor? _normBias;
    private Tensor? _projInWeight;
    private Tensor? _projInBias;

    // Transformer block inner layers
    private readonly TransformerSubBlock _selfAttn;
    private readonly TransformerSubBlock _crossAttn;
    private readonly FeedForwardBlock _ffn;

    // Output projection
    private Tensor? _projOutWeight;
    private Tensor? _projOutBias;

    /// <summary>Creates a cross-attention block.</summary>
    /// <param name="channels">Number of input/output channels.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="crossAttentionDim">Dimension of the cross-attention context (text encoder hidden size).</param>
    /// <param name="normGroups">Number of groups for the input GroupNorm.</param>
    /// <param name="normEps">GroupNorm epsilon.</param>
    public CrossAttentionBlock(int channels, int numHeads, int crossAttentionDim, int normGroups = 32, float normEps = 1e-5f)
    {
        _channels = channels;
        _numHeads = numHeads;
        _headDim = channels / numHeads;
        _crossAttentionDim = crossAttentionDim;

        _selfAttn = new TransformerSubBlock(channels, numHeads, channels);
        _crossAttn = new TransformerSubBlock(channels, numHeads, crossAttentionDim);
        _ffn = new FeedForwardBlock(channels);
    }

    /// <summary>Loads weights from named tensors.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _normWeight = weights[$"{prefix}.norm.weight"];
        _normBias = weights[$"{prefix}.norm.bias"];
        _projInWeight = weights[$"{prefix}.proj_in.weight"];
        _projInBias = weights[$"{prefix}.proj_in.bias"];

        _selfAttn.LoadWeights(weights, $"{prefix}.transformer_blocks.0.attn1", $"{prefix}.transformer_blocks.0.norm1");
        _crossAttn.LoadWeights(weights, $"{prefix}.transformer_blocks.0.attn2", $"{prefix}.transformer_blocks.0.norm2");
        _ffn.LoadWeights(weights, $"{prefix}.transformer_blocks.0.ff", $"{prefix}.transformer_blocks.0.norm3");

        _projOutWeight = weights[$"{prefix}.proj_out.weight"];
        _projOutBias = weights[$"{prefix}.proj_out.bias"];
    }

    /// <summary>Forward pass: input [B, C, H, W] + context [B, seqLen, crossDim] → output [B, C, H, W].</summary>
    public Tensor Forward(IBackend backend, Tensor input, Tensor context)
    {
        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int height = (int)input.Shape[2];
        int width = (int)input.Shape[3];
        int spatial = height * width;

        // 1. GroupNorm on spatial input
        TensorShape spatialShape = new TensorShape(batch, channels, height, width);
        Tensor normed = new Tensor(spatialShape, DType.F32);
        backend.GroupNorm(normed, input, _normWeight!, _normBias!, 32, 1e-6f);

        // 2. Reshape [B, C, H, W] → [B, H*W, C] and project in
        TensorShape seqShape = new TensorShape(batch, spatial, channels);
        Tensor hidden = new Tensor(seqShape, DType.F32);
        ReshapeSpatialToSequence(hidden, normed, batch, channels, spatial);
        normed.Dispose();

        // proj_in: [B, spatial, C] → [B, spatial, C]
        Tensor projected = LinearProjectBatched(hidden, _projInWeight!, _projInBias!, batch, spatial, channels, channels);
        hidden.Dispose();
        hidden = projected;

        // 3. Self-attention (context = self)
        Tensor selfOut = _selfAttn.Forward(backend, hidden, hidden);
        hidden.Dispose();
        hidden = selfOut;

        // 4. Cross-attention (context = text embeddings)
        Tensor crossOut = _crossAttn.Forward(backend, hidden, context);
        hidden.Dispose();
        hidden = crossOut;

        // 5. Feed-forward
        Tensor ffnOut = _ffn.Forward(backend, hidden);
        hidden.Dispose();
        hidden = ffnOut;

        // 6. proj_out: [B, spatial, C] → [B, spatial, C]
        Tensor projOut = LinearProjectBatched(hidden, _projOutWeight!, _projOutBias!, batch, spatial, channels, channels);
        hidden.Dispose();

        // 7. Reshape [B, H*W, C] → [B, C, H, W] and add residual
        Tensor output = new Tensor(spatialShape, DType.F32);
        ReshapeSequenceToSpatial(output, projOut, batch, channels, spatial);
        projOut.Dispose();

        // Add residual from input
        Tensor result = new Tensor(spatialShape, DType.F32);
        backend.Add(result, output, input);
        output.Dispose();

        return result;
    }

    /// <summary>Linear projection for batched sequence: output = input @ weight^T + bias.</summary>
    private static Tensor LinearProjectBatched(Tensor input, Tensor weight, Tensor bias, int batch, int seqLen, int inDim, int outDim)
    {
        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor output = new Tensor(outShape, DType.F32);

        float* inPtr = (float*)input.DataPointer;
        float* wPtr = (float*)weight.DataPointer;
        float* bPtr = (float*)bias.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        // weight is [outDim, inDim]
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int inOffset = (b * seqLen + s) * inDim;
                int outOffset = (b * seqLen + s) * outDim;
                for (int o = 0; o < outDim; o++)
                {
                    float sum = bPtr[o];
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

    /// <summary>Reshapes [B, C, H*W] → [B, H*W, C] via transposed copy.</summary>
    private static void ReshapeSpatialToSequence(Tensor output, Tensor input, int batch, int channels, int spatial)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int s = 0; s < spatial; s++)
                {
                    outPtr[(b * spatial + s) * channels + c] = inPtr[(b * channels + c) * spatial + s];
                }
            }
        }
    }

    /// <summary>Reshapes [B, H*W, C] → [B, C, H*W] via transposed copy.</summary>
    private static void ReshapeSequenceToSpatial(Tensor output, Tensor input, int batch, int channels, int spatial)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < spatial; s++)
            {
                for (int c = 0; c < channels; c++)
                {
                    outPtr[(b * channels + c) * spatial + s] = inPtr[(b * spatial + s) * channels + c];
                }
            }
        }
    }
}

/// <summary>Attention sub-block: LayerNorm → MultiHeadAttention → Residual. Used for both self-attention and cross-attention.</summary>
internal sealed unsafe class TransformerSubBlock
{
    private readonly int _channels;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _contextDim;

    // LayerNorm
    private Tensor? _normWeight;
    private Tensor? _normBias;

    // Q from hidden, K/V from context
    private Tensor? _toQWeight;
    private Tensor? _toQBias;
    private Tensor? _toKWeight;
    private Tensor? _toKBias;
    private Tensor? _toVWeight;
    private Tensor? _toVBias;
    private Tensor? _toOutWeight;
    private Tensor? _toOutBias;

    public TransformerSubBlock(int channels, int numHeads, int contextDim)
    {
        _channels = channels;
        _numHeads = numHeads;
        _headDim = channels / numHeads;
        _contextDim = contextDim;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string attnPrefix, string normPrefix)
    {
        _normWeight = weights[$"{normPrefix}.weight"];
        _normBias = weights[$"{normPrefix}.bias"];
        _toQWeight = weights[$"{attnPrefix}.to_q.weight"];
        _toKWeight = weights[$"{attnPrefix}.to_k.weight"];
        _toVWeight = weights[$"{attnPrefix}.to_v.weight"];
        _toOutWeight = weights[$"{attnPrefix}.to_out.0.weight"];
        _toOutBias = weights[$"{attnPrefix}.to_out.0.bias"];

        // Q bias and K/V bias may not exist in some models — use zero if absent
        weights.TryGetValue($"{attnPrefix}.to_q.bias", out _toQBias);
        weights.TryGetValue($"{attnPrefix}.to_k.bias", out _toKBias);
        weights.TryGetValue($"{attnPrefix}.to_v.bias", out _toVBias);
    }

    /// <summary>Forward: hidden [B, seqLen, C] + context [B, ctxLen, ctxDim] → output [B, seqLen, C] with residual.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor context)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];
        int ctxLen = (int)context.Shape[1];

        // LayerNorm
        TensorShape hidShape = new TensorShape(batch, seqLen, _channels);
        Tensor normed = new Tensor(hidShape, DType.F32);
        backend.LayerNorm(normed, hidden, _normWeight!, _normBias!, 1e-5f);

        // Q from normed hidden, K/V from context
        Tensor query = LinearProject(normed, _toQWeight!, _toQBias, batch, seqLen, _channels, _channels);
        Tensor key = LinearProject(context, _toKWeight!, _toKBias, batch, ctxLen, _contextDim, _channels);
        Tensor value = LinearProject(context, _toVWeight!, _toVBias, batch, ctxLen, _contextDim, _channels);
        normed.Dispose();

        // Reshape to multi-head 4D: [B, S, numHeads*headDim] → [B, numHeads, S, headDim]
        TensorShape qMhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        TensorShape kvMhShape = new TensorShape(batch, _numHeads, ctxLen, _headDim);
        Tensor queryMh = new Tensor(qMhShape, DType.F32);
        Tensor keyMh = new Tensor(kvMhShape, DType.F32);
        Tensor valueMh = new Tensor(kvMhShape, DType.F32);

        ReshapeToMultiHead(queryMh, query, batch, seqLen, _numHeads, _headDim);
        ReshapeToMultiHead(keyMh, key, batch, ctxLen, _numHeads, _headDim);
        ReshapeToMultiHead(valueMh, value, batch, ctxLen, _numHeads, _headDim);
        query.Dispose();
        key.Dispose();
        value.Dispose();

        // SDPA
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(qMhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, queryMh, keyMh, valueMh, null, scale);
        queryMh.Dispose();
        keyMh.Dispose();
        valueMh.Dispose();

        // Reshape back: [B, numHeads, seqLen, headDim] → [B, seqLen, numHeads*headDim]
        Tensor merged = new Tensor(hidShape, DType.F32);
        ReshapeFromMultiHead(merged, attnOut, batch, seqLen, _numHeads, _headDim);
        attnOut.Dispose();

        // Output projection
        Tensor projected = LinearProject(merged, _toOutWeight!, _toOutBias, batch, seqLen, _channels, _channels);
        merged.Dispose();

        // Residual
        Tensor output = new Tensor(hidShape, DType.F32);
        backend.Add(output, hidden, projected);
        projected.Dispose();

        return output;
    }

    private static Tensor LinearProject(Tensor input, Tensor weight, Tensor? bias, int batch, int seqLen, int inDim, int outDim)
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

        // Add bias if present
        if (bias is not null)
        {
            float* bPtr = (float*)bias.DataPointer;
            float* oPtr = (float*)output.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    int offset = (b * seqLen + s) * outDim;
                    for (int d = 0; d < outDim; d++)
                    {
                        oPtr[offset + d] += bPtr[d];
                    }
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

/// <summary>Feed-forward network: LayerNorm → Linear → GEGLU → Linear → Residual.</summary>
internal sealed unsafe class FeedForwardBlock
{
    private readonly int _channels;

    // LayerNorm
    private Tensor? _normWeight;
    private Tensor? _normBias;

    // GEGLU: net.0.proj (projects to 2 * innerDim for gating)
    private Tensor? _geGluProjWeight;
    private Tensor? _geGluProjBias;

    // Output linear: net.2
    private Tensor? _outLinearWeight;
    private Tensor? _outLinearBias;

    public FeedForwardBlock(int channels)
    {
        _channels = channels;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string ffPrefix, string normPrefix)
    {
        _normWeight = weights[$"{normPrefix}.weight"];
        _normBias = weights[$"{normPrefix}.bias"];
        _geGluProjWeight = weights[$"{ffPrefix}.net.0.proj.weight"];
        _geGluProjBias = weights[$"{ffPrefix}.net.0.proj.bias"];
        _outLinearWeight = weights[$"{ffPrefix}.net.2.weight"];
        _outLinearBias = weights[$"{ffPrefix}.net.2.bias"];
    }

    /// <summary>Forward: hidden [B, seqLen, C] → output [B, seqLen, C] with residual.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];

        // LayerNorm
        TensorShape shape = new TensorShape(batch, seqLen, _channels);
        Tensor normed = new Tensor(shape, DType.F32);
        backend.LayerNorm(normed, hidden, _normWeight!, _normBias!, 1e-5f);

        // GEGLU: project to 2*innerDim, split, gate
        // innerDim = channels * 4 (diffusers default mult=4)
        int innerDim = _channels * 4;
        int geGluOutDim = innerDim * 2;

        TensorShape geGluShape = new TensorShape(batch, seqLen, geGluOutDim);
        Tensor geGluOut = new Tensor(geGluShape, DType.F32);
        LinearForward(geGluOut, normed, _geGluProjWeight!, _geGluProjBias!, batch, seqLen, _channels, geGluOutDim);
        normed.Dispose();

        // Split and apply GELU gate: output = x[:innerDim] * GELU(x[innerDim:])
        TensorShape innerShape = new TensorShape(batch, seqLen, innerDim);
        Tensor gated = new Tensor(innerShape, DType.F32);
        ApplyGeGlu(gated, geGluOut, batch, seqLen, innerDim);
        geGluOut.Dispose();

        // Output linear: [B, seqLen, innerDim] → [B, seqLen, channels]
        Tensor outLinear = new Tensor(shape, DType.F32);
        LinearForward(outLinear, gated, _outLinearWeight!, _outLinearBias!, batch, seqLen, innerDim, _channels);
        gated.Dispose();

        // Residual
        Tensor output = new Tensor(shape, DType.F32);
        backend.Add(output, hidden, outLinear);
        outLinear.Dispose();

        return output;
    }

    /// <summary>GEGLU: splits input in half along last dim, applies GELU gate.</summary>
    private static void ApplyGeGlu(Tensor output, Tensor input, int batch, int seqLen, int innerDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int inOffset = (b * seqLen + s) * (innerDim * 2);
                int outOffset = (b * seqLen + s) * innerDim;
                for (int d = 0; d < innerDim; d++)
                {
                    float x = inPtr[inOffset + d];
                    float gate = inPtr[inOffset + innerDim + d];
                    // GELU(gate) * x
                    float geluGate = gate * 0.5f * (1.0f + MathF.Tanh(0.7978845608f * (gate + 0.044715f * gate * gate * gate)));
                    outPtr[outOffset + d] = x * geluGate;
                }
            }
        }
    }

    private static void LinearForward(Tensor output, Tensor input, Tensor weight, Tensor bias, int batch, int seqLen, int inDim, int outDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* wPtr = (float*)weight.DataPointer;
        float* bPtr = (float*)bias.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int inOffset = (b * seqLen + s) * inDim;
                int outOffset = (b * seqLen + s) * outDim;
                for (int o = 0; o < outDim; o++)
                {
                    float sum = bPtr[o];
                    int wOffset = o * inDim;
                    for (int i = 0; i < inDim; i++)
                    {
                        sum += inPtr[inOffset + i] * wPtr[wOffset + i];
                    }
                    outPtr[outOffset + o] = sum;
                }
            }
        }
    }
}

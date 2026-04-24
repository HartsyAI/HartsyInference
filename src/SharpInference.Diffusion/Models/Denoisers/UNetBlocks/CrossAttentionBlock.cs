using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.UNetBlocks;

/// <summary>Spatial transformer block for UNet cross-attention. Wraps N BasicTransformerBlocks (each: SelfAttn→CrossAttn→FFN). SD1.5 uses 1 block, SDXL uses [1,2,10] per level.</summary>
public sealed unsafe class CrossAttentionBlock
{
    private readonly int _channels;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _crossAttentionDim;
    private readonly int _numTransformerBlocks;

    // Input projection: GroupNorm + linear proj_in (or conv 1x1)
    private Tensor? _normWeight;
    private Tensor? _normBias;
    private Tensor? _projInWeight;
    private Tensor? _projInBias;

    // Multiple transformer blocks (SDXL can have up to 10 per spatial transformer)
    private readonly TransformerSubBlock[] _selfAttns;
    private readonly TransformerSubBlock[] _crossAttns;
    private readonly FeedForwardBlock[] _ffns;

    // Output projection
    private Tensor? _projOutWeight;
    private Tensor? _projOutBias;

    /// <summary>Creates a cross-attention block with N transformer layers.</summary>
    /// <param name="channels">Number of input/output channels.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="crossAttentionDim">Dimension of the cross-attention context (text encoder hidden size).</param>
    /// <param name="numTransformerBlocks">Number of BasicTransformerBlocks. SD1.5=1, SDXL=[1,2,10] per level.</param>
    /// <param name="normGroups">Number of groups for the input GroupNorm.</param>
    /// <param name="normEps">GroupNorm epsilon.</param>
    public CrossAttentionBlock(int channels, int numHeads, int crossAttentionDim, int numTransformerBlocks = 1, int normGroups = 32, float normEps = 1e-5f)
    {
        _channels = channels;
        _numHeads = numHeads;
        _headDim = channels / numHeads;
        _crossAttentionDim = crossAttentionDim;
        _numTransformerBlocks = numTransformerBlocks;

        _selfAttns = new TransformerSubBlock[numTransformerBlocks];
        _crossAttns = new TransformerSubBlock[numTransformerBlocks];
        _ffns = new FeedForwardBlock[numTransformerBlocks];

        for (int i = 0; i < numTransformerBlocks; i++)
        {
            _selfAttns[i] = new TransformerSubBlock(channels, numHeads, channels);
            _crossAttns[i] = new TransformerSubBlock(channels, numHeads, crossAttentionDim);
            _ffns[i] = new FeedForwardBlock(channels);
        }
    }

    /// <summary>Loads weights from named tensors.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _normWeight = weights[$"{prefix}.norm.weight"];
        _normBias = weights[$"{prefix}.norm.bias"];
        _projInWeight = weights[$"{prefix}.proj_in.weight"];
        _projInBias = weights[$"{prefix}.proj_in.bias"];

        for (int i = 0; i < _numTransformerBlocks; i++)
        {
            _selfAttns[i].LoadWeights(weights, $"{prefix}.transformer_blocks.{i}.attn1", $"{prefix}.transformer_blocks.{i}.norm1");
            _crossAttns[i].LoadWeights(weights, $"{prefix}.transformer_blocks.{i}.attn2", $"{prefix}.transformer_blocks.{i}.norm2");
            _ffns[i].LoadWeights(weights, $"{prefix}.transformer_blocks.{i}.ff", $"{prefix}.transformer_blocks.{i}.norm3");
        }

        _projOutWeight = weights[$"{prefix}.proj_out.weight"];
        _projOutBias = weights[$"{prefix}.proj_out.bias"];
    }

    /// <summary>Enumerates all weight tensors held by this block and its sub-blocks.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_normWeight is not null) yield return _normWeight;
        if (_normBias is not null) yield return _normBias;
        if (_projInWeight is not null) yield return _projInWeight;
        if (_projInBias is not null) yield return _projInBias;
        for (int i = 0; i < _numTransformerBlocks; i++)
        {
            foreach (Tensor w in _selfAttns[i].EnumerateWeights()) yield return w;
            foreach (Tensor w in _crossAttns[i].EnumerateWeights()) yield return w;
            foreach (Tensor w in _ffns[i].EnumerateWeights()) yield return w;
        }
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
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
        // When SDXL (useLinearProjection=true), weights are [C, C] (linear)
        // When SD1.5, weights are [C, C, 1, 1] (1x1 conv) but math is identical for proj_in/proj_out
        Tensor projected = new Tensor(seqShape, DType.F32);
        backend.Linear(projected, hidden, _projInWeight!, _projInBias!);
        hidden.Dispose();
        hidden = projected;

        // 3. Run N transformer blocks (SD1.5: 1, SDXL level 2: 10)
        for (int i = 0; i < _numTransformerBlocks; i++)
        {
            // Self-attention (context = self)
            Tensor selfOut = _selfAttns[i].Forward(backend, hidden, hidden);
            hidden.Dispose();
            hidden = selfOut;

            // Cross-attention (context = text embeddings)
            Tensor crossOut = _crossAttns[i].Forward(backend, hidden, context);
            hidden.Dispose();
            hidden = crossOut;

            // Feed-forward
            Tensor ffnOut = _ffns[i].Forward(backend, hidden);
            hidden.Dispose();
            hidden = ffnOut;
        }

        // 4. proj_out: [B, spatial, C] → [B, spatial, C]
        Tensor projOut = new Tensor(seqShape, DType.F32);
        backend.Linear(projOut, hidden, _projOutWeight!, _projOutBias!);
        hidden.Dispose();

        // 5. Reshape [B, H*W, C] → [B, C, H, W] and add residual
        Tensor output = new Tensor(spatialShape, DType.F32);
        ReshapeSequenceToSpatial(output, projOut, batch, channels, spatial);
        projOut.Dispose();

        // Add residual from input
        Tensor result = new Tensor(spatialShape, DType.F32);
        backend.Add(result, output, input);
        output.Dispose();

        return result;
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

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_normWeight is not null) yield return _normWeight;
        if (_normBias is not null) yield return _normBias;
        if (_toQWeight is not null) yield return _toQWeight;
        if (_toQBias is not null) yield return _toQBias;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toKBias is not null) yield return _toKBias;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toVBias is not null) yield return _toVBias;
        if (_toOutWeight is not null) yield return _toOutWeight;
        if (_toOutBias is not null) yield return _toOutBias;
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

        // Q from normed hidden; K/V from normed hidden (self-attn) or raw context (cross-attn)
        // Diffusers: self-attention passes norm_hidden_states for Q/K/V; cross-attention uses raw encoder_hidden_states for K/V
        Tensor kvSource = ReferenceEquals(hidden, context) ? normed : context;
        int kvSeqLen = ReferenceEquals(hidden, context) ? seqLen : ctxLen;
        int kvDim = ReferenceEquals(hidden, context) ? _channels : _contextDim;

        Tensor query = new Tensor(hidShape, DType.F32);
        backend.Linear(query, normed, _toQWeight!, _toQBias);

        TensorShape kvOutShape = new TensorShape(batch, kvSeqLen, _channels);
        Tensor key = new Tensor(kvOutShape, DType.F32);
        backend.Linear(key, kvSource, _toKWeight!, _toKBias);

        Tensor value = new Tensor(kvOutShape, DType.F32);
        backend.Linear(value, kvSource, _toVWeight!, _toVBias);
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
        Tensor projected = new Tensor(hidShape, DType.F32);
        backend.Linear(projected, merged, _toOutWeight!, _toOutBias);
        merged.Dispose();

        // Residual
        Tensor output = new Tensor(hidShape, DType.F32);
        backend.Add(output, hidden, projected);
        projected.Dispose();

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

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_normWeight is not null) yield return _normWeight;
        if (_normBias is not null) yield return _normBias;
        if (_geGluProjWeight is not null) yield return _geGluProjWeight;
        if (_geGluProjBias is not null) yield return _geGluProjBias;
        if (_outLinearWeight is not null) yield return _outLinearWeight;
        if (_outLinearBias is not null) yield return _outLinearBias;
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
        backend.Linear(geGluOut, normed, _geGluProjWeight!, _geGluProjBias!);
        normed.Dispose();

        // Split and apply GELU gate: output = x[:innerDim] * GELU(x[innerDim:])
        TensorShape innerShape = new TensorShape(batch, seqLen, innerDim);
        Tensor gated = new Tensor(innerShape, DType.F32);
        ApplyGeGlu(gated, geGluOut, batch, seqLen, innerDim);
        geGluOut.Dispose();

        // Output linear: [B, seqLen, innerDim] → [B, seqLen, channels]
        Tensor outLinear = new Tensor(shape, DType.F32);
        backend.Linear(outLinear, gated, _outLinearWeight!, _outLinearBias!);
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

}

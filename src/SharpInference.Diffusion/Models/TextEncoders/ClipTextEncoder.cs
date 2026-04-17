using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.TextEncoders;

/// <summary>CLIP text encoder (transformer). Takes token IDs and produces hidden states for conditioning diffusion models. Matches HuggingFace CLIPTextModel output.</summary>
public sealed unsafe class ClipTextEncoder
{
    private readonly ClipTextEncoderConfig _config;

    // Embeddings
    private Tensor? _tokenEmbeddingWeight;    // [vocabSize, hiddenSize]
    private Tensor? _positionEmbeddingWeight; // [maxPos, hiddenSize]

    // Transformer layers
    private readonly ClipTransformerLayer[] _layers;

    // Final layer norm
    private Tensor? _finalLayerNormWeight;
    private Tensor? _finalLayerNormBias;

    /// <summary>The configuration this encoder was built with.</summary>
    public ClipTextEncoderConfig Config => _config;

    /// <summary>Creates a CLIP text encoder with the specified configuration.</summary>
    public ClipTextEncoder(ClipTextEncoderConfig config)
    {
        _config = config;
        _layers = new ClipTransformerLayer[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            _layers[i] = new ClipTransformerLayer(config.HiddenSize, config.NumHeads, config.IntermediateSize, config.LayerNormEps, config.UseQuickGelu);
        }
    }

    /// <summary>Loads weights from a dictionary of named tensors. Keys should match diffusers naming (e.g., "text_model.encoder.layers.0.self_attn.q_proj.weight").</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "text_model")
    {
        _tokenEmbeddingWeight = weights[$"{prefix}.embeddings.token_embedding.weight"];
        _positionEmbeddingWeight = weights[$"{prefix}.embeddings.position_embedding.weight"];

        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i].LoadWeights(weights, $"{prefix}.encoder.layers.{i}");
        }

        _finalLayerNormWeight = weights[$"{prefix}.final_layer_norm.weight"];
        _finalLayerNormBias = weights[$"{prefix}.final_layer_norm.bias"];
    }

    /// <summary>Encodes token IDs [B, seqLen] into hidden states [B, seqLen, hiddenSize]. Returns the last hidden state (before final layer norm) for SD1.5 conditioning.</summary>
    public Tensor Encode(IBackend backend, ReadOnlySpan<int[]> batchTokenIds)
    {
        int batch = batchTokenIds.Length;
        int seqLen = batchTokenIds[0].Length;
        int hiddenSize = _config.HiddenSize;

        // 1. Token embedding lookup + position embedding
        TensorShape hiddenShape = new TensorShape(batch, seqLen, hiddenSize);
        Tensor hidden = new Tensor(hiddenShape, DType.F32);
        EmbedTokens(hidden, batchTokenIds, batch, seqLen, hiddenSize);

        // 2. Build causal attention mask [seqLen, seqLen]
        // CLIP uses causal masking: mask[i,j] = -inf if j > i (can't attend to future tokens)
        Tensor causalMask = BuildCausalMask(seqLen);

        // 3. Run through transformer layers
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor layerOut = _layers[i].Forward(backend, hidden, causalMask);
            hidden.Dispose();
            hidden = layerOut;
        }

        causalMask.Dispose();

        // CLIPTextModel applies final_layer_norm to encoder output before returning last_hidden_state.
        // This matches HuggingFace CLIPTextTransformer.forward() behavior.
        Tensor normed = new Tensor(hiddenShape, DType.F32);
        backend.LayerNorm(normed, hidden, _finalLayerNormWeight!, _finalLayerNormBias!, _config.LayerNormEps);
        hidden.Dispose();

        return normed;
    }

    /// <summary>Token embedding lookup + position embedding addition. Writes directly into the output tensor.</summary>
    private void EmbedTokens(Tensor output, ReadOnlySpan<int[]> batchTokenIds, int batch, int seqLen, int hiddenSize)
    {
        float* outPtr = (float*)output.DataPointer;
        float* tokenPtr = (float*)_tokenEmbeddingWeight!.DataPointer;
        float* posPtr = (float*)_positionEmbeddingWeight!.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int[] tokenIds = batchTokenIds[b];
            for (int s = 0; s < seqLen; s++)
            {
                int tokenId = tokenIds[s];
                int outOffset = (b * seqLen + s) * hiddenSize;
                int tokenOffset = tokenId * hiddenSize;
                int posOffset = s * hiddenSize;

                for (int h = 0; h < hiddenSize; h++)
                {
                    outPtr[outOffset + h] = tokenPtr[tokenOffset + h] + posPtr[posOffset + h];
                }
            }
        }
    }

    /// <summary>Builds a causal attention mask [seqLen, seqLen] where mask[i,j] = 0 if j &lt;= i, -inf otherwise.</summary>
    private static Tensor BuildCausalMask(int seqLen)
    {
        TensorShape maskShape = new TensorShape(seqLen, seqLen);
        Tensor mask = new Tensor(maskShape, DType.F32);
        float* maskPtr = (float*)mask.DataPointer;

        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j < seqLen; j++)
            {
                maskPtr[i * seqLen + j] = j <= i ? 0.0f : float.NegativeInfinity;
            }
        }

        return mask;
    }
}

/// <summary>Single CLIP transformer layer: LayerNorm → Self-Attention → Residual → LayerNorm → MLP → Residual.</summary>
internal sealed unsafe class ClipTransformerLayer
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _intermediateSize;
    private readonly float _layerNormEps;
    private readonly bool _useQuickGelu;

    // Layer norm 1 (before self-attention)
    private Tensor? _layerNorm1Weight;
    private Tensor? _layerNorm1Bias;

    // Self-attention projections
    private Tensor? _qProjWeight;
    private Tensor? _qProjBias;
    private Tensor? _kProjWeight;
    private Tensor? _kProjBias;
    private Tensor? _vProjWeight;
    private Tensor? _vProjBias;
    private Tensor? _outProjWeight;
    private Tensor? _outProjBias;

    // Layer norm 2 (before MLP)
    private Tensor? _layerNorm2Weight;
    private Tensor? _layerNorm2Bias;

    // MLP
    private Tensor? _mlpFc1Weight;
    private Tensor? _mlpFc1Bias;
    private Tensor? _mlpFc2Weight;
    private Tensor? _mlpFc2Bias;

    public ClipTransformerLayer(int hiddenSize, int numHeads, int intermediateSize, float layerNormEps, bool useQuickGelu)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = hiddenSize / numHeads;
        _intermediateSize = intermediateSize;
        _layerNormEps = layerNormEps;
        _useQuickGelu = useQuickGelu;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _layerNorm1Weight = weights[$"{prefix}.layer_norm1.weight"];
        _layerNorm1Bias = weights[$"{prefix}.layer_norm1.bias"];
        _qProjWeight = weights[$"{prefix}.self_attn.q_proj.weight"];
        _qProjBias = weights[$"{prefix}.self_attn.q_proj.bias"];
        _kProjWeight = weights[$"{prefix}.self_attn.k_proj.weight"];
        _kProjBias = weights[$"{prefix}.self_attn.k_proj.bias"];
        _vProjWeight = weights[$"{prefix}.self_attn.v_proj.weight"];
        _vProjBias = weights[$"{prefix}.self_attn.v_proj.bias"];
        _outProjWeight = weights[$"{prefix}.self_attn.out_proj.weight"];
        _outProjBias = weights[$"{prefix}.self_attn.out_proj.bias"];
        _layerNorm2Weight = weights[$"{prefix}.layer_norm2.weight"];
        _layerNorm2Bias = weights[$"{prefix}.layer_norm2.bias"];
        _mlpFc1Weight = weights[$"{prefix}.mlp.fc1.weight"];
        _mlpFc1Bias = weights[$"{prefix}.mlp.fc1.bias"];
        _mlpFc2Weight = weights[$"{prefix}.mlp.fc2.weight"];
        _mlpFc2Bias = weights[$"{prefix}.mlp.fc2.bias"];
    }

    /// <summary>Forward pass: hidden [B, seqLen, hiddenSize] → output [B, seqLen, hiddenSize].</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor causalMask)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];

        // Pre-norm 1 → Self-Attention → Residual
        TensorShape shape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor normed1 = new Tensor(shape, DType.F32);
        backend.LayerNorm(normed1, hidden, _layerNorm1Weight!, _layerNorm1Bias!, _layerNormEps);

        Tensor attnOut = MultiHeadSelfAttention(backend, normed1, causalMask, batch, seqLen);
        normed1.Dispose();

        Tensor residual1 = new Tensor(shape, DType.F32);
        backend.Add(residual1, hidden, attnOut);
        attnOut.Dispose();

        // Pre-norm 2 → MLP → Residual
        Tensor normed2 = new Tensor(shape, DType.F32);
        backend.LayerNorm(normed2, residual1, _layerNorm2Weight!, _layerNorm2Bias!, _layerNormEps);

        Tensor mlpOut = MlpForward(backend, normed2, batch, seqLen);
        normed2.Dispose();

        Tensor residual2 = new Tensor(shape, DType.F32);
        backend.Add(residual2, residual1, mlpOut);
        residual1.Dispose();
        mlpOut.Dispose();

        return residual2;
    }

    /// <summary>Multi-head self-attention with causal mask.</summary>
    private Tensor MultiHeadSelfAttention(IBackend backend, Tensor input, Tensor causalMask, int batch, int seqLen)
    {
        TensorShape seqShape = new TensorShape(batch, seqLen, _hiddenSize);

        // Project Q, K, V: [B, seqLen, hiddenSize] @ [hiddenSize, hiddenSize]^T
        Tensor query = ProjectLinear(backend, input, _qProjWeight!, _qProjBias!, batch, seqLen, _hiddenSize, _hiddenSize);
        Tensor key = ProjectLinear(backend, input, _kProjWeight!, _kProjBias!, batch, seqLen, _hiddenSize, _hiddenSize);
        Tensor value = ProjectLinear(backend, input, _vProjWeight!, _vProjBias!, batch, seqLen, _hiddenSize, _hiddenSize);

        // Reshape to multi-head 4D: [B, seqLen, numHeads*headDim] → [B, numHeads, seqLen, headDim]
        // AttentionKernels expects [B, H, S, D]
        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        Tensor queryMh = new Tensor(mhShape, DType.F32);
        Tensor keyMh = new Tensor(mhShape, DType.F32);
        Tensor valueMh = new Tensor(mhShape, DType.F32);

        ReshapeToMultiHead4D(queryMh, query, batch, seqLen, _numHeads, _headDim);
        ReshapeToMultiHead4D(keyMh, key, batch, seqLen, _numHeads, _headDim);
        ReshapeToMultiHead4D(valueMh, value, batch, seqLen, _numHeads, _headDim);
        query.Dispose();
        key.Dispose();
        value.Dispose();

        // Expand causal mask [seqLen, seqLen] → [1, 1, seqLen, seqLen] for broadcasting
        TensorShape maskShape4D = new TensorShape(1, 1, seqLen, seqLen);
        Tensor causalMask4D = new Tensor(maskShape4D, DType.F32);
        Buffer.MemoryCopy((void*)causalMask.DataPointer, (void*)causalMask4D.DataPointer,
            seqLen * seqLen * sizeof(float), seqLen * seqLen * sizeof(float));

        // Scaled dot-product attention with causal mask
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, queryMh, keyMh, valueMh, causalMask4D, scale);
        queryMh.Dispose();
        keyMh.Dispose();
        valueMh.Dispose();
        causalMask4D.Dispose();

        // Reshape back: [B, numHeads, seqLen, headDim] → [B, seqLen, hiddenSize]
        Tensor merged = new Tensor(seqShape, DType.F32);
        ReshapeFromMultiHead4D(merged, attnOut, batch, seqLen, _numHeads, _headDim);
        attnOut.Dispose();

        // Output projection
        Tensor projected = ProjectLinear(backend, merged, _outProjWeight!, _outProjBias!, batch, seqLen, _hiddenSize, _hiddenSize);
        merged.Dispose();

        return projected;
    }

    /// <summary>MLP: FC1 → QuickGELU/GELU → FC2.</summary>
    private Tensor MlpForward(IBackend backend, Tensor input, int batch, int seqLen)
    {
        // FC1: [B, seqLen, hiddenSize] → [B, seqLen, intermediateSize]
        Tensor fc1Out = ProjectLinear(backend, input, _mlpFc1Weight!, _mlpFc1Bias!, batch, seqLen, _hiddenSize, _intermediateSize);

        // Activation
        TensorShape fc1Shape = new TensorShape(batch, seqLen, _intermediateSize);
        Tensor activated = new Tensor(fc1Shape, DType.F32);

        if (_useQuickGelu)
        {
            QuickGelu(activated, fc1Out);
        }
        else
        {
            backend.Gelu(activated, fc1Out);
        }
        fc1Out.Dispose();

        // FC2: [B, seqLen, intermediateSize] → [B, seqLen, hiddenSize]
        Tensor fc2Out = ProjectLinear(backend, activated, _mlpFc2Weight!, _mlpFc2Bias!, batch, seqLen, _intermediateSize, _hiddenSize);
        activated.Dispose();

        return fc2Out;
    }

    /// <summary>Quick GELU: x * sigmoid(1.702 * x). Used by OpenAI CLIP.</summary>
    private static void QuickGelu(Tensor output, Tensor input)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)input.ElementCount;

        for (int i = 0; i < count; i++)
        {
            float x = inPtr[i];
            outPtr[i] = x * (1.0f / (1.0f + MathF.Exp(-1.702f * x)));
        }
    }

    /// <summary>Linear projection: output = input @ weight^T + bias. input [B, seqLen, inDim], weight [outDim, inDim].</summary>
    private static Tensor ProjectLinear(IBackend backend, Tensor input, Tensor weight, Tensor bias, int batch, int seqLen, int inDim, int outDim)
    {
        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor output = new Tensor(outShape, DType.F32);

        // Transpose weight [outDim, inDim] → [inDim, outDim]
        TensorShape weightTShape = new TensorShape(inDim, outDim);
        Tensor weightT = new Tensor(weightTShape, DType.F32);
        TransposeMatrix(weight, weightT, outDim, inDim);

        // Batched matmul: [B, seqLen, inDim] @ [inDim, outDim] = [B, seqLen, outDim]
        backend.BatchedMatMul(output, input, weightT);
        weightT.Dispose();

        // Add bias broadcast
        AddBiasBroadcast(output, bias, batch, seqLen, outDim);

        return output;
    }

    /// <summary>Transpose 2D matrix [rows, cols] → [cols, rows].</summary>
    private static void TransposeMatrix(Tensor src, Tensor dst, int rows, int cols)
    {
        float* srcPtr = (float*)src.DataPointer;
        float* dstPtr = (float*)dst.DataPointer;

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                dstPtr[c * rows + r] = srcPtr[r * cols + c];
            }
        }
    }

    /// <summary>Adds bias [outDim] to output [B, seqLen, outDim] in-place.</summary>
    private static void AddBiasBroadcast(Tensor output, Tensor bias, int batch, int seqLen, int outDim)
    {
        float* outPtr = (float*)output.DataPointer;
        float* biasPtr = (float*)bias.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int offset = (b * seqLen + s) * outDim;
                for (int d = 0; d < outDim; d++)
                {
                    outPtr[offset + d] += biasPtr[d];
                }
            }
        }
    }

    /// <summary>Reshapes [B, seqLen, numHeads*headDim] → [B, numHeads, seqLen, headDim] via copy.</summary>
    private static void ReshapeToMultiHead4D(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
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
                    // Output layout: [B, H, S, D]
                    int outOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        outPtr[outOffset + d] = inPtr[inOffset + d];
                    }
                }
            }
        }
    }

    /// <summary>Reshapes [B, numHeads, seqLen, headDim] → [B, seqLen, numHeads*headDim] via copy.</summary>
    private static void ReshapeFromMultiHead4D(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    // Input layout: [B, H, S, D]
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

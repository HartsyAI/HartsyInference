using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Tokenizers;

namespace HartsyInference.Diffusion.Models.TextEncoders;

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

    /// <summary>Loads weights from a dictionary of named tensors. Keys should match diffusers naming (e.g., "text_model.encoder.layers.0.self_attn.q_proj.weight"). Embedding and projection weights are auto-cast to F32 since they are accessed via float* DataPointer.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "text_model")
    {
        _tokenEmbeddingWeight = EnsureF32(weights[$"{prefix}.embeddings.token_embedding.weight"]);
        _positionEmbeddingWeight = EnsureF32(weights[$"{prefix}.embeddings.position_embedding.weight"]);

        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i].LoadWeights(weights, $"{prefix}.encoder.layers.{i}");
        }

        _finalLayerNormWeight = EnsureF32(weights[$"{prefix}.final_layer_norm.weight"]);
        _finalLayerNormBias = EnsureF32(weights[$"{prefix}.final_layer_norm.bias"]);

        // text_projection exists for CLIP-G (SDXL text_encoder_2) — used for pooled output
        if (weights.TryGetValue("text_projection.weight", out Tensor? textProj))
            _textProjectionWeight = EnsureF32(textProj);
    }

    private static Tensor EnsureF32(Tensor tensor) =>
        tensor.DType != DType.F32 ? tensor.CastTo(DType.F32) : tensor;

    // Text projection weight for pooled output (CLIP-G only, null for CLIP-L)
    private Tensor? _textProjectionWeight;

    /// <summary>Yields every weight tensor — token + position embeddings, every transformer layer's
    /// parameters, the final LayerNorm, and the optional <c>text_projection</c>. Used by
    /// <c>backend.PreloadWeights(encoder.EnumerateWeights())</c> for GPU weight cache priming, and
    /// by sharing pipelines that need to free CLIP weights when transitioning stages.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_tokenEmbeddingWeight is not null) yield return _tokenEmbeddingWeight;
        if (_positionEmbeddingWeight is not null) yield return _positionEmbeddingWeight;
        for (int i = 0; i < _layers.Length; i++)
        {
            foreach (Tensor w in _layers[i].EnumerateWeights()) yield return w;
        }
        if (_finalLayerNormWeight is not null) yield return _finalLayerNormWeight;
        if (_finalLayerNormBias is not null) yield return _finalLayerNormBias;
        if (_textProjectionWeight is not null) yield return _textProjectionWeight;
    }

    /// <summary>Encodes token IDs [B, seqLen] into hidden states [B, seqLen, hiddenSize]. Returns the hidden state with final layer norm applied. Used by SD1.5.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="batchTokenIds">Token IDs [B, seqLen].</param>
    /// <param name="layersFromEnd">"CLIP skip": 1 (default) runs all transformer layers (standard last_hidden_state); 2 stops one layer early (penultimate, the "clip skip 2" many SD1.5 anime checkpoints were trained with); etc. Final layer norm is applied to whichever layer's output is taken, matching ComfyUI's <c>CLIPSetLastLayer</c> semantics. Clamped to [1, numLayers].</param>
    public Tensor Encode(IBackend backend, ReadOnlySpan<int[]> batchTokenIds, int layersFromEnd = 1,
        IReadOnlyDictionary<int, Tensor>? inlineEmbeddings = null)
    {
        int batch = batchTokenIds.Length;
        int seqLen = batchTokenIds[0].Length;
        int hiddenSize = _config.HiddenSize;

        // 1. Token embedding lookup + position embedding
        TensorShape hiddenShape = new TensorShape(batch, seqLen, hiddenSize);
        Tensor hidden = new Tensor(hiddenShape, DType.F32);
        EmbedTokens(hidden, batchTokenIds, batch, seqLen, hiddenSize, inlineEmbeddings);

        // 2. Build causal attention mask [seqLen, seqLen]
        Tensor causalMask = BuildCausalMask(seqLen);

        // 3. Run through transformer layers, optionally stopping early (CLIP skip)
        int layersToRun = _layers.Length - Math.Clamp(layersFromEnd, 1, _layers.Length) + 1;
        for (int i = 0; i < layersToRun; i++)
        {
            Tensor layerOut = _layers[i].Forward(backend, hidden, causalMask);
            hidden.Dispose();
            hidden = layerOut;
        }

        causalMask.Dispose();

        // CLIPTextModel applies final_layer_norm to encoder output before returning last_hidden_state.
        Tensor normed = new Tensor(hiddenShape, DType.F32);
        backend.LayerNorm(normed, hidden, _finalLayerNormWeight!, _finalLayerNormBias!, _config.LayerNormEps);
        hidden.Dispose();

        return normed;
    }

    /// <summary>Encodes token IDs and returns a hidden state taken <paramref name="layersFromEnd"/> layers from the end (before final layer norm). Used by SDXL/SD3 for CLIP-L and CLIP-G. Also returns pooled output [B, projectionDim] from the EOS token if text_projection weight exists.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="batchTokenIds">Token IDs [B, seqLen].</param>
    /// <param name="eosTokenPositions">Position of the EOS token in each batch element (for pooled output extraction).</param>
    /// <param name="layersFromEnd">"CLIP skip": 2 (default) returns the penultimate layer output, the layer SDXL/SD3 are specified against; 1 returns the final layer output; higher values stop earlier. Clamped to [1, numLayers]. The pooled output is always taken from the full last_hidden_state regardless of this value, matching diffusers/ComfyUI.</param>
    /// <returns>Tuple of (hiddenStates [B, seqLen, hiddenSize], pooledOutput [B, projectionDim] or null).</returns>
    public (Tensor hiddenStates, Tensor? pooledOutput) EncodePenultimate(IBackend backend, ReadOnlySpan<int[]> batchTokenIds, ReadOnlySpan<int> eosTokenPositions,
        int layersFromEnd = 2,
        IReadOnlyDictionary<int, Tensor>? inlineEmbeddings = null)
    {
        int batch = batchTokenIds.Length;
        int seqLen = batchTokenIds[0].Length;
        int hiddenSize = _config.HiddenSize;

        // 1. Token embedding lookup + position embedding
        TensorShape hiddenShape = new TensorShape(batch, seqLen, hiddenSize);
        Tensor hidden = new Tensor(hiddenShape, DType.F32);
        EmbedTokens(hidden, batchTokenIds, batch, seqLen, hiddenSize, inlineEmbeddings);

        // 2. Build causal attention mask
        Tensor causalMask = BuildCausalMask(seqLen);

        // 3. Run every transformer layer, but snapshot the hidden state at the CLIP-skip layer
        //    (layersFromEnd=2 → penultimate). Pooled output always comes from the full last layer,
        //    so we never stop early — we just take a copy at the requested depth.
        int numLayers = _layers.Length;
        int layersToCapture = numLayers - Math.Clamp(layersFromEnd, 1, numLayers) + 1;
        Tensor? captured = null;
        for (int i = 0; i < numLayers; i++)
        {
            Tensor layerOut = _layers[i].Forward(backend, hidden, causalMask);
            hidden.Dispose();
            hidden = layerOut;

            // Snapshot before subsequent layers mutate `hidden` (before final LN).
            if (i + 1 == layersToCapture)
            {
                captured = hidden.To(hidden.Device);
            }
        }
        causalMask.Dispose();

        // captured is non-null because layersToCapture ∈ [1, numLayers].
        Tensor skipped = captured!;

        // Apply final layer norm to the full last-layer output to get last_hidden_state for pooling.
        Tensor normedFull = new Tensor(hiddenShape, DType.F32);
        backend.LayerNorm(normedFull, hidden, _finalLayerNormWeight!, _finalLayerNormBias!, _config.LayerNormEps);
        hidden.Dispose();

        // Extract pooled output from EOS token position (raw EOS hidden when there's no text_projection,
        // projected when there is — handled inside ExtractPooledOutput). Always non-null now so dual-pooled
        // models (HiDream/SD3 concat CLIP-L + CLIP-G pooled) don't NRE on CLIP-L's missing projection.
        Tensor pooledOutput = ExtractPooledOutput(normedFull, eosTokenPositions, batch, seqLen, hiddenSize);

        normedFull.Dispose();

        return (skipped, pooledOutput);
    }

    /// <summary>Encodes ComfyUI-weighted token chunks into hidden states <c>[1, seqLen*numChunks, hiddenSize]</c> with final layer norm. Each chunk is encoded independently, weighted via the empty-baseline formula (<see cref="EmphasisMath.ApplyComfy"/>), and concatenated along the sequence axis. Used by SD1.5.</summary>
    public Tensor EncodeWeighted(IBackend backend, IReadOnlyList<int[]> tokenIdChunks, IReadOnlyList<float[]> tokenWeightChunks, int layersFromEnd = 1)
    {
        ValidateWeightedChunks(tokenIdChunks, tokenWeightChunks);
        int hiddenSize = _config.HiddenSize;
        int numChunks = tokenIdChunks.Count;
        int seqLen = tokenIdChunks[0].Length;

        int[] emptyChunk = BuildEmptyChunk(seqLen);
        Tensor zEmpty = Encode(backend, new int[][] { emptyChunk }, layersFromEnd);
        ReadOnlySpan<float> emptySpan = zEmpty.AsReadOnlySpan<float>();

        Tensor result = new Tensor(new TensorShape(1, seqLen * numChunks, hiddenSize), DType.F32);
        Span<float> resultSpan = result.AsSpan<float>();
        for (int c = 0; c < numChunks; c++)
        {
            Tensor z = Encode(backend, new int[][] { tokenIdChunks[c] }, layersFromEnd);
            Span<float> zSpan = z.AsSpan<float>();
            EmphasisMath.ApplyComfy(zSpan, emptySpan, tokenWeightChunks[c], seqLen, hiddenSize);
            zSpan.CopyTo(resultSpan.Slice(c * seqLen * hiddenSize, seqLen * hiddenSize));
            z.Dispose();
        }
        zEmpty.Dispose();
        return result;
    }

    /// <summary>Penultimate-layer variant of <see cref="EncodeWeighted"/> for SDXL/SD3. Returns weighted penultimate hidden states <c>[1, seqLen*numChunks, hiddenSize]</c> plus an unweighted pooled output taken from the first chunk's EOS (null when there is no text_projection).</summary>
    public (Tensor hiddenStates, Tensor? pooledOutput) EncodeWeightedPenultimate(IBackend backend, IReadOnlyList<int[]> tokenIdChunks, IReadOnlyList<float[]> tokenWeightChunks, ReadOnlySpan<int> eosTokenPositions, int layersFromEnd = 2)
    {
        ValidateWeightedChunks(tokenIdChunks, tokenWeightChunks);
        int hiddenSize = _config.HiddenSize;
        int numChunks = tokenIdChunks.Count;
        int seqLen = tokenIdChunks[0].Length;

        int[] emptyChunk = BuildEmptyChunk(seqLen);
        (Tensor zEmpty, _) = EncodePenultimate(backend, new int[][] { emptyChunk }, stackalloc int[] { 1 }, layersFromEnd);
        ReadOnlySpan<float> emptySpan = zEmpty.AsReadOnlySpan<float>();

        Tensor result = new Tensor(new TensorShape(1, seqLen * numChunks, hiddenSize), DType.F32);
        Span<float> resultSpan = result.AsSpan<float>();
        Tensor? pooled = null;
        Span<int> eosSpan = stackalloc int[1];
        for (int c = 0; c < numChunks; c++)
        {
            eosSpan[0] = c < eosTokenPositions.Length ? eosTokenPositions[c] : 0;
            (Tensor z, Tensor? chunkPooled) = EncodePenultimate(backend, new int[][] { tokenIdChunks[c] }, eosSpan, layersFromEnd);
            Span<float> zSpan = z.AsSpan<float>();
            EmphasisMath.ApplyComfy(zSpan, emptySpan, tokenWeightChunks[c], seqLen, hiddenSize);
            zSpan.CopyTo(resultSpan.Slice(c * seqLen * hiddenSize, seqLen * hiddenSize));
            z.Dispose();
            if (c == 0)
            {
                pooled = chunkPooled;
            }
            else
            {
                chunkPooled?.Dispose();
            }
        }
        zEmpty.Dispose();
        return (result, pooled);
    }

    private static void ValidateWeightedChunks(IReadOnlyList<int[]> idChunks, IReadOnlyList<float[]> weightChunks)
    {
        if (idChunks.Count == 0)
        {
            throw new ArgumentException("At least one token chunk is required.", nameof(idChunks));
        }
        if (idChunks.Count != weightChunks.Count)
        {
            throw new ArgumentException($"id chunk count {idChunks.Count} must equal weight chunk count {weightChunks.Count}.");
        }
        int seqLen = idChunks[0].Length;
        for (int c = 0; c < idChunks.Count; c++)
        {
            if (idChunks[c].Length != seqLen || weightChunks[c].Length != seqLen)
            {
                throw new ArgumentException($"All chunks must have equal length {seqLen}.");
            }
        }
    }

    private static int[] BuildEmptyChunk(int seqLen)
    {
        int[] chunk = new int[seqLen];
        chunk[0] = ClipTokenizer.StartOfTextId;
        for (int i = 1; i < seqLen; i++)
        {
            chunk[i] = ClipTokenizer.EndOfTextId;
        }
        return chunk;
    }

    /// <summary>Extracts pooled output: takes the hidden state at the EOS token position and projects through text_projection.</summary>
    private Tensor ExtractPooledOutput(Tensor normedHidden, ReadOnlySpan<int> eosPositions, int batch, int seqLen, int hiddenSize)
    {
        int projDim = _config.ProjectionDim;

        // Extract EOS token hidden states [B, hiddenSize]
        TensorShape eosShape = new TensorShape(batch, hiddenSize);
        Tensor eosHidden = new Tensor(eosShape, DType.F32);
        float* srcPtr = (float*)normedHidden.DataPointer;
        float* dstPtr = (float*)eosHidden.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int eosPos = eosPositions[b];
            int srcOffset = (b * seqLen + eosPos) * hiddenSize;
            int dstOffset = b * hiddenSize;
            for (int d = 0; d < hiddenSize; d++)
            {
                dstPtr[dstOffset + d] = srcPtr[srcOffset + d];
            }
        }

        // No text_projection (CLIP-L in HiDream's quad-encoder, and SDXL where CLIP-L pooled is unused):
        // the pooled output IS the raw EOS hidden state [B, hiddenSize]. Only projected encoders (CLIP-G /
        // SD3 CLIP-L) run the matmul below. Safe for SD3/SDXL — they only consume pooled where a projection exists.
        if (_textProjectionWeight is null || projDim <= 0)
            return eosHidden;

        // Project through text_projection. The stored weight is `nn.Linear(hidden, proj).weight`,
        // shape `[proj, hidden]` (PyTorch's `out_features, in_features` convention), and forward is
        // `output = x @ weight.T` → `output[o] = Σ_i x[i] * weight[o, i] = Σ_i x[i] * w[o*hidden + i]`.
        // For non-symmetric square matrices (CLIP-L/G's 768/1280 text_projection) the wrong transpose
        // produces "noisy but bounded" output that fooled smoke tests but mangled SD3's pooled
        // conditioning — see PHASE_3_DEVIATIONS notes for SD3.5 patch-grid debugging.
        TensorShape pooledShape = new TensorShape(batch, projDim);
        Tensor pooled = new Tensor(pooledShape, DType.F32);
        float* ePtr = (float*)eosHidden.DataPointer;
        float* wPtr = (float*)_textProjectionWeight!.DataPointer;
        float* pPtr = (float*)pooled.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int inOffset = b * hiddenSize;
            for (int o = 0; o < projDim; o++)
            {
                float sum = 0f;
                int wRow = o * hiddenSize;
                for (int i = 0; i < hiddenSize; i++)
                {
                    sum += ePtr[inOffset + i] * wPtr[wRow + i];
                }
                pPtr[b * projDim + o] = sum;
            }
        }

        eosHidden.Dispose();
        return pooled;
    }

    /// <summary>Token embedding lookup + position embedding addition. Writes directly into the output tensor. When <paramref name="inlineEmbeddings"/> contains a token id (a textual-inversion placeholder), its <c>[hiddenSize]</c> vector replaces the token-embedding lookup before the position embedding is added.</summary>
    private void EmbedTokens(Tensor output, ReadOnlySpan<int[]> batchTokenIds, int batch, int seqLen, int hiddenSize,
        IReadOnlyDictionary<int, Tensor>? inlineEmbeddings = null)
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
                int posOffset = s * hiddenSize;

                if (inlineEmbeddings is not null && inlineEmbeddings.TryGetValue(tokenId, out Tensor? embedding))
                {
                    float* embPtr = (float*)embedding.DataPointer;
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        outPtr[outOffset + h] = embPtr[h] + posPtr[posOffset + h];
                    }
                    continue;
                }

                int tokenOffset = tokenId * hiddenSize;
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
        // All weights auto-cast to F32 — ProjectLinear/TransposeMatrix/AddBiasBroadcast use float* DataPointer
        _layerNorm1Weight = EnsureF32(weights[$"{prefix}.layer_norm1.weight"]);
        _layerNorm1Bias = EnsureF32(weights[$"{prefix}.layer_norm1.bias"]);
        _qProjWeight = EnsureF32(weights[$"{prefix}.self_attn.q_proj.weight"]);
        _qProjBias = EnsureF32(weights[$"{prefix}.self_attn.q_proj.bias"]);
        _kProjWeight = EnsureF32(weights[$"{prefix}.self_attn.k_proj.weight"]);
        _kProjBias = EnsureF32(weights[$"{prefix}.self_attn.k_proj.bias"]);
        _vProjWeight = EnsureF32(weights[$"{prefix}.self_attn.v_proj.weight"]);
        _vProjBias = EnsureF32(weights[$"{prefix}.self_attn.v_proj.bias"]);
        _outProjWeight = EnsureF32(weights[$"{prefix}.self_attn.out_proj.weight"]);
        _outProjBias = EnsureF32(weights[$"{prefix}.self_attn.out_proj.bias"]);
        _layerNorm2Weight = EnsureF32(weights[$"{prefix}.layer_norm2.weight"]);
        _layerNorm2Bias = EnsureF32(weights[$"{prefix}.layer_norm2.bias"]);
        _mlpFc1Weight = EnsureF32(weights[$"{prefix}.mlp.fc1.weight"]);
        _mlpFc1Bias = EnsureF32(weights[$"{prefix}.mlp.fc1.bias"]);
        _mlpFc2Weight = EnsureF32(weights[$"{prefix}.mlp.fc2.weight"]);
        _mlpFc2Bias = EnsureF32(weights[$"{prefix}.mlp.fc2.bias"]);
    }

    private static Tensor EnsureF32(Tensor tensor) =>
        tensor.DType != DType.F32 ? tensor.CastTo(DType.F32) : tensor;

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [
            _layerNorm1Weight, _layerNorm1Bias,
            _qProjWeight, _qProjBias, _kProjWeight, _kProjBias, _vProjWeight, _vProjBias,
            _outProjWeight, _outProjBias,
            _layerNorm2Weight, _layerNorm2Bias,
            _mlpFc1Weight, _mlpFc1Bias, _mlpFc2Weight, _mlpFc2Bias,
        ];
        foreach (Tensor? t in all) if (t is not null) yield return t;
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

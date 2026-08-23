using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>CLIP vision tower (image branch). Same transformer architecture as <see cref="ClipTextEncoder"/> but with a Conv2D patch embedding, a prepended learned CLS token, no causal mask, and an optional visual_projection that maps the CLS token down to <see cref="ClipVisionEncoderConfig.ProjectionDim"/> for the contrastive-aligned image embedding. Used as the image-prompt encoder for IP-Adapter (CLS-based for IP-Adapter standard, full-sequence penultimate for IP-Adapter Plus).
/// <para>Weight loader matches HuggingFace diffusers naming: <c>vision_model.embeddings.{patch_embedding,class_embedding,position_embedding}</c>, <c>vision_model.pre_layrnorm</c> (note the HF typo, kept for compat), per-layer <c>vision_model.encoder.layers.{i}.*</c>, <c>vision_model.post_layernorm.*</c>, and the top-level <c>visual_projection.weight</c>.</para></summary>
public sealed unsafe class ClipVisionEncoder
{
    private readonly ClipVisionEncoderConfig _config;
    private readonly int _numPatches;
    private readonly int _seqLen; // numPatches + 1 (CLS)

    private Tensor? _patchEmbeddingWeight;       // [hidden, channels, patch, patch]
    private Tensor? _classEmbedding;             // [hidden]
    private Tensor? _positionEmbeddingWeight;    // [numPatches+1, hidden]
    private Tensor? _preLayerNormWeight, _preLayerNormBias;
    private readonly ClipVisionTransformerLayer[] _layers;
    private Tensor? _postLayerNormWeight, _postLayerNormBias;
    private Tensor? _visualProjectionWeight;     // [projectionDim, hidden]

    /// <summary>The configuration this encoder was built with.</summary>
    public ClipVisionEncoderConfig Config => _config;

    /// <summary>Creates a CLIP vision encoder.</summary>
    public ClipVisionEncoder(ClipVisionEncoderConfig config)
    {
        _config = config;
        int patchGrid = config.ImageSize / config.PatchSize;
        _numPatches = patchGrid * patchGrid;
        _seqLen = _numPatches + 1;
        _layers = new ClipVisionTransformerLayer[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            _layers[i] = new ClipVisionTransformerLayer(config.HiddenSize, config.NumHeads, config.IntermediateSize, config.LayerNormEps, config.UseQuickGelu);
        }
    }

    /// <summary>Loads weights from the diffusers safetensors layout. The CLIP vision encoder lives under <c>vision_model.*</c> in CLIPVisionModel checkpoints; the optional projection sits at the top level as <c>visual_projection.weight</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "vision_model")
    {
        _patchEmbeddingWeight = TensorCasts.EnsureF32(weights[$"{prefix}.embeddings.patch_embedding.weight"]);
        _classEmbedding = TensorCasts.EnsureF32(weights[$"{prefix}.embeddings.class_embedding"]);
        _positionEmbeddingWeight = TensorCasts.EnsureF32(weights[$"{prefix}.embeddings.position_embedding.weight"]);

        // HF kept the original "layrnorm" typo from OpenAI's release. Some IPA-friendly
        // checkpoints fix it to "layernorm" — try both so we accept either spelling.
        if (weights.TryGetValue($"{prefix}.pre_layrnorm.weight", out Tensor? preLnW))
        {
            _preLayerNormWeight = TensorCasts.EnsureF32(preLnW);
            _preLayerNormBias = TensorCasts.EnsureF32(weights[$"{prefix}.pre_layrnorm.bias"]);
        }
        else
        {
            _preLayerNormWeight = TensorCasts.EnsureF32(weights[$"{prefix}.pre_layernorm.weight"]);
            _preLayerNormBias = TensorCasts.EnsureF32(weights[$"{prefix}.pre_layernorm.bias"]);
        }

        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i].LoadWeights(weights, $"{prefix}.encoder.layers.{i}");
        }

        _postLayerNormWeight = TensorCasts.EnsureF32(weights[$"{prefix}.post_layernorm.weight"]);
        _postLayerNormBias = TensorCasts.EnsureF32(weights[$"{prefix}.post_layernorm.bias"]);

        // visual_projection is optional — only IPA standard uses it (the projected CLS embed).
        // IPA Plus skips it entirely and uses penultimate hidden states pre-projection.
        if (weights.TryGetValue("visual_projection.weight", out Tensor? proj))
        {
            _visualProjectionWeight = TensorCasts.EnsureF32(proj);
        }
    }

    /// <summary>Yields all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_patchEmbeddingWeight is not null) yield return _patchEmbeddingWeight;
        if (_classEmbedding is not null) yield return _classEmbedding;
        if (_positionEmbeddingWeight is not null) yield return _positionEmbeddingWeight;
        if (_preLayerNormWeight is not null) yield return _preLayerNormWeight;
        if (_preLayerNormBias is not null) yield return _preLayerNormBias;
        for (int i = 0; i < _layers.Length; i++)
        {
            foreach (Tensor w in _layers[i].EnumerateWeights()) yield return w;
        }
        if (_postLayerNormWeight is not null) yield return _postLayerNormWeight;
        if (_postLayerNormBias is not null) yield return _postLayerNormBias;
        if (_visualProjectionWeight is not null) yield return _visualProjectionWeight;
    }

    /// <summary>Encodes a CLIP-normalized image into a projected CLS embedding <c>[B, projectionDim]</c>. This is the input IP-Adapter standard's image projection MLP consumes. Requires <see cref="ClipVisionEncoderConfig.ProjectionDim"/> &gt; 0 and a loaded <c>visual_projection</c> weight.</summary>
    public Tensor EncodeImageEmbeds(IBackend backend, Tensor pixelValues)
    {
        if (_visualProjectionWeight is null)
        {
            throw new InvalidOperationException(
                "EncodeImageEmbeds requires a loaded visual_projection weight. The checkpoint either lacks one or LoadWeights was called with a prefix that didn't include 'visual_projection'.");
        }
        Tensor lastHidden = ForwardTransformer(backend, pixelValues, layersToRun: _layers.Length, applyPostLayernorm: true);
        // CLS token is index 0 along the sequence axis. Slice it out.
        int batch = (int)lastHidden.Shape[0];
        int hidden = _config.HiddenSize;
        Tensor cls = SliceCls(lastHidden, batch, hidden);
        lastHidden.Dispose();
        // Project CLS to contrastive embedding space.
        Tensor projected = ProjectVisual(cls, batch);
        cls.Dispose();
        return projected;
    }

    /// <summary>Encodes a CLIP-normalized image into the penultimate layer's hidden states for ALL tokens (CLS + patches). Returns shape <c>[B, seqLen=numPatches+1, hiddenSize]</c>. This is the input IP-Adapter Plus's resampler consumes — pre-final-norm, pre-projection, all tokens.</summary>
    public Tensor EncodeHiddenStates(IBackend backend, Tensor pixelValues)
    {
        // diffusers' IPAdapterPlus uses image_encoder(pixel_values, output_hidden_states=True).hidden_states[-2]
        // which is the output of layer (numLayers - 1) = the penultimate layer post-residual,
        // before the post_layernorm and visual_projection.
        return ForwardTransformer(backend, pixelValues, layersToRun: _layers.Length - 1, applyPostLayernorm: false);
    }

    /// <summary>Runs the vision transformer and returns a deep copy of the hidden states <c>[B, seqLen, hidden]</c>
    /// captured immediately after each 1-indexed layer listed in <paramref name="afterLayers"/> (no post_layernorm).
    /// Used by CLIPSeg, whose decoder fuses activations from layers <c>[3, 6, 9]</c>. The returned tensors are owned
    /// by the caller. Order matches <paramref name="afterLayers"/>.</summary>
    public Tensor[] EncodeExtractLayers(IBackend backend, Tensor pixelValues, int[] afterLayers)
    {
        int batch = (int)pixelValues.Shape[0];
        int hidden = _config.HiddenSize;
        int patchGrid = _config.ImageSize / _config.PatchSize;

        TensorShape patchOutShape = new TensorShape(batch, hidden, patchGrid, patchGrid);
        Tensor patchOut = new Tensor(patchOutShape, DType.F32);
        backend.Conv2D(patchOut, pixelValues, _patchEmbeddingWeight!, null, _config.PatchSize, _config.PatchSize, 0, 0);
        Tensor flat = patchOut.Reshape(new TensorShape(batch, hidden, _numPatches));
        Tensor patchSeq = new Tensor(new TensorShape(batch, _numPatches, hidden), DType.F32);
        backend.Transpose2D(patchSeq, flat, hidden, _numPatches);
        patchOut.Dispose();

        TensorShape seqShape = new TensorShape(batch, _seqLen, hidden);
        Tensor embedded = new Tensor(seqShape, DType.F32);
        BuildEmbedded(embedded, patchSeq, batch, hidden);
        patchSeq.Dispose();

        Tensor h = new Tensor(seqShape, DType.F32);
        backend.LayerNorm(h, embedded, _preLayerNormWeight!, _preLayerNormBias!, _config.LayerNormEps);
        embedded.Dispose();

        Tensor[] captured = new Tensor[afterLayers.Length];
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, h);
            h.Dispose();
            h = next;
            int layerNum = i + 1; // output of the (i+1)-th layer, matching HF hidden_states indexing
            for (int e = 0; e < afterLayers.Length; e++)
            {
                if (afterLayers[e] == layerNum)
                {
                    Tensor copy = new Tensor(seqShape, DType.F32);
                    long bytes = seqShape.ElementCount * sizeof(float);
                    Buffer.MemoryCopy((void*)h.DataPointer, (void*)copy.DataPointer, bytes, bytes);
                    captured[e] = copy;
                }
            }
        }
        h.Dispose();
        return captured;
    }

    /// <summary>Shared transformer driver. Runs patch embed → CLS prepend → pos embed → pre_layernorm → first <paramref name="layersToRun"/> transformer layers → optional post_layernorm. Returns <c>[B, seqLen, hidden]</c> in F32.</summary>
    private Tensor ForwardTransformer(IBackend backend, Tensor pixelValues, int layersToRun, bool applyPostLayernorm)
    {
        if (pixelValues.Shape.Rank != 4 || pixelValues.Shape[1] != _config.NumChannels
            || pixelValues.Shape[2] != _config.ImageSize || pixelValues.Shape[3] != _config.ImageSize)
        {
            throw new ArgumentException(
                $"pixelValues must be [B, {_config.NumChannels}, {_config.ImageSize}, {_config.ImageSize}]; got {pixelValues.Shape}.", nameof(pixelValues));
        }
        int batch = (int)pixelValues.Shape[0];
        int hidden = _config.HiddenSize;
        int patchGrid = _config.ImageSize / _config.PatchSize;

        // 1. Patch embedding: Conv2D with kernel=stride=patch_size, no bias.
        //    Input  [B, 3, H, W]  →  output [B, hidden, gridH, gridW] in F32.
        TensorShape patchOutShape = new TensorShape(batch, hidden, patchGrid, patchGrid);
        Tensor patchOut = new Tensor(patchOutShape, DType.F32);
        backend.Conv2D(patchOut, pixelValues, _patchEmbeddingWeight!, null, _config.PatchSize, _config.PatchSize, 0, 0);

        // 2. Reshape [B, hidden, gridH, gridW] → [B, hidden, numPatches] → transpose → [B, numPatches, hidden].
        Tensor flat = patchOut.Reshape(new TensorShape(batch, hidden, _numPatches));
        Tensor patchSeq = new Tensor(new TensorShape(batch, _numPatches, hidden), DType.F32);
        backend.Transpose2D(patchSeq, flat, hidden, _numPatches);
        patchOut.Dispose();

        // 3. Prepend CLS token + add positional embedding.
        TensorShape seqShape = new TensorShape(batch, _seqLen, hidden);
        Tensor embedded = new Tensor(seqShape, DType.F32);
        BuildEmbedded(embedded, patchSeq, batch, hidden);
        patchSeq.Dispose();

        // 4. pre_layernorm.
        Tensor normed = new Tensor(seqShape, DType.F32);
        backend.LayerNorm(normed, embedded, _preLayerNormWeight!, _preLayerNormBias!, _config.LayerNormEps);
        embedded.Dispose();
        Tensor h = normed;

        // 5. Run transformer layers.
        for (int i = 0; i < layersToRun; i++)
        {
            Tensor next = _layers[i].Forward(backend, h);
            h.Dispose();
            h = next;
        }

        // 6. Optional post_layernorm. Skipped when caller wants penultimate hidden states (Plus path).
        if (applyPostLayernorm)
        {
            Tensor finalNormed = new Tensor(seqShape, DType.F32);
            backend.LayerNorm(finalNormed, h, _postLayerNormWeight!, _postLayerNormBias!, _config.LayerNormEps);
            h.Dispose();
            h = finalNormed;
        }
        return h;
    }

    /// <summary>Builds <c>[B, seqLen, hidden]</c> = concat(CLS broadcast over batch, patchSeq) + position_embedding (broadcast over batch).</summary>
    private void BuildEmbedded(Tensor output, Tensor patchSeq, int batch, int hidden)
    {
        float* outPtr = (float*)output.DataPointer;
        float* patchPtr = (float*)patchSeq.DataPointer;
        float* clsPtr = (float*)_classEmbedding!.DataPointer;
        float* posPtr = (float*)_positionEmbeddingWeight!.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            // Position 0: CLS embedding + position[0]
            int outOff = b * _seqLen * hidden;
            for (int h = 0; h < hidden; h++)
            {
                outPtr[outOff + h] = clsPtr[h] + posPtr[h];
            }
            // Positions 1..numPatches: patch token + position[i]
            for (int p = 0; p < _numPatches; p++)
            {
                int srcOff = (b * _numPatches + p) * hidden;
                int dstOff = (b * _seqLen + (p + 1)) * hidden;
                int posOff = (p + 1) * hidden;
                for (int h = 0; h < hidden; h++)
                {
                    outPtr[dstOff + h] = patchPtr[srcOff + h] + posPtr[posOff + h];
                }
            }
        }
    }

    /// <summary>Slice CLS token (index 0 of the sequence dim) to <c>[B, hidden]</c>.</summary>
    private Tensor SliceCls(Tensor lastHidden, int batch, int hidden)
    {
        Tensor cls = new Tensor(new TensorShape(batch, hidden), DType.F32);
        float* sp = (float*)lastHidden.DataPointer;
        float* dp = (float*)cls.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int srcOff = b * _seqLen * hidden;
            int dstOff = b * hidden;
            for (int h = 0; h < hidden; h++)
            {
                dp[dstOff + h] = sp[srcOff + h];
            }
        }
        return cls;
    }

    /// <summary>Project CLS through visual_projection (no bias, weight shape [projDim, hidden], output = x @ weight.T).</summary>
    private Tensor ProjectVisual(Tensor cls, int batch)
    {
        int hidden = _config.HiddenSize;
        int projDim = _config.ProjectionDim;
        Tensor projected = new Tensor(new TensorShape(batch, projDim), DType.F32);
        float* cp = (float*)cls.DataPointer;
        float* wp = (float*)_visualProjectionWeight!.DataPointer;
        float* pp = (float*)projected.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int inOff = b * hidden;
            int outOff = b * projDim;
            for (int o = 0; o < projDim; o++)
            {
                float sum = 0f;
                int wRow = o * hidden;
                for (int i = 0; i < hidden; i++)
                {
                    sum += cp[inOff + i] * wp[wRow + i];
                }
                pp[outOff + o] = sum;
            }
        }
        return projected;
    }
}

/// <summary>Single CLIP-Vision transformer layer — same shape as the text encoder's transformer
/// layer but without the causal mask (image tokens attend bidirectionally). Pre-norm style:
/// LayerNorm → SelfAttn → Residual → LayerNorm → MLP → Residual. Static helpers
/// (<c>ProjectLinear</c>, <c>ReshapeToMultiHead4D</c>, etc.) are duplicated here rather than
/// shared with the text encoder to keep the two paths independently auditable — the helpers
/// are small and the duplication keeps the perf-critical inner loops next to where they're used.</summary>
internal sealed unsafe class ClipVisionTransformerLayer
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _intermediateSize;
    private readonly float _layerNormEps;
    private readonly bool _useQuickGelu;

    private Tensor? _layerNorm1Weight, _layerNorm1Bias;
    private Tensor? _qProjWeight, _qProjBias;
    private Tensor? _kProjWeight, _kProjBias;
    private Tensor? _vProjWeight, _vProjBias;
    private Tensor? _outProjWeight, _outProjBias;
    private Tensor? _layerNorm2Weight, _layerNorm2Bias;
    private Tensor? _mlpFc1Weight, _mlpFc1Bias;
    private Tensor? _mlpFc2Weight, _mlpFc2Bias;

    public ClipVisionTransformerLayer(int hiddenSize, int numHeads, int intermediateSize, float layerNormEps, bool useQuickGelu)
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
        _layerNorm1Weight = TensorCasts.EnsureF32(weights[$"{prefix}.layer_norm1.weight"]);
        _layerNorm1Bias = TensorCasts.EnsureF32(weights[$"{prefix}.layer_norm1.bias"]);
        _qProjWeight = TensorCasts.EnsureF32(weights[$"{prefix}.self_attn.q_proj.weight"]);
        _qProjBias = TensorCasts.EnsureF32(weights[$"{prefix}.self_attn.q_proj.bias"]);
        _kProjWeight = TensorCasts.EnsureF32(weights[$"{prefix}.self_attn.k_proj.weight"]);
        _kProjBias = TensorCasts.EnsureF32(weights[$"{prefix}.self_attn.k_proj.bias"]);
        _vProjWeight = TensorCasts.EnsureF32(weights[$"{prefix}.self_attn.v_proj.weight"]);
        _vProjBias = TensorCasts.EnsureF32(weights[$"{prefix}.self_attn.v_proj.bias"]);
        _outProjWeight = TensorCasts.EnsureF32(weights[$"{prefix}.self_attn.out_proj.weight"]);
        _outProjBias = TensorCasts.EnsureF32(weights[$"{prefix}.self_attn.out_proj.bias"]);
        _layerNorm2Weight = TensorCasts.EnsureF32(weights[$"{prefix}.layer_norm2.weight"]);
        _layerNorm2Bias = TensorCasts.EnsureF32(weights[$"{prefix}.layer_norm2.bias"]);
        _mlpFc1Weight = TensorCasts.EnsureF32(weights[$"{prefix}.mlp.fc1.weight"]);
        _mlpFc1Bias = TensorCasts.EnsureF32(weights[$"{prefix}.mlp.fc1.bias"]);
        _mlpFc2Weight = TensorCasts.EnsureF32(weights[$"{prefix}.mlp.fc2.weight"]);
        _mlpFc2Bias = TensorCasts.EnsureF32(weights[$"{prefix}.mlp.fc2.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_layerNorm1Weight, _layerNorm1Bias, _qProjWeight, _qProjBias, _kProjWeight, _kProjBias, _vProjWeight, _vProjBias, _outProjWeight, _outProjBias, _layerNorm2Weight, _layerNorm2Bias, _mlpFc1Weight, _mlpFc1Bias, _mlpFc2Weight, _mlpFc2Bias];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor hidden)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];
        TensorShape shape = new TensorShape(batch, seqLen, _hiddenSize);

        Tensor normed1 = new Tensor(shape, DType.F32);
        backend.LayerNorm(normed1, hidden, _layerNorm1Weight!, _layerNorm1Bias!, _layerNormEps);
        Tensor attnOut = MultiHeadSelfAttention(backend, normed1, batch, seqLen);
        normed1.Dispose();
        Tensor residual1 = new Tensor(shape, DType.F32);
        backend.Add(residual1, hidden, attnOut);
        attnOut.Dispose();

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

    private Tensor MultiHeadSelfAttention(IBackend backend, Tensor input, int batch, int seqLen)
    {
        TensorShape seqShape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor query = ProjectLinear(backend, input, _qProjWeight!, _qProjBias!, batch, seqLen, _hiddenSize, _hiddenSize);
        Tensor key = ProjectLinear(backend, input, _kProjWeight!, _kProjBias!, batch, seqLen, _hiddenSize, _hiddenSize);
        Tensor value = ProjectLinear(backend, input, _vProjWeight!, _vProjBias!, batch, seqLen, _hiddenSize, _hiddenSize);

        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        Tensor queryMh = new Tensor(mhShape, DType.F32);
        Tensor keyMh = new Tensor(mhShape, DType.F32);
        Tensor valueMh = new Tensor(mhShape, DType.F32);
        ReshapeToMultiHead4D(queryMh, query, batch, seqLen, _numHeads, _headDim);
        ReshapeToMultiHead4D(keyMh, key, batch, seqLen, _numHeads, _headDim);
        ReshapeToMultiHead4D(valueMh, value, batch, seqLen, _numHeads, _headDim);
        query.Dispose(); key.Dispose(); value.Dispose();

        // No mask — vision attention is bidirectional.
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, queryMh, keyMh, valueMh, null, scale);
        queryMh.Dispose(); keyMh.Dispose(); valueMh.Dispose();

        Tensor merged = new Tensor(seqShape, DType.F32);
        ReshapeFromMultiHead4D(merged, attnOut, batch, seqLen, _numHeads, _headDim);
        attnOut.Dispose();

        Tensor projected = ProjectLinear(backend, merged, _outProjWeight!, _outProjBias!, batch, seqLen, _hiddenSize, _hiddenSize);
        merged.Dispose();
        return projected;
    }

    private Tensor MlpForward(IBackend backend, Tensor input, int batch, int seqLen)
    {
        Tensor fc1Out = ProjectLinear(backend, input, _mlpFc1Weight!, _mlpFc1Bias!, batch, seqLen, _hiddenSize, _intermediateSize);
        TensorShape fc1Shape = new TensorShape(batch, seqLen, _intermediateSize);
        Tensor activated = new Tensor(fc1Shape, DType.F32);
        if (_useQuickGelu) QuickGelu(activated, fc1Out);
        else backend.Gelu(activated, fc1Out);
        fc1Out.Dispose();
        Tensor fc2Out = ProjectLinear(backend, activated, _mlpFc2Weight!, _mlpFc2Bias!, batch, seqLen, _intermediateSize, _hiddenSize);
        activated.Dispose();
        return fc2Out;
    }

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

    private static Tensor ProjectLinear(IBackend backend, Tensor input, Tensor weight, Tensor bias, int batch, int seqLen, int inDim, int outDim)
    {
        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor output = new Tensor(outShape, DType.F32);
        TensorShape weightTShape = new TensorShape(inDim, outDim);
        Tensor weightT = new Tensor(weightTShape, DType.F32);
        TransposeMatrix(weight, weightT, outDim, inDim);
        backend.BatchedMatMul(output, input, weightT);
        weightT.Dispose();
        AddBiasBroadcast(output, bias, batch, seqLen, outDim);
        return output;
    }

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
                    int outOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        outPtr[outOffset + d] = inPtr[inOffset + d];
                    }
                }
            }
        }
    }

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

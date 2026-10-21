using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Siglip;

/// <summary>SigLIP vision tower. Structurally similar to CLIP's ViT but with three key differences:
/// <list type="number">
///   <item><b>No CLS token.</b> Position embeddings are <c>[num_patches, hidden]</c>, applied directly to the patch tokens.</item>
///   <item><b>No pre-norm before block 0.</b> Patch embed + position embed → transformer blocks (each with LN inside).</item>
///   <item><b>Attention-pooling head</b> instead of CLS extraction: a learned <c>probe</c> vector attends over the patch tokens via standard MHA → output → LayerNorm → 2-layer MLP. Produces the final image embedding.</item>
/// </list>
/// All linear projections route through <see cref="IBackend"/>. The attention pooling step uses
/// the combined <c>in_proj_weight/in_proj_bias</c> layout (PyTorch's <c>nn.MultiheadAttention</c>
/// convention) where Q/K/V are stacked in a single weight tensor of shape <c>[3*dim, dim]</c>.</summary>
public sealed unsafe class SiglipVisionEncoder
{
    private readonly SiglipPreset _preset;
    private readonly int _numPatches;

    // Patch embedding (Conv2D with kernel=stride=patch_size).
    private Tensor? _patchEmbeddingWeight; // [hidden, 3, patch, patch]
    private Tensor? _patchEmbeddingBias;   // [hidden]
    private Tensor? _positionEmbedding;    // [numPatches, hidden]

    // Transformer layers.
    private readonly SiglipTransformerLayer[] _layers;

    // Attention pooling head.
    private Tensor? _headProbe;            // [1, 1, hidden]
    private Tensor? _headAttnInProjWeight; // [3*hidden, hidden]
    private Tensor? _headAttnInProjBias;   // [3*hidden]
    private Tensor? _headAttnOutProjWeight; // [hidden, hidden]
    private Tensor? _headAttnOutProjBias;   // [hidden]
    private Tensor?[]? _headSlices;         // in_proj Q/K/V weight then bias slices, copied once
    private AttentionLayer? _headAttention;
    private Tensor? _headLnWeight, _headLnBias;
    private Tensor? _headMlpFc1Weight, _headMlpFc1Bias;
    private Tensor? _headMlpFc2Weight, _headMlpFc2Bias;

    public SiglipPreset Preset => _preset;
    public int EmbeddingDim => _preset.EmbeddingDim;

    public SiglipVisionEncoder(SiglipPreset preset)
    {
        _preset = preset;
        _numPatches = preset.NumPatches;
        _layers = new SiglipTransformerLayer[preset.NumLayers];
        for (int i = 0; i < preset.NumLayers; i++)
            _layers[i] = new SiglipTransformerLayer(preset.HiddenSize, preset.NumHeads, preset.IntermediateSize, preset.LayerNormEps);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "vision_model")
    {
        _patchEmbeddingWeight = EnsureF32(weights[$"{prefix}.embeddings.patch_embedding.weight"]);
        _patchEmbeddingBias = EnsureF32(weights[$"{prefix}.embeddings.patch_embedding.bias"]);
        _positionEmbedding = EnsureF32(weights[$"{prefix}.embeddings.position_embedding.weight"]);

        for (int i = 0; i < _layers.Length; i++)
            _layers[i].LoadWeights(weights, $"{prefix}.encoder.layers.{i}");

        // The post-encoder layernorm is named differently in SigLIP than CLIP — check both forms.
        if (weights.TryGetValue($"{prefix}.post_layernorm.weight", out Tensor? postLn))
        {
            // Post-encoder LN is applied to the patch tokens before they feed into the attention
            // pooling head. Most SigLIP checkpoints have it.
            _postLnWeight = EnsureF32(postLn);
            _postLnBias = EnsureF32(weights[$"{prefix}.post_layernorm.bias"]);
        }

        // Attention pooling head.
        _headProbe = EnsureF32(weights[$"{prefix}.head.probe"]);
        _headAttnInProjWeight = EnsureF32(weights[$"{prefix}.head.attention.in_proj_weight"]);
        _headAttnInProjBias = EnsureF32(weights[$"{prefix}.head.attention.in_proj_bias"]);
        _headAttnOutProjWeight = EnsureF32(weights[$"{prefix}.head.attention.out_proj.weight"]);
        _headAttnOutProjBias = EnsureF32(weights[$"{prefix}.head.attention.out_proj.bias"]);
        if (_headSlices is not null) foreach (Tensor? t in _headSlices) t?.Dispose();
        _headSlices = null;
        _headAttention = null;
        _headLnWeight = EnsureF32(weights[$"{prefix}.head.layernorm.weight"]);
        _headLnBias = EnsureF32(weights[$"{prefix}.head.layernorm.bias"]);
        _headMlpFc1Weight = EnsureF32(weights[$"{prefix}.head.mlp.fc1.weight"]);
        _headMlpFc1Bias = EnsureF32(weights[$"{prefix}.head.mlp.fc1.bias"]);
        _headMlpFc2Weight = EnsureF32(weights[$"{prefix}.head.mlp.fc2.weight"]);
        _headMlpFc2Bias = EnsureF32(weights[$"{prefix}.head.mlp.fc2.bias"]);
        HeadAttention();
    }

    private Tensor? _postLnWeight, _postLnBias;

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_patchEmbeddingWeight is not null) yield return _patchEmbeddingWeight;
        if (_patchEmbeddingBias is not null) yield return _patchEmbeddingBias;
        if (_positionEmbedding is not null) yield return _positionEmbedding;
        for (int i = 0; i < _layers.Length; i++)
            foreach (Tensor t in _layers[i].EnumerateWeights()) yield return t;
        if (_postLnWeight is not null) yield return _postLnWeight;
        if (_postLnBias is not null) yield return _postLnBias;
        Tensor?[] head = [_headProbe, _headAttnInProjWeight, _headAttnInProjBias,
            _headAttnOutProjWeight, _headAttnOutProjBias,
            _headLnWeight, _headLnBias,
            _headMlpFc1Weight, _headMlpFc1Bias, _headMlpFc2Weight, _headMlpFc2Bias];
        foreach (Tensor? t in head) if (t is not null) yield return t;
        if (_headSlices is not null) foreach (Tensor? t in _headSlices) if (t is not null) yield return t;
    }

    /// <summary>Encodes a preprocessed image into a <c>[1, embeddingDim]</c> tensor (un-normalized; caller L2-normalizes as needed).</summary>
    public Tensor Encode(IBackend backend, Tensor pixelValues)
    {
        int batch = (int)pixelValues.Shape[0];
        int hidden = _preset.HiddenSize;

        Tensor h = EncodeHiddenStates(backend, pixelValues);

        // Attention pooling head: probe (1 query) attends over patch tokens.
        Tensor pooled = AttentionPool(backend, h, batch, hidden);
        h.Dispose();

        // Head MLP: LN → FC1 + GELU → FC2 → residual add.
        Tensor embedding = HeadMlp(backend, pooled, batch, hidden);
        pooled.Dispose();
        return embedding;
    }

    /// <summary>Encodes a preprocessed image into the post-layernorm patch-token hidden states <c>[B, numPatches, hiddenSize]</c> — HF's <c>last_hidden_state</c>, the form FLUX.1 Redux consumes (the attention-pooling head is skipped).</summary>
    public Tensor EncodeHiddenStates(IBackend backend, Tensor pixelValues)
    {
        if (pixelValues.Shape.Rank != 4 || pixelValues.Shape[1] != 3
            || pixelValues.Shape[2] != _preset.ImageSize || pixelValues.Shape[3] != _preset.ImageSize)
        {
            throw new ArgumentException($"pixelValues must be [B, 3, {_preset.ImageSize}, {_preset.ImageSize}]; got {pixelValues.Shape}.", nameof(pixelValues));
        }

        int batch = (int)pixelValues.Shape[0];
        int hidden = _preset.HiddenSize;
        int patchGrid = _preset.ImageSize / _preset.PatchSize;

        // 1. Patch embedding: Conv2D with kernel=stride=patch_size, with bias.
        Tensor patchOut = new Tensor(new TensorShape(batch, hidden, patchGrid, patchGrid), DType.F32);
        backend.Conv2D(patchOut, pixelValues, _patchEmbeddingWeight!, _patchEmbeddingBias!,
            _preset.PatchSize, _preset.PatchSize, 0, 0);

        // 2. Reshape [B, hidden, gridH, gridW] → [B, numPatches, hidden] + add position embeddings.
        Tensor flat = patchOut.Reshape(new TensorShape(batch, hidden, _numPatches));
        TensorShape seqShape = new TensorShape(batch, _numPatches, hidden);
        Tensor seq = new Tensor(seqShape, DType.F32);
        backend.Transpose2D(seq, flat, hidden, _numPatches);
        patchOut.Dispose();
        AddPositionEmbedding(seq, batch);

        // 3. Run transformer layers.
        Tensor h = seq;
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, h);
            h.Dispose();
            h = next;
        }

        // 4. Optional post_layernorm (some SigLIP variants have it before the pooling head).
        if (_postLnWeight is not null)
        {
            Tensor normed = new Tensor(h.Shape, DType.F32);
            backend.LayerNorm(normed, h, _postLnWeight!, _postLnBias!, _preset.LayerNormEps);
            h.Dispose();
            h = normed;
        }
        return h;
    }

    private void AddPositionEmbedding(Tensor seq, int batch)
    {
        float* sp = (float*)seq.DataPointer;
        float* pp = (float*)_positionEmbedding!.DataPointer;
        int hidden = _preset.HiddenSize;
        for (int b = 0; b < batch; b++)
        {
            for (int p = 0; p < _numPatches; p++)
            {
                int sOff = (b * _numPatches + p) * hidden;
                int pOff = p * hidden;
                for (int h = 0; h < hidden; h++) sp[sOff + h] += pp[pOff + h];
            }
        }
    }

    /// <summary>Attention pooling: probe [1,1,D] queries patch tokens [B, numPatches, D] via
    /// standard multi-head attention, producing [B, 1, D]. Combined in_proj_weight is
    /// [3*D, D] following PyTorch's MultiheadAttention layout (Q stacked atop K stacked atop V).</summary>
    private Tensor AttentionPool(IBackend backend, Tensor patchTokens, int batch, int hidden)
    {
        // probe is [1, 1, hidden]; broadcast to [B, 1, hidden].
        TensorShape probeShape = new TensorShape(batch, 1, hidden);
        Tensor probe = new Tensor(probeShape, DType.F32);
        float* probePtr = (float*)probe.DataPointer;
        float* probeSrc = (float*)_headProbe!.DataPointer;
        for (int b = 0; b < batch; b++)
            new ReadOnlySpan<float>(probeSrc, hidden).CopyTo(new Span<float>(probePtr + b * hidden, hidden));

        // Q from probe; K, V from patch tokens, each from its slice of the combined in_proj weight.
        return HeadAttention().Forward(backend, probe, patchTokens, new AttentionCallContext { ReleaseQueryAfterProjection = true });
    }

    /// <summary>The pooling head's attention layer. The combined in_proj weight is <c>[3*D, D]</c> in PyTorch nn.MultiheadAttention layout
    /// (Q stacked atop K stacked atop V); its three <c>[D, D]</c> slices are copied once.</summary>
    private AttentionLayer HeadAttention()
    {
        if (_headAttention is not null) return _headAttention;
        int hidden = _preset.HiddenSize, heads = _preset.NumHeads;
        Tensor[] s = new Tensor[6];
        for (int i = 0; i < 3; i++)
        {
            s[i] = new Tensor(new TensorShape(hidden, hidden), DType.F32);
            new ReadOnlySpan<float>((float*)_headAttnInProjWeight!.DataPointer + (long)i * hidden * hidden, hidden * hidden)
                .CopyTo(new Span<float>((float*)s[i].DataPointer, hidden * hidden));
            s[3 + i] = new Tensor(new TensorShape(hidden), DType.F32);
            new ReadOnlySpan<float>((float*)_headAttnInProjBias!.DataPointer + (long)i * hidden, hidden)
                .CopyTo(new Span<float>((float*)s[3 + i].DataPointer, hidden));
        }
        _headSlices = s;
        return _headAttention = new AttentionLayer(
            new AttentionLayerConfig { Heads = heads, KvHeads = heads, HeadDim = hidden / heads, ComputeDType = DType.F32 },
            new AttentionLayerWeights
            {
                Query = s[0], QueryBias = s[3], Key = s[1], KeyBias = s[4], Value = s[2], ValueBias = s[5],
                Output = _headAttnOutProjWeight, OutputBias = _headAttnOutProjBias,
            });
    }

    /// <summary>Head MLP: <c>output = pooled + Fc2(GELU(Fc1(LN(pooled))))</c>. Standard residual transformer FFN.</summary>
    private Tensor HeadMlp(IBackend backend, Tensor pooled, int batch, int hidden)
    {
        TensorShape pShape = pooled.Shape;
        // LN.
        Tensor normed = new Tensor(pShape, DType.F32);
        backend.LayerNorm(normed, pooled, _headLnWeight!, _headLnBias!, _preset.LayerNormEps);

        // Fc1: hidden → intermediate.
        int interSize = _preset.IntermediateSize;
        Tensor fc1Out = new Tensor(new TensorShape(batch, 1, interSize), DType.F32);
        backend.Linear(fc1Out, normed, _headMlpFc1Weight!, _headMlpFc1Bias!);
        normed.Dispose();

        // GELU.
        Tensor activated = new Tensor(fc1Out.Shape, DType.F32);
        backend.Gelu(activated, fc1Out);
        fc1Out.Dispose();

        // Fc2: intermediate → hidden.
        Tensor fc2Out = new Tensor(new TensorShape(batch, 1, hidden), DType.F32);
        backend.Linear(fc2Out, activated, _headMlpFc2Weight!, _headMlpFc2Bias!);
        activated.Dispose();

        // Residual add (pooled + MLP output).
        Tensor result = new Tensor(new TensorShape(batch, 1, hidden), DType.F32);
        backend.Add(result, pooled, fc2Out);
        fc2Out.Dispose();

        // Reshape [B, 1, D] → [B, D] for return.
        Tensor flat = result.Reshape(new TensorShape(batch, hidden));
        Tensor copy = new Tensor(flat.Shape, DType.F32);
        new ReadOnlySpan<float>((float*)flat.DataPointer, batch * hidden).CopyTo(new Span<float>((float*)copy.DataPointer, batch * hidden));
        result.Dispose();
        return copy;
    }

    private static Tensor EnsureF32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}

/// <summary>Single SigLIP transformer layer: LN → MHA (with separate Q/K/V) → Add → LN → MLP (GELU) → Add.</summary>
internal sealed unsafe class SiglipTransformerLayer
{
    private readonly int _hidden;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _interSize;
    private readonly float _lnEps;

    private Tensor? _ln1W, _ln1B, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB;
    private Tensor? _ln2W, _ln2B, _fc1W, _fc1B, _fc2W, _fc2B;
    private AttentionLayer? _attn;

    public SiglipTransformerLayer(int hidden, int numHeads, int interSize, float lnEps)
    {
        _hidden = hidden;
        _numHeads = numHeads;
        _headDim = hidden / numHeads;
        _interSize = interSize;
        _lnEps = lnEps;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _ln1W = EnsureF32(weights[$"{prefix}.layer_norm1.weight"]);
        _ln1B = EnsureF32(weights[$"{prefix}.layer_norm1.bias"]);
        _qW = EnsureF32(weights[$"{prefix}.self_attn.q_proj.weight"]);
        _qB = EnsureF32(weights[$"{prefix}.self_attn.q_proj.bias"]);
        _kW = EnsureF32(weights[$"{prefix}.self_attn.k_proj.weight"]);
        _kB = EnsureF32(weights[$"{prefix}.self_attn.k_proj.bias"]);
        _vW = EnsureF32(weights[$"{prefix}.self_attn.v_proj.weight"]);
        _vB = EnsureF32(weights[$"{prefix}.self_attn.v_proj.bias"]);
        _oW = EnsureF32(weights[$"{prefix}.self_attn.out_proj.weight"]);
        _oB = EnsureF32(weights[$"{prefix}.self_attn.out_proj.bias"]);
        _ln2W = EnsureF32(weights[$"{prefix}.layer_norm2.weight"]);
        _ln2B = EnsureF32(weights[$"{prefix}.layer_norm2.bias"]);
        _fc1W = EnsureF32(weights[$"{prefix}.mlp.fc1.weight"]);
        _fc1B = EnsureF32(weights[$"{prefix}.mlp.fc1.bias"]);
        _fc2W = EnsureF32(weights[$"{prefix}.mlp.fc2.weight"]);
        _fc2B = EnsureF32(weights[$"{prefix}.mlp.fc2.bias"]);
        _attn = new AttentionLayer(
            new AttentionLayerConfig { Heads = _numHeads, KvHeads = _numHeads, HeadDim = _headDim, ComputeDType = DType.F32 },
            new AttentionLayerWeights
            {
                Query = _qW, QueryBias = _qB, Key = _kW, KeyBias = _kB, Value = _vW, ValueBias = _vB, Output = _oW, OutputBias = _oB,
            });
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_ln1W, _ln1B, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _ln2W, _ln2B, _fc1W, _fc1B, _fc2W, _fc2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor hidden)
    {
        int batch = (int)hidden.Shape[0];
        int seq = (int)hidden.Shape[1];
        TensorShape shape = new TensorShape(batch, seq, _hidden);

        // LN → MHA.
        Tensor normed1 = new Tensor(shape, DType.F32);
        backend.LayerNorm(normed1, hidden, _ln1W!, _ln1B!, _lnEps);
        Tensor attnOut = _attn!.Forward(backend, normed1, normed1, default);
        normed1.Dispose();
        Tensor residual1 = new Tensor(shape, DType.F32);
        backend.Add(residual1, hidden, attnOut);
        attnOut.Dispose();

        // LN → MLP.
        Tensor normed2 = new Tensor(shape, DType.F32);
        backend.LayerNorm(normed2, residual1, _ln2W!, _ln2B!, _lnEps);
        Tensor mlpOut = Mlp(backend, normed2, batch, seq);
        normed2.Dispose();
        Tensor residual2 = new Tensor(shape, DType.F32);
        backend.Add(residual2, residual1, mlpOut);
        residual1.Dispose();
        mlpOut.Dispose();
        return residual2;
    }

    private Tensor Mlp(IBackend backend, Tensor input, int batch, int seq)
    {
        Tensor fc1Out = new Tensor(new TensorShape(batch, seq, _interSize), DType.F32);
        backend.Linear(fc1Out, input, _fc1W!, _fc1B!);
        Tensor activated = new Tensor(fc1Out.Shape, DType.F32);
        backend.Gelu(activated, fc1Out);
        fc1Out.Dispose();
        Tensor fc2Out = new Tensor(new TensorShape(batch, seq, _hidden), DType.F32);
        backend.Linear(fc2Out, activated, _fc2W!, _fc2B!);
        activated.Dispose();
        return fc2Out;
    }

    private static Tensor EnsureF32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}

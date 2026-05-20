using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Whisper;

/// <summary>Whisper text decoder. Autoregressive transformer over token IDs that
/// cross-attends to the encoder's audio features.
///
/// <para><b>KV-cache pattern:</b> on each decode step the self-attention K/V tensors
/// are appended for the new position; the encoder cross-attention K/V are computed
/// once via <see cref="PrecomputeCrossKv"/> and reused. We expose
/// <see cref="StartDecode"/> / <see cref="DecodeStep"/> so the pipeline owns the
/// growing token buffer and can stop on EOT without per-step reallocation. The cache
/// is sized to <c>MaxTextPositions = 448</c> at start to avoid per-step grow.</para>
///
/// <para><b>Weight tying:</b> HF safetensors usually omit <c>proj_out.weight</c>;
/// when missing we use <c>model.decoder.embed_tokens.weight</c> transposed.</para></summary>
public sealed unsafe class WhisperDecoder : IDisposable
{
    private readonly WhisperConfig _cfg;
    private readonly WhisperDecoderLayer[] _layers;

    // Embeddings
    private Tensor? _embedTokens;       // [vocab, d_model]
    private Tensor? _embedPositions;    // [max_text_positions, d_model] — learned

    // Final layer norm
    private Tensor? _finalLnWeight;
    private Tensor? _finalLnBias;

    // Output projection (often weight-tied to embed_tokens)
    private Tensor? _projOutWeight;
    private bool _projOutIsTied;

    private bool _weightsLoaded;
    private int _disposed;

    /// <summary>Config this decoder was built with.</summary>
    public WhisperConfig Config => _cfg;

    public WhisperDecoder(WhisperConfig cfg)
    {
        _cfg = cfg;
        _layers = new WhisperDecoderLayer[cfg.DecoderLayers];
        for (int i = 0; i < cfg.DecoderLayers; i++)
            _layers[i] = new WhisperDecoderLayer(cfg);
    }

    /// <summary>Loads weights from a HF safetensors dictionary. Standard layout uses
    /// the <c>model.decoder.*</c> prefix; <c>proj_out.weight</c> sits at the root and
    /// is auto-tied to <c>embed_tokens.weight</c> when absent.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "model.decoder")
    {
        _embedTokens = WhisperOps.EnsureF32(weights[$"{prefix}.embed_tokens.weight"]);
        _embedPositions = WhisperOps.EnsureF32(weights[$"{prefix}.embed_positions.weight"]);

        for (int i = 0; i < _layers.Length; i++)
            _layers[i].LoadWeights(weights, $"{prefix}.layers.{i}");

        _finalLnWeight = WhisperOps.EnsureF32(weights[$"{prefix}.layer_norm.weight"]);
        _finalLnBias = WhisperOps.EnsureF32(weights[$"{prefix}.layer_norm.bias"]);

        if (weights.TryGetValue("proj_out.weight", out Tensor? proj))
        {
            _projOutWeight = WhisperOps.EnsureF32(proj);
            _projOutIsTied = false;
        }
        else
        {
            // Weight tying — embed_tokens.weight is [vocab, d_model]; logits = hidden @ embed^T.
            // We reuse the embed tensor directly and apply the transpose at compute time.
            _projOutWeight = _embedTokens;
            _projOutIsTied = true;
        }
        _weightsLoaded = true;
    }

    /// <summary>State carried across decode steps. The pipeline owns one of these for
    /// the duration of a single transcription.</summary>
    public sealed class DecodeState : IDisposable
    {
        private readonly WhisperConfig _cfg;
        /// <summary>Cross-attention K/V precomputed from the encoder output. Indexed by layer.</summary>
        public Tensor[] CrossKey { get; }
        public Tensor[] CrossValue { get; }
        /// <summary>Self-attention K/V cache for each layer. Pre-allocated to
        /// <c>[1, num_heads, MaxTextPositions, head_dim]</c>; the valid prefix is
        /// <see cref="CurrentPos"/> entries.</summary>
        public Tensor[] SelfKey { get; }
        public Tensor[] SelfValue { get; }
        /// <summary>Number of tokens written into the self-attention cache so far.</summary>
        public int CurrentPos { get; set; }

        public DecodeState(WhisperConfig cfg, int encoderSeqLen)
        {
            _cfg = cfg;
            CrossKey = new Tensor[cfg.DecoderLayers];
            CrossValue = new Tensor[cfg.DecoderLayers];
            SelfKey = new Tensor[cfg.DecoderLayers];
            SelfValue = new Tensor[cfg.DecoderLayers];
            TensorShape crossShape = new(1, cfg.NumHeads, encoderSeqLen, cfg.HeadDim);
            TensorShape selfShape = new(1, cfg.NumHeads, cfg.MaxTextPositions, cfg.HeadDim);
            for (int i = 0; i < cfg.DecoderLayers; i++)
            {
                CrossKey[i] = new Tensor(crossShape, DType.F32);
                CrossValue[i] = new Tensor(crossShape, DType.F32);
                SelfKey[i] = new Tensor(selfShape, DType.F32);
                SelfValue[i] = new Tensor(selfShape, DType.F32);
            }
        }

        public void Dispose()
        {
            foreach (Tensor t in CrossKey) t.Dispose();
            foreach (Tensor t in CrossValue) t.Dispose();
            foreach (Tensor t in SelfKey) t.Dispose();
            foreach (Tensor t in SelfValue) t.Dispose();
        }
    }

    /// <summary>Allocates fresh decoder state and precomputes cross-attention K/V from
    /// the encoder output. The pipeline calls this once per audio chunk.</summary>
    public DecodeState StartDecode(IBackend backend, Tensor encoderHidden)
    {
        ThrowIfDisposed();
        if (!_weightsLoaded) throw new InvalidOperationException("Call LoadWeights before StartDecode.");

        int batch = (int)encoderHidden.Shape[0];
        if (batch != 1) throw new NotSupportedException("WhisperDecoder currently supports batch=1 only.");
        int encSeq = (int)encoderHidden.Shape[1];

        DecodeState state = new(_cfg, encSeq);
        for (int i = 0; i < _layers.Length; i++)
            _layers[i].PrecomputeCrossKv(backend, encoderHidden, state.CrossKey[i], state.CrossValue[i], encSeq);
        return state;
    }

    /// <summary>Runs the decoder for a single new token position. <paramref name="tokenIds"/>
    /// is a span of <see cref="WhisperDecoder.DecodeState.CurrentPos"/> existing tokens plus
    /// the newest token at the end — i.e. on first call pass the full prompt sequence
    /// (e.g. <c>[SOT, lang, transcribe, notimestamps]</c>) and on subsequent calls pass
    /// either the full sequence so far or just the newly appended token; the decoder
    /// uses positions <c>[CurrentPos, CurrentPos + len(tokenIds))</c> of the cache.</summary>
    /// <returns>Logits over the vocabulary <c>[1, vocabSize]</c> for the LAST token in
    /// <paramref name="tokenIds"/>.</returns>
    public Tensor DecodeStep(IBackend backend, ReadOnlySpan<int> tokenIds, DecodeState state)
    {
        ThrowIfDisposed();
        int newCount = tokenIds.Length;
        if (newCount == 0) throw new ArgumentException("tokenIds is empty", nameof(tokenIds));
        if (state.CurrentPos + newCount > _cfg.MaxTextPositions)
            throw new InvalidOperationException($"Decode would exceed MaxTextPositions ({_cfg.MaxTextPositions})");

        int d = _cfg.HiddenSize;
        int posStart = state.CurrentPos;

        // Embed the new tokens. hidden shape: [1, newCount, d_model].
        Tensor hidden = new(new TensorShape(1, newCount, d), DType.F32);
        EmbedAndAddPos(hidden, tokenIds, posStart, d);

        // Run through decoder layers.
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, hidden,
                state.SelfKey[i], state.SelfValue[i],
                state.CrossKey[i], state.CrossValue[i],
                posStart, newCount);
            hidden.Dispose();
            hidden = next;
        }

        // Final LN.
        Tensor normed = new(hidden.Shape, DType.F32);
        backend.LayerNorm(normed, hidden, _finalLnWeight!, _finalLnBias!, _cfg.LayerNormEps);
        hidden.Dispose();

        // Compute logits for the LAST token only (the only one we sample on).
        // Slice [1, 1, d] out of [1, newCount, d].
        Tensor lastHidden = new(new TensorShape(1, 1, d), DType.F32);
        float* normedPtr = (float*)normed.DataPointer;
        float* lastPtr = (float*)lastHidden.DataPointer;
        int srcOff = (newCount - 1) * d;
        for (int k = 0; k < d; k++) lastPtr[k] = normedPtr[srcOff + k];
        normed.Dispose();

        // Logits = lastHidden @ embed^T (weight-tied case) or lastHidden @ proj_out^T.
        Tensor logits = new(new TensorShape(1, _cfg.VocabSize), DType.F32);
        ComputeLogits(lastHidden, logits, _cfg.VocabSize, d);
        lastHidden.Dispose();

        state.CurrentPos += newCount;
        return logits;
    }

    private void EmbedAndAddPos(Tensor hidden, ReadOnlySpan<int> tokenIds, int posStart, int d)
    {
        float* hPtr = (float*)hidden.DataPointer;
        float* eTok = (float*)_embedTokens!.DataPointer;
        float* ePos = (float*)_embedPositions!.DataPointer;
        for (int s = 0; s < tokenIds.Length; s++)
        {
            int tok = tokenIds[s];
            int pos = posStart + s;
            int hOff = s * d;
            int tokOff = tok * d;
            int posOff = pos * d;
            for (int k = 0; k < d; k++) hPtr[hOff + k] = eTok[tokOff + k] + ePos[posOff + k];
        }
    }

    private void ComputeLogits(Tensor hidden, Tensor logits, int vocab, int d)
    {
        // logits[v] = sum_k hidden[k] * embed[v, k]. The embed weight is stored as
        // [vocab, d_model] — no transpose needed here, the row already corresponds to a
        // vocab entry. (Same layout when proj_out exists, which HF stores as [vocab, d].)
        float* hPtr = (float*)hidden.DataPointer;
        float* wPtr = (float*)_projOutWeight!.DataPointer;
        float* lPtr = (float*)logits.DataPointer;
        for (int v = 0; v < vocab; v++)
        {
            float acc = 0f;
            int wRow = v * d;
            for (int k = 0; k < d; k++) acc += hPtr[k] * wPtr[wRow + k];
            lPtr[v] = acc;
        }
        _ = _projOutIsTied; // tied vs untied are computed identically; field kept for debugging.
    }

    /// <summary>Enumerates every loaded weight tensor for backend preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embedTokens is not null) yield return _embedTokens;
        if (_embedPositions is not null) yield return _embedPositions;
        foreach (WhisperDecoderLayer layer in _layers)
            foreach (Tensor t in layer.EnumerateWeights()) yield return t;
        if (_finalLnWeight is not null) yield return _finalLnWeight;
        if (_finalLnBias is not null) yield return _finalLnBias;
        if (_projOutWeight is not null && !_projOutIsTied) yield return _projOutWeight;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(WhisperDecoder));
    }

    public void Dispose() { Interlocked.Exchange(ref _disposed, 1); }
}

/// <summary>Single Whisper decoder block. Pre-norm self-attention (causal, with KV
/// cache) → pre-norm cross-attention (against precomputed encoder K/V) → pre-norm MLP.
/// Weight names match the HF transformers Whisper layout
/// (<c>self_attn / encoder_attn / fc1 / fc2</c>).</summary>
internal sealed unsafe class WhisperDecoderLayer
{
    private readonly WhisperConfig _cfg;

    // Self-attention
    private Tensor? _selfLnW; private Tensor? _selfLnB;
    private Tensor? _selfQW; private Tensor? _selfQB;
    private Tensor? _selfKW;
    private Tensor? _selfVW; private Tensor? _selfVB;
    private Tensor? _selfOutW; private Tensor? _selfOutB;

    // Cross-attention
    private Tensor? _crossLnW; private Tensor? _crossLnB;
    private Tensor? _crossQW; private Tensor? _crossQB;
    private Tensor? _crossKW;
    private Tensor? _crossVW; private Tensor? _crossVB;
    private Tensor? _crossOutW; private Tensor? _crossOutB;

    // MLP
    private Tensor? _fc1W; private Tensor? _fc1B;
    private Tensor? _fc2W; private Tensor? _fc2B;
    private Tensor? _finalLnW; private Tensor? _finalLnB;

    public WhisperDecoderLayer(WhisperConfig cfg) { _cfg = cfg; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _selfLnW = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn_layer_norm.weight"]);
        _selfLnB = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn_layer_norm.bias"]);
        _selfQW = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.q_proj.weight"]);
        _selfQB = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.q_proj.bias"]);
        _selfKW = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.k_proj.weight"]);
        _selfVW = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.v_proj.weight"]);
        _selfVB = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.v_proj.bias"]);
        _selfOutW = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.out_proj.weight"]);
        _selfOutB = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.out_proj.bias"]);

        _crossLnW = WhisperOps.EnsureF32(weights[$"{prefix}.encoder_attn_layer_norm.weight"]);
        _crossLnB = WhisperOps.EnsureF32(weights[$"{prefix}.encoder_attn_layer_norm.bias"]);
        _crossQW = WhisperOps.EnsureF32(weights[$"{prefix}.encoder_attn.q_proj.weight"]);
        _crossQB = WhisperOps.EnsureF32(weights[$"{prefix}.encoder_attn.q_proj.bias"]);
        _crossKW = WhisperOps.EnsureF32(weights[$"{prefix}.encoder_attn.k_proj.weight"]);
        _crossVW = WhisperOps.EnsureF32(weights[$"{prefix}.encoder_attn.v_proj.weight"]);
        _crossVB = WhisperOps.EnsureF32(weights[$"{prefix}.encoder_attn.v_proj.bias"]);
        _crossOutW = WhisperOps.EnsureF32(weights[$"{prefix}.encoder_attn.out_proj.weight"]);
        _crossOutB = WhisperOps.EnsureF32(weights[$"{prefix}.encoder_attn.out_proj.bias"]);

        _fc1W = WhisperOps.EnsureF32(weights[$"{prefix}.fc1.weight"]);
        _fc1B = WhisperOps.EnsureF32(weights[$"{prefix}.fc1.bias"]);
        _fc2W = WhisperOps.EnsureF32(weights[$"{prefix}.fc2.weight"]);
        _fc2B = WhisperOps.EnsureF32(weights[$"{prefix}.fc2.bias"]);
        _finalLnW = WhisperOps.EnsureF32(weights[$"{prefix}.final_layer_norm.weight"]);
        _finalLnB = WhisperOps.EnsureF32(weights[$"{prefix}.final_layer_norm.bias"]);
    }

    /// <summary>Precomputes cross-attention K and V from the encoder hidden state.
    /// Run once per audio chunk at decode start; the result is held in
    /// <see cref="WhisperDecoder.DecodeState"/>.</summary>
    public void PrecomputeCrossKv(IBackend backend, Tensor encHidden, Tensor outK, Tensor outV, int encSeqLen)
    {
        int d = _cfg.HiddenSize;
        // Project K, V from encoder hidden [1, encSeq, d] → [1, encSeq, d].
        Tensor k = WhisperOps.ProjectLinear(backend, encHidden, _crossKW!, bias: null, 1, encSeqLen, d, d);
        Tensor v = WhisperOps.ProjectLinear(backend, encHidden, _crossVW!, _crossVB, 1, encSeqLen, d, d);
        // Reshape into [1, H, encSeq, headDim] directly into the state tensors.
        WhisperOps.ReshapeToMultiHead4D(outK, k, 1, encSeqLen, _cfg.NumHeads, _cfg.HeadDim);
        WhisperOps.ReshapeToMultiHead4D(outV, v, 1, encSeqLen, _cfg.NumHeads, _cfg.HeadDim);
        k.Dispose(); v.Dispose();
    }

    /// <summary>Forward pass for a chunk of <paramref name="newCount"/> new positions
    /// starting at absolute position <paramref name="posStart"/>. Updates the self-attn
    /// KV cache slots [posStart, posStart+newCount).</summary>
    public Tensor Forward(IBackend backend, Tensor hidden,
        Tensor selfK, Tensor selfV, Tensor crossK, Tensor crossV,
        int posStart, int newCount)
    {
        int d = _cfg.HiddenSize;
        int totalPos = posStart + newCount;
        TensorShape inShape = new(1, newCount, d);

        // --- Self-attention sub-block (causal, with KV cache append) ---
        Tensor normed = new(inShape, DType.F32);
        backend.LayerNorm(normed, hidden, _selfLnW!, _selfLnB!, _cfg.LayerNormEps);
        Tensor q = WhisperOps.ProjectLinear(backend, normed, _selfQW!, _selfQB, 1, newCount, d, d);
        Tensor k = WhisperOps.ProjectLinear(backend, normed, _selfKW!, bias: null, 1, newCount, d, d);
        Tensor v = WhisperOps.ProjectLinear(backend, normed, _selfVW!, _selfVB, 1, newCount, d, d);
        normed.Dispose();

        TensorShape mhNew = new(1, _cfg.NumHeads, newCount, _cfg.HeadDim);
        Tensor qMh = new(mhNew, DType.F32);
        WhisperOps.ReshapeToMultiHead4D(qMh, q, 1, newCount, _cfg.NumHeads, _cfg.HeadDim);
        q.Dispose();

        // Write new K, V into the cache slots [posStart..posStart+newCount).
        Tensor kMhNew = new(mhNew, DType.F32);
        Tensor vMhNew = new(mhNew, DType.F32);
        WhisperOps.ReshapeToMultiHead4D(kMhNew, k, 1, newCount, _cfg.NumHeads, _cfg.HeadDim);
        WhisperOps.ReshapeToMultiHead4D(vMhNew, v, 1, newCount, _cfg.NumHeads, _cfg.HeadDim);
        k.Dispose(); v.Dispose();
        AppendToKvCache(selfK, kMhNew, posStart, _cfg.NumHeads, _cfg.MaxTextPositions, _cfg.HeadDim, newCount);
        AppendToKvCache(selfV, vMhNew, posStart, _cfg.NumHeads, _cfg.MaxTextPositions, _cfg.HeadDim, newCount);
        kMhNew.Dispose(); vMhNew.Dispose();

        // Build a view of the cache covering [0..totalPos).
        Tensor kCached = SliceKvPrefix(selfK, _cfg.NumHeads, totalPos, _cfg.MaxTextPositions, _cfg.HeadDim);
        Tensor vCached = SliceKvPrefix(selfV, _cfg.NumHeads, totalPos, _cfg.MaxTextPositions, _cfg.HeadDim);

        // Causal mask for the new positions: each new position s attends to cached
        // positions [0..posStart+s]. Mask shape: [1, 1, newCount, totalPos].
        Tensor causalMask = BuildIncrementalCausalMask(posStart, newCount, totalPos);

        float scale = 1f / MathF.Sqrt(_cfg.HeadDim);
        Tensor attnOut = new(mhNew, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kCached, vCached, causalMask, scale);
        qMh.Dispose(); kCached.Dispose(); vCached.Dispose(); causalMask.Dispose();

        Tensor merged = new(inShape, DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(merged, attnOut, 1, newCount, _cfg.NumHeads, _cfg.HeadDim);
        attnOut.Dispose();
        Tensor selfProj = WhisperOps.ProjectLinear(backend, merged, _selfOutW!, _selfOutB, 1, newCount, d, d);
        merged.Dispose();
        Tensor res1 = new(inShape, DType.F32);
        backend.Add(res1, hidden, selfProj);
        selfProj.Dispose();

        // --- Cross-attention sub-block ---
        Tensor normed2 = new(inShape, DType.F32);
        backend.LayerNorm(normed2, res1, _crossLnW!, _crossLnB!, _cfg.LayerNormEps);
        Tensor crossQ = WhisperOps.ProjectLinear(backend, normed2, _crossQW!, _crossQB, 1, newCount, d, d);
        normed2.Dispose();
        Tensor crossQMh = new(mhNew, DType.F32);
        WhisperOps.ReshapeToMultiHead4D(crossQMh, crossQ, 1, newCount, _cfg.NumHeads, _cfg.HeadDim);
        crossQ.Dispose();

        Tensor crossAttn = new(mhNew, DType.F32);
        backend.ScaledDotProductAttention(crossAttn, crossQMh, crossK, crossV, mask: null, scale);
        crossQMh.Dispose();
        Tensor crossMerged = new(inShape, DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(crossMerged, crossAttn, 1, newCount, _cfg.NumHeads, _cfg.HeadDim);
        crossAttn.Dispose();
        Tensor crossProj = WhisperOps.ProjectLinear(backend, crossMerged, _crossOutW!, _crossOutB, 1, newCount, d, d);
        crossMerged.Dispose();
        Tensor res2 = new(inShape, DType.F32);
        backend.Add(res2, res1, crossProj);
        res1.Dispose(); crossProj.Dispose();

        // --- MLP sub-block ---
        Tensor normed3 = new(inShape, DType.F32);
        backend.LayerNorm(normed3, res2, _finalLnW!, _finalLnB!, _cfg.LayerNormEps);
        Tensor fc1 = WhisperOps.ProjectLinear(backend, normed3, _fc1W!, _fc1B, 1, newCount, d, _cfg.IntermediateSize);
        normed3.Dispose();
        Tensor activated = new(new TensorShape(1, newCount, _cfg.IntermediateSize), DType.F32);
        backend.Gelu(activated, fc1);
        fc1.Dispose();
        Tensor fc2 = WhisperOps.ProjectLinear(backend, activated, _fc2W!, _fc2B, 1, newCount, _cfg.IntermediateSize, d);
        activated.Dispose();
        Tensor res3 = new(inShape, DType.F32);
        backend.Add(res3, res2, fc2);
        res2.Dispose(); fc2.Dispose();
        return res3;
    }

    /// <summary>Writes <paramref name="src"/> ([1, H, newCount, D]) into <paramref name="cache"/>
    /// ([1, H, MaxPos, D]) starting at position <paramref name="posStart"/>.</summary>
    private static void AppendToKvCache(Tensor cache, Tensor src, int posStart, int heads, int maxPos, int headDim, int newCount)
    {
        float* cPtr = (float*)cache.DataPointer;
        float* sPtr = (float*)src.DataPointer;
        for (int h = 0; h < heads; h++)
            for (int s = 0; s < newCount; s++)
            {
                int sOff = (h * newCount + s) * headDim;
                int cOff = (h * maxPos + posStart + s) * headDim;
                for (int d = 0; d < headDim; d++) cPtr[cOff + d] = sPtr[sOff + d];
            }
    }

    /// <summary>Returns a fresh [1, H, totalPos, D] tensor copied out of the
    /// MaxPos-allocated cache. We materialize rather than view because SDPA expects a
    /// contiguous [S_kv, D] inner block per head and the cache is over-allocated.</summary>
    private static Tensor SliceKvPrefix(Tensor cache, int heads, int totalPos, int maxPos, int headDim)
    {
        Tensor sliced = new(new TensorShape(1, heads, totalPos, headDim), DType.F32);
        float* cPtr = (float*)cache.DataPointer;
        float* sPtr = (float*)sliced.DataPointer;
        for (int h = 0; h < heads; h++)
        {
            int cBase = h * maxPos * headDim;
            int sBase = h * totalPos * headDim;
            for (int p = 0; p < totalPos; p++)
            {
                int cOff = cBase + p * headDim;
                int sOff = sBase + p * headDim;
                for (int d = 0; d < headDim; d++) sPtr[sOff + d] = cPtr[cOff + d];
            }
        }
        return sliced;
    }

    /// <summary>Builds the mask for new-token self-attention. Shape [1, 1, newCount, totalPos];
    /// position s in the new chunk (= absolute position posStart+s) can see cache positions
    /// [0..posStart+s], everything beyond is -inf.</summary>
    private static Tensor BuildIncrementalCausalMask(int posStart, int newCount, int totalPos)
    {
        Tensor mask = new(new TensorShape(1, 1, newCount, totalPos), DType.F32);
        float* p = (float*)mask.DataPointer;
        for (int s = 0; s < newCount; s++)
        {
            int allowedEnd = posStart + s;
            int rowOff = s * totalPos;
            for (int kp = 0; kp < totalPos; kp++)
                p[rowOff + kp] = kp <= allowedEnd ? 0f : float.NegativeInfinity;
        }
        return mask;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all =
        [
            _selfLnW, _selfLnB, _selfQW, _selfQB, _selfKW, _selfVW, _selfVB, _selfOutW, _selfOutB,
            _crossLnW, _crossLnB, _crossQW, _crossQB, _crossKW, _crossVW, _crossVB, _crossOutW, _crossOutB,
            _fc1W, _fc1B, _fc2W, _fc2B, _finalLnW, _finalLnB,
        ];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}

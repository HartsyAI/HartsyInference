using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Whisper;

/// <summary>Whisper text decoder — an autoregressive transformer over token IDs that cross-attends to the encoder's audio features.</summary>
/// <remarks><b>KV-cache pattern:</b> on each decode step the self-attention K/V tensors are appended for the new position; the encoder cross-attention K/V are computed once via <see cref="PrecomputeCrossKv"/> and reused. We expose <see cref="StartDecode"/> / <see cref="DecodeStep"/> so the pipeline owns the growing token buffer and can stop on EOT without per-step reallocation. The cache is sized to <c>MaxTextPositions = 448</c> at start to avoid per-step grow.
///
/// <para><b>Weight tying:</b> HF safetensors usually omit <c>proj_out.weight</c>; when missing we use <c>model.decoder.embed_tokens.weight</c> transposed.</para></remarks>
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

    // Host copies of the two embedding tables, taken at load. The token table doubles as the tied logits weight and
    // goes device-resident, and a DataPointer read of a device-resident weight gives that copy up; weights are never
    // written, so the host bytes stay authoritative and the per-step lookup reads them without touching residency.
    private float* _embedTokensHost;
    private float* _embedPositionsHost;

    // Everything a device op reads, kept resident for the decode (the position table is only read on the host).
    private Tensor[] _deviceWeights = [];

    private bool _weightsLoaded;
    private int _disposed;

    public WhisperConfig Config => _cfg;

    public WhisperDecoder(WhisperConfig cfg)
    {
        _cfg = cfg;
        _layers = new WhisperDecoderLayer[cfg.DecoderLayers];
        for (int i = 0; i < cfg.DecoderLayers; i++)
            _layers[i] = new WhisperDecoderLayer(cfg);
    }

    /// <summary>Loads weights from a HF safetensors dictionary; standard layout uses the <c>model.decoder.*</c> prefix, with <c>proj_out.weight</c> at the root, auto-tied to <c>embed_tokens.weight</c> when absent.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "model.decoder")
    {
        _embedTokens = WhisperOps.EnsureF32(weights[$"{prefix}.embed_tokens.weight"]);
        RequireVocabShape(_embedTokens, $"{prefix}.embed_tokens.weight");
        _embedPositions = WhisperOps.EnsureF32(weights[$"{prefix}.embed_positions.weight"]);

        for (int i = 0; i < _layers.Length; i++)
            _layers[i].LoadWeights(weights, $"{prefix}.layers.{i}");

        _finalLnWeight = WhisperOps.EnsureF32(weights[$"{prefix}.layer_norm.weight"]);
        _finalLnBias = WhisperOps.EnsureF32(weights[$"{prefix}.layer_norm.bias"]);

        if (weights.TryGetValue("proj_out.weight", out Tensor? proj))
        {
            _projOutWeight = WhisperOps.EnsureF32(proj);
            RequireVocabShape(_projOutWeight, "proj_out.weight");
            _projOutIsTied = false;
        }
        else
        {
            // Weight tying — embed_tokens.weight is [vocab, d_model]; logits = hidden @ embed^T.
            // We reuse the embed tensor directly and apply the transpose at compute time.
            _projOutWeight = _embedTokens;
            _projOutIsTied = true;
        }
        if (_embedPositions.Shape.Rank != 2 || _embedPositions.Shape[0] < _cfg.MaxTextPositions || _embedPositions.Shape[1] != _cfg.HiddenSize)
        {
            throw new InvalidOperationException(
                $"{prefix}.embed_positions.weight has shape {_embedPositions.Shape}; expected at least [{_cfg.MaxTextPositions}, {_cfg.HiddenSize}].");
        }
        _embedTokensHost = (float*)_embedTokens.DataPointer;
        _embedPositionsHost = (float*)_embedPositions.DataPointer;

        List<Tensor> device = [];
        foreach (WhisperDecoderLayer layer in _layers) device.AddRange(layer.EnumerateWeights());
        device.Add(_finalLnWeight);
        device.Add(_finalLnBias);
        device.Add(_projOutWeight);
        _deviceWeights = [.. device];
        _weightsLoaded = true;
    }

    /// <summary>Fails fast when a <c>[vocab, d_model]</c> weight disagrees with the config: the logits loop reads exactly <see cref="WhisperConfig.VocabSize"/> rows, so an oversized config silently reads whatever tensor follows the embedding in the file as an extra logit (the English-only releases are one row short of the multilingual 51865).</summary>
    private void RequireVocabShape(Tensor weight, string name)
    {
        if (weight.Shape.Rank == 2 && weight.Shape[0] == _cfg.VocabSize && weight.Shape[1] == _cfg.HiddenSize) return;
        throw new InvalidOperationException(
            $"{name} has shape {weight.Shape} but the config expects [{_cfg.VocabSize}, {_cfg.HiddenSize}]. " +
            "The vocabulary size must match the checkpoint: 51865 multilingual, 51866 v3+, 51864 English-only (*.en).");
    }

    /// <summary>State carried across decode steps; the pipeline owns one of these for the duration of a single transcription.</summary>
    public sealed class DecodeState : IDisposable
    {
        private readonly WhisperConfig _cfg;
        /// <summary>Cross-attention K/V precomputed from the encoder output. Indexed by layer.</summary>
        public Tensor[] CrossKey { get; }
        public Tensor[] CrossValue { get; }
        /// <summary>Self-attention K/V cache for each layer, pre-allocated to <c>[1, num_heads, MaxTextPositions, head_dim]</c>; the valid prefix is <see cref="CurrentPos"/> entries.</summary>
        public Tensor[] SelfKey { get; }
        public Tensor[] SelfValue { get; }
        /// <summary>Number of tokens written into the self-attention cache so far.</summary>
        public int CurrentPos { get; set; }

        /// <summary>Stage attribution for each step's embed / layers / logits parts; null unless profiling.</summary>
        internal WhisperStageTimer? Timer { get; set; }

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

    /// <summary>Allocates fresh decoder state and precomputes cross-attention K/V from the encoder output; the pipeline calls this once per audio chunk.</summary>
    public DecodeState StartDecode(IBackend backend, Tensor encoderHidden)
    {
        ThrowIfDisposed();
        if (!_weightsLoaded) throw new InvalidOperationException("Call LoadWeights before StartDecode.");

        int batch = (int)encoderHidden.Shape[0];
        if (batch != 1) throw new NotSupportedException("WhisperDecoder currently supports batch=1 only.");
        int encSeq = (int)encoderHidden.Shape[1];

        // Idempotent, so an eviction since the last utterance is simply undone; small tensors (biases, norms) are
        // below the auto-promotion floor and would otherwise upload on every op of every step.
        backend.PreloadWeights(_deviceWeights);
        DecodeState state = new(_cfg, encSeq);
        try
        {
            for (int i = 0; i < _layers.Length; i++)
            {
                // The self-attention cache lives on the device for the whole decode; each step appends in place.
                backend.ResidentAllocateKv(state.SelfKey[i]);
                backend.ResidentAllocateKv(state.SelfValue[i]);
                _layers[i].PrecomputeCrossKv(backend, encoderHidden, state.CrossKey[i], state.CrossValue[i], encSeq);
            }
        }
        catch
        {
            state.Dispose();
            throw;
        }
        return state;
    }

    /// <summary>Runs the decoder for a single new token position; on first call pass the full prompt sequence (e.g. <c>[SOT, lang, transcribe, notimestamps]</c>) and on subsequent calls pass either the full sequence so far or just the newly appended token — the decoder uses positions <c>[CurrentPos, CurrentPos + len(tokenIds))</c> of the cache.</summary>
    /// <returns>Logits over the vocabulary <c>[1, vocabSize]</c> for the LAST token in <paramref name="tokenIds"/>.</returns>
    public Tensor DecodeStep(IBackend backend, ReadOnlySpan<int> tokenIds, DecodeState state)
    {
        ThrowIfDisposed();
        int newCount = tokenIds.Length;
        if (newCount == 0) throw new ArgumentException("tokenIds is empty", nameof(tokenIds));
        if (state.CurrentPos + newCount > _cfg.MaxTextPositions)
            throw new InvalidOperationException($"Decode would exceed MaxTextPositions ({_cfg.MaxTextPositions})");

        int d = _cfg.HiddenSize;
        int posStart = state.CurrentPos;
        int totalPos = posStart + newCount;

        // Embed the new tokens. hidden shape: [1, newCount, d_model].
        Tensor hidden = new(new TensorShape(1, newCount, d), DType.F32);
        EmbedAndAddPos(hidden, tokenIds, posStart, d);
        state.Timer?.Accumulate("step.embed");

        // Run through decoder layers; every layer sees the same causal mask for these positions.
        Tensor causalMask = WhisperDecoderLayer.BuildIncrementalCausalMask(posStart, newCount, totalPos);
        try
        {
            for (int i = 0; i < _layers.Length; i++)
            {
                Tensor next = _layers[i].Forward(backend, hidden,
                    state.SelfKey[i], state.SelfValue[i],
                    state.CrossKey[i], state.CrossValue[i],
                    posStart, newCount, causalMask);
                hidden.Dispose();
                hidden = next;
            }
        }
        finally
        {
            causalMask.Dispose();
        }
        state.Timer?.Accumulate("step.layers");

        // Final LN.
        Tensor normed = new(hidden.Shape, DType.F32);
        backend.LayerNorm(normed, hidden, _finalLnWeight!, _finalLnBias!, _cfg.LayerNormEps);
        hidden.Dispose();

        // Logits for the LAST token only (the only one we sample on): slice [1, 1, d] out of [1, newCount, d].
        Tensor lastHidden = normed;
        if (newCount > 1)
        {
            lastHidden = new Tensor(new TensorShape(1, 1, d), DType.F32);
            backend.SliceRows(lastHidden, normed, newCount - 1);
            normed.Dispose();
        }

        // Logits = lastHidden @ proj_out^T (the token table when tied). Full F32: this GEMM decides every greedy
        // token, and TF32 would round the 768-term dot products the host loop used to sum exactly.
        Tensor logits = new(new TensorShape(1, _cfg.VocabSize), DType.F32);
        WhisperOps.LinearFullPrecision(backend, logits, lastHidden, _projOutWeight!, bias: null);
        lastHidden.Dispose();
        state.Timer?.Accumulate("step.logits");

        state.CurrentPos += newCount;
        return logits;
    }

    private void EmbedAndAddPos(Tensor hidden, ReadOnlySpan<int> tokenIds, int posStart, int d)
    {
        float* hPtr = (float*)hidden.DataPointer;
        float* eTok = _embedTokensHost;
        float* ePos = _embedPositionsHost;
        for (int s = 0; s < tokenIds.Length; s++)
        {
            int tok = tokenIds[s];
            if ((uint)tok >= (uint)_cfg.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), tok, $"Token id is outside the vocabulary [0, {_cfg.VocabSize}).");
            int pos = posStart + s;
            int hOff = s * d;
            int tokOff = tok * d;
            int posOff = pos * d;
            for (int k = 0; k < d; k++) hPtr[hOff + k] = eTok[tokOff + k] + ePos[posOff + k];
        }
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

/// <summary>Single Whisper decoder block: pre-norm self-attention (causal, with KV cache) → pre-norm cross-attention (against precomputed encoder K/V) → pre-norm MLP; weight names match the HF transformers Whisper layout (<c>self_attn / encoder_attn / fc1 / fc2</c>).</summary>
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

    /// <summary>Precomputes cross-attention K and V from the encoder hidden state, run once per audio chunk at decode start; the result is held in <see cref="WhisperDecoder.DecodeState"/>.</summary>
    public void PrecomputeCrossKv(IBackend backend, Tensor encHidden, Tensor outK, Tensor outV, int encSeqLen)
    {
        int d = _cfg.HiddenSize;
        // Project K, V from encoder hidden [1, encSeq, d] → [1, encSeq, d].
        Tensor k = WhisperOps.ProjectLinear(backend, encHidden, _crossKW!, bias: null, 1, encSeqLen, d, d);
        Tensor v = WhisperOps.ProjectLinear(backend, encHidden, _crossVW!, _crossVB, 1, encSeqLen, d, d);
        // Permute into [1, H, encSeq, headDim] directly into the state tensors, which stay on the device.
        backend.Permute0213(outK, k, encSeqLen, _cfg.NumHeads, _cfg.HeadDim);
        backend.Permute0213(outV, v, encSeqLen, _cfg.NumHeads, _cfg.HeadDim);
        k.Dispose(); v.Dispose();
    }

    /// <summary>Forward pass for a chunk of <paramref name="newCount"/> new positions starting at absolute position <paramref name="posStart"/>; updates the self-attn KV cache slots [posStart, posStart+newCount). <paramref name="causalMask"/> is <see cref="BuildIncrementalCausalMask"/> for these positions.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden,
        Tensor selfK, Tensor selfV, Tensor crossK, Tensor crossV,
        int posStart, int newCount, Tensor causalMask)
    {
        int d = _cfg.HiddenSize;
        int totalPos = posStart + newCount;
        int heads = _cfg.NumHeads, headDim = _cfg.HeadDim;
        TensorShape inShape = new(1, newCount, d);

        // --- Self-attention sub-block (causal, with KV cache append) ---
        Tensor normed = new(inShape, DType.F32);
        backend.LayerNorm(normed, hidden, _selfLnW!, _selfLnB!, _cfg.LayerNormEps);
        Tensor q = WhisperOps.ProjectLinear(backend, normed, _selfQW!, _selfQB, 1, newCount, d, d);
        Tensor k = WhisperOps.ProjectLinear(backend, normed, _selfKW!, bias: null, 1, newCount, d, d);
        Tensor v = WhisperOps.ProjectLinear(backend, normed, _selfVW!, _selfVB, 1, newCount, d, d);
        normed.Dispose();

        TensorShape mhNew = new(1, heads, newCount, headDim);
        Tensor qMh = new(mhNew, DType.F32);
        backend.Permute0213(qMh, q, newCount, heads, headDim);
        q.Dispose();

        // Write new K, V into the cache slots [posStart..posStart+newCount).
        Tensor kMhNew = new(mhNew, DType.F32);
        Tensor vMhNew = new(mhNew, DType.F32);
        backend.Permute0213(kMhNew, k, newCount, heads, headDim);
        backend.Permute0213(vMhNew, v, newCount, heads, headDim);
        k.Dispose(); v.Dispose();
        backend.KvCacheAppend(selfK, kMhNew, posStart);
        backend.KvCacheAppend(selfV, vMhNew, posStart);
        kMhNew.Dispose(); vMhNew.Dispose();

        // Contiguous [1, H, totalPos, D] copies of the valid prefix: attention reads exactly the keys and values it
        // always did, through the same kernel.
        Tensor kCached = new(new TensorShape(1, heads, totalPos, headDim), DType.F32);
        Tensor vCached = new(new TensorShape(1, heads, totalPos, headDim), DType.F32);
        backend.SliceTimeRange(kCached, selfK, 0, totalPos);
        backend.SliceTimeRange(vCached, selfV, 0, totalPos);

        float scale = 1f / MathF.Sqrt(headDim);
        Tensor attnOut = new(mhNew, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kCached, vCached, causalMask, scale);
        qMh.Dispose(); kCached.Dispose(); vCached.Dispose();

        Tensor merged = new(inShape, DType.F32);
        backend.Permute0213(merged, attnOut, heads, newCount, headDim);
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
        backend.Permute0213(crossQMh, crossQ, newCount, heads, headDim);
        crossQ.Dispose();

        Tensor crossAttn = new(mhNew, DType.F32);
        backend.ScaledDotProductAttention(crossAttn, crossQMh, crossK, crossV, mask: null, scale);
        crossQMh.Dispose();
        Tensor crossMerged = new(inShape, DType.F32);
        backend.Permute0213(crossMerged, crossAttn, heads, newCount, headDim);
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

    /// <summary>Builds the mask for new-token self-attention, shape [1, 1, newCount, totalPos]: position s in the new chunk (= absolute position posStart+s) can see cache positions [0..posStart+s], everything beyond is -inf.</summary>
    internal static Tensor BuildIncrementalCausalMask(int posStart, int newCount, int totalPos)
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

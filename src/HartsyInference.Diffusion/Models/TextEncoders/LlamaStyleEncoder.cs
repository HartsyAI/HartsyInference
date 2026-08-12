using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.TextEncoders;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Llama-family decoder transformer used as a text encoder for diffusion conditioning. Supports GQA, RMSNorm, RoPE (theta-configurable), SwiGLU MLP, and optional per-head Q/K RMSNorm (Qwen3). Runs as an encoder: single forward pass returning <c>last_hidden_state</c> as <c>[B, seqLen, hiddenSize]</c>; causal attention mask matches how the model was trained.</summary>
public sealed unsafe class LlamaStyleEncoder : IDisposable
{
    private readonly LlamaStyleEncoderConfig _config;
    private readonly LlamaBlock[] _blocks;

    // Token embedding [vocab, hidden]
    private Tensor? _embedWeight;
    // Final RMSNorm scale [hidden]
    private Tensor? _finalNormWeight;

    // RoPE precomputed cos/sin for max_position_embeddings × headDim/2.
    // Lazy-built by EnsureRopeTable so we only allocate a table sized to the longest prompt actually used.
    private float[]? _ropeCos;
    private float[]? _ropeSin;
    // Gemma-3-style second rope table for LOCAL (sliding-window) layers, built from LocalRopeTheta.
    // Null unless the config sets LocalRopeTheta > 0.
    private float[]? _ropeCosLocal;
    private float[]? _ropeSinLocal;
    private int _ropeBuiltForMaxLen;

    /// <summary>One request's causal mask and full-width RoPE tables. These are ordinary activation tensors, not
    /// model weights: creating them through a backend copy gives CUDA one resident H2D upload without the synchronous
    /// teardown semantics of <see cref="IBackend.FreeWeights"/>.</summary>
    private sealed class ForwardConstants : IDisposable
    {
        private readonly Tensor[] _owned;
        private int _disposed;

        public ForwardConstants(Tensor causalMask, Tensor globalCos, Tensor globalSin, Tensor? localCos, Tensor? localSin)
        {
            CausalMask = causalMask;
            GlobalCos = globalCos;
            GlobalSin = globalSin;
            LocalCos = localCos;
            LocalSin = localSin;
            _owned = localCos is null
                ? [causalMask, globalCos, globalSin]
                : [causalMask, globalCos, globalSin, localCos, localSin!];
        }

        public Tensor CausalMask { get; }
        public Tensor GlobalCos { get; }
        public Tensor GlobalSin { get; }
        public Tensor? LocalCos { get; }
        public Tensor? LocalSin { get; }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            foreach (Tensor tensor in _owned)
            {
                try { tensor.Dispose(); }
                catch (Exception error)
                {
                    // Request-constant cleanup must not replace a successful encoder result or mask the
                    // exception that caused an encode to unwind. Tensor.Dispose already attempts all of its
                    // bindings; record the backend failure and continue releasing the remaining constants.
                    Logs.Error("Failed to release a Llama encoder request constant.", error);
                }
            }
        }
    }

    private int _disposed;

    public LlamaStyleEncoder(LlamaStyleEncoderConfig config)
    {
        _config = config;
        _blocks = new LlamaBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
            _blocks[i] = new LlamaBlock(config);
    }

    /// <summary>Number of transformer blocks. Useful for callers that want to request a specific HF-indexed hidden state via <see cref="EncodeMultiLayer"/> (e.g., Z-Image needs <c>NumLayers - 1</c> for diffusers' <c>hidden_states[-2]</c>).</summary>
    public int NumLayers => _config.NumLayers;

    // Diagnostic (HARTSY_TE_PROBE=1): per-layer absmax of the hidden stream — forces a host sync per
    // layer, so leave off outside debugging sessions.
    private static readonly bool TeProbe = Environment.GetEnvironmentVariable("HARTSY_TE_PROBE") == "1";

    private static unsafe void ProbeAbsmax(string label, Tensor t)
    {
        float* p = (float*)t.DataPointer;
        float mx = 0f;
        long n = t.ElementCount;
        for (long e = 0; e < n; e++) { float a = MathF.Abs(p[e]); if (a > mx) mx = a; }
        Logs.Debug($"[TE probe] {label}: absmax={mx:F3}");
    }

    /// <summary>Loads all weights from a HuggingFace-style key dict (keys like
    /// <c>model.layers.{i}.self_attn.q_proj.weight</c>). Embeddings and norm scales are materialized as F32 to
    /// match the backend gather/RMSNorm contracts; projection matrices retain their checkpoint dtype.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        // Normalize any ComfyUI quantization companions (fold fp8 weight_scale into Fp8ScaleFactor, drop
        // comfy_quant blobs, fail clearly on still-unsupported U8 quants). No-op for plain BF16/F16/F32
        // checkpoints. Without this, fp8_scaled encoders silently dropped their per-tensor scale and
        // U8-packed quants died deep in a GEMM with an opaque "GPU cast from U8 to F32" error.
        weights = TextEncoderQuantNormalizer.Normalize(weights);

        // Some checkpoints ship the decoder under a `model.` wrapper (full Causal-LM exports), others ship it
        // bare (e.g. Qwen3-Embedding-0.6B: `embed_tokens.weight`, `layers.N.*`, `norm.weight`). Auto-detect.
        string prefix = weights.ContainsKey("model.embed_tokens.weight") ? "model." : "";

        Tensor rawEmbed = weights[$"{prefix}embed_tokens.weight"];
        _embedWeight = CastToF32IfNeeded(rawEmbed);

        if (_config.HasFinalNorm)
        {
            Tensor rawFinalNorm = weights[$"{prefix}norm.weight"];
            _finalNormWeight = CastToF32IfNeeded(rawFinalNorm);
            if (_config.RmsNormScalePlusOne) AddOneInPlace(_finalNormWeight);
        }

        for (int i = 0; i < _config.NumLayers; i++)
            _blocks[i].LoadWeights(weights, $"{prefix}layers.{i}");

        Logs.Verbose($"LlamaStyleEncoder loaded: {_config.NumLayers} layers, hidden={_config.HiddenSize}, " +
                  $"q_heads={_config.NumQueryHeads}, kv_heads={_config.NumKvHeads}, head_dim={_config.HeadDim}");
    }

    /// <summary>Enumerates all weight tensors, used by GPU-preload paths.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embedWeight is not null) yield return _embedWeight;
        if (_finalNormWeight is not null) yield return _finalNormWeight;
        for (int i = 0; i < _blocks.Length; i++)
            foreach (Tensor w in _blocks[i].EnumerateWeights())
                yield return w;
    }

    /// <summary>Encodes a batch of token id arrays (all rows must have the same length). Returns last_hidden_state shaped <c>[B, seqLen, hiddenSize]</c> as F32. Causal mask applied.</summary>
    public Tensor Encode(IBackend backend, int[][] tokenIds)
    {
        ThrowIfDisposed();

        (int batch, int seqLen) = ValidateTokenBatch(tokenIds);
        if (seqLen > _config.MaxPositionEmbeddings)
            throw new InvalidOperationException(
                $"Prompt length {seqLen} exceeds MaxPositionEmbeddings ({_config.MaxPositionEmbeddings}).");

        EnsureRopeTable(seqLen);

        Tensor? hidden = null;
        using ForwardConstants constants = PrepareForwardConstants(
            backend, batch, seqLen, _ropeCos!, _ropeSin!, _ropeCosLocal, _ropeSinLocal);
        try
        {
            // 1. Gather token embeddings on the selected backend. This used to be a scalar host loop whose
            // result had to be uploaded before the first RMSNorm, despite the embedding table already being
            // resident on CUDA.
            hidden = EmbeddingLookup(backend, tokenIds, batch, seqLen);

            // 2. Layer loop — each block reads `hidden`, allocates a new tensor, returns it. Old hidden disposed.
            for (int i = 0; i < _config.NumLayers; i++)
            {
                (Tensor rcos, Tensor rsin) = RopeForLayer(i, constants);
                Tensor next = _blocks[i].Forward(backend, hidden, constants.CausalMask, rcos, rsin, seqLen);
                Tensor previous = hidden;
                hidden = next;
                DisposeBestEffort(previous, "previous layer activation");
            }

            // 3. Final RMSNorm (skipped for checkpoints that don't ship one — e.g. Mistral-Small-3
            // distilled for Flux.2 Dev. In that case return the last block's raw output.
            if (_config.HasFinalNorm)
            {
                Tensor output = new(new TensorShape(batch, seqLen, _config.HiddenSize), DType.F32);
                try
                {
                    backend.RmsNorm(output, hidden, _finalNormWeight!, _config.RmsNormEps);
                }
                catch
                {
                    DisposeBestEffort(output, "final norm output after failure");
                    throw;
                }
                Tensor previous = hidden;
                hidden = null;
                DisposeBestEffort(previous, "pre-final-norm activation");
                return output;
            }

            Tensor result = hidden!;
            hidden = null;
            return result;
        }
        finally
        {
            DisposeBestEffort(hidden, "hidden activation");
        }
    }

    /// <summary>Looks up token embeddings <c>[1, seqLen, hidden]</c> (F32). Exposed so multimodal callers (Qwen3-VL)
    /// can splice merged vision tokens into the sequence before running <see cref="EncodeEmbedsMrope"/>.</summary>
    public Tensor LookupEmbeddings(int[] tokenIds)
    {
        ThrowIfDisposed();
        ValidateTokenBatch([tokenIds]);
        if (tokenIds.Length > _config.MaxPositionEmbeddings)
            throw new ArgumentOutOfRangeException(
                nameof(tokenIds), tokenIds.Length, $"Prompt exceeds maximum position {_config.MaxPositionEmbeddings}.");
        return EmbeddingLookupHost([tokenIds], 1, tokenIds.Length);
    }

    /// <summary>Runs the decoder over caller-supplied <c>inputsEmbeds [1, seqLen, hidden]</c> with a precomputed
    /// per-token RoPE <c>(cos, sin)</c> table (sized <c>seqLen · headDim/2</c>) — this is how Qwen3-VL's
    /// <b>interleaved M-RoPE</b> is driven (the 3D position ids collapse into one split-half table identical in shape to
    /// the 1D table, so every <see cref="LlamaBlock"/> applies it unchanged). Optional Qwen3-VL <b>deepstack</b>
    /// injection adds <paramref name="deepstackFeatures"/>[i] to the visual-token positions after decoder layer i (for
    /// the first <c>deepstackFeatures.Count</c> layers). Returns the final hidden state (post <c>model.norm</c>) as
    /// <c>[1, seqLen, hidden]</c>.</summary>
    /// <param name="visualMask">Length-<c>seqLen</c> mask, true at image-token positions (the order matches the rows of each deepstack feature tensor).</param>
    public Tensor EncodeEmbedsMrope(IBackend backend, Tensor inputsEmbeds, float[] ropeCos, float[] ropeSin,
        IReadOnlyList<Tensor>? deepstackFeatures = null, bool[]? visualMask = null)
    {
        ThrowIfDisposed();
        if (inputsEmbeds.DType != DType.F32 || inputsEmbeds.Shape.Rank != 3 || inputsEmbeds.Shape[0] != 1
            || inputsEmbeds.Shape[2] != _config.HiddenSize)
        {
            throw new ArgumentException(
                $"inputsEmbeds must be F32 [1, seqLen, {_config.HiddenSize}], got {inputsEmbeds.DType} {inputsEmbeds.Shape}.",
                nameof(inputsEmbeds));
        }
        int seqLen = (int)inputsEmbeds.Shape[1];
        int H = _config.HiddenSize;
        if (seqLen <= 0 || seqLen > _config.MaxPositionEmbeddings)
            throw new ArgumentOutOfRangeException(
                nameof(inputsEmbeds), seqLen, $"Sequence length must be in [1,{_config.MaxPositionEmbeddings}].");
        ValidateAttentionGeometry();
        if (deepstackFeatures is not null && deepstackFeatures.Count > _config.NumLayers)
            throw new ArgumentException(
                $"Deepstack feature count {deepstackFeatures.Count} exceeds decoder depth {_config.NumLayers}.",
                nameof(deepstackFeatures));

        (int[] visualRows, float[] visualScales) = ValidateDeepstack(deepstackFeatures, visualMask, seqLen, H);
        Tensor? hidden = null;
        using ForwardConstants constants = PrepareForwardConstants(backend, 1, seqLen, ropeCos, ropeSin);
        try
        {
            hidden = new Tensor(new TensorShape(1, seqLen, H), DType.F32);
            backend.SliceRowsGeneric(hidden, inputsEmbeds, rowOffset: 0);

            for (int i = 0; i < _config.NumLayers; i++)
            {
                Tensor next = _blocks[i].Forward(
                    backend, hidden, constants.CausalMask, constants.GlobalCos, constants.GlobalSin, seqLen);
                Tensor previous = hidden;
                hidden = next;
                DisposeBestEffort(previous, "previous M-RoPE layer activation");
                if (deepstackFeatures is not null && i < deepstackFeatures.Count)
                    backend.ScatterAddWeightedRows(hidden, deepstackFeatures[i], visualRows, visualScales);
            }

            if (_config.HasFinalNorm)
            {
                Tensor output = new(new TensorShape(1, seqLen, H), DType.F32);
                try { backend.RmsNorm(output, hidden, _finalNormWeight!, _config.RmsNormEps); }
                catch { DisposeBestEffort(output, "M-RoPE final norm output after failure"); throw; }
                Tensor previous = hidden;
                hidden = null;
                DisposeBestEffort(previous, "M-RoPE pre-final-norm activation");
                return output;
            }

            Tensor result = hidden!;
            hidden = null;
            return result;
        }
        finally
        {
            DisposeBestEffort(hidden, "M-RoPE hidden activation");
        }
    }

    private static (int[] Rows, float[] Scales) ValidateDeepstack(
        IReadOnlyList<Tensor>? features, bool[]? visualMask, int seqLen, int hiddenSize)
    {
        if (features is null || features.Count == 0)
        {
            if (visualMask is not null && visualMask.Length != seqLen)
                throw new ArgumentException($"visualMask length {visualMask.Length} does not match sequence {seqLen}.", nameof(visualMask));
            return ([], []);
        }

        if (visualMask is null || visualMask.Length != seqLen)
            throw new ArgumentException(
                $"Deepstack features require a visualMask of length {seqLen}; got {visualMask?.Length ?? 0}.",
                nameof(visualMask));

        int[] rows = Enumerable.Range(0, seqLen).Where(i => visualMask[i]).ToArray();
        if (rows.Length == 0)
            throw new ArgumentException("Deepstack features were supplied, but visualMask selects no rows.", nameof(visualMask));
        foreach (Tensor feature in features)
        {
            if (feature.DType != DType.F32 || feature.Shape.Rank < 2
                || feature.Shape[feature.Shape.Rank - 1] != hiddenSize
                || feature.ElementCount != (long)rows.Length * hiddenSize)
            {
                throw new ArgumentException(
                    $"Each deepstack feature must be F32 [{rows.Length},{hiddenSize}], got {feature.DType} {feature.Shape}.",
                    nameof(features));
            }
        }

        return (rows, Enumerable.Repeat(1f, rows.Length).ToArray());
    }

    /// <summary>Encodes a prompt and concatenates hidden states from selected intermediate layers along the feature axis. Used by Flux.2 Klein (layers 9, 18, 27 → 7680 dim for Qwen3-4B). HuggingFace indexing: <c>k=0</c> is the embedding output (pre-layer-0); <c>k=1..N</c> is post-layer-(k−1). Final RMSNorm is NOT applied to intermediate outputs (matches HF — only the last hidden state passes through <c>model.norm</c>).</summary>
    /// <param name="layerIndices">Layer indices in HuggingFace convention: 0 = embeddings, k = post-layer-(k-1). Must be sorted ascending and within [0, NumLayers].</param>
    /// <param name="interleavedLayout">Channel ordering of the concatenated taps. <c>false</c> (default) = <b>tap-major</b>
    /// <c>[layer0(H) | layer1(H) | …]</c> (diffusers' <c>permute(0,2,1,3).reshape(B,S,N*H)</c>, used by Flux.2 Klein).
    /// <c>true</c> = <b>hidden-major</b> <c>c = h*K + layerSlot</c> (upstream Ideogram 4's <c>permute(stack,(1,2,3,0)).reshape</c>
    /// and ComfyUI's <c>permute(0,2,3,1).reshape(B,S,H*N)</c>) — the 13 tap values of hidden-dim 0, then of hidden-dim 1, …
    /// Ideogram 4's <c>llm_cond_norm</c>/<c>llm_cond_proj</c> were trained on this order, so feeding tap-major scrambles
    /// every input channel of the projection. For a single tap (<c>K=1</c>) the two layouts are identical.</param>
    /// <returns>F32 tensor of shape <c>[batch, seqLen, layerIndices.Length × hiddenSize]</c>.</returns>
    public Tensor EncodeMultiLayer(IBackend backend, int[][] tokenIds, int[] layerIndices, bool interleavedLayout = false)
    {
        ThrowIfDisposed();
        if (layerIndices is null || layerIndices.Length == 0)
            throw new ArgumentException("Must request at least one layer.", nameof(layerIndices));
        for (int i = 0; i < layerIndices.Length; i++)
        {
            if (layerIndices[i] < 0 || layerIndices[i] > _config.NumLayers)
                throw new ArgumentOutOfRangeException(nameof(layerIndices),
                    $"Layer index {layerIndices[i]} out of range [0, {_config.NumLayers}].");
            if (i > 0 && layerIndices[i] <= layerIndices[i - 1])
                throw new ArgumentException("Layer indices must be strictly ascending.", nameof(layerIndices));
        }

        (int batch, int seqLen) = ValidateTokenBatch(tokenIds);
        int H = _config.HiddenSize;
        int K = layerIndices.Length;
        if (seqLen > _config.MaxPositionEmbeddings)
            throw new InvalidOperationException(
                $"Prompt length {seqLen} exceeds MaxPositionEmbeddings ({_config.MaxPositionEmbeddings}).");

        EnsureRopeTable(seqLen);

        Tensor? hidden = null;
        bool hiddenIsCaptured = false;
        List<Tensor> captures = new(K);
        using ForwardConstants constants = PrepareForwardConstants(
            backend, batch, seqLen, _ropeCos!, _ropeSin!, _ropeCosLocal, _ropeSinLocal);
        try
        {
            hidden = EmbeddingLookup(backend, tokenIds, batch, seqLen);
            if (TeProbe) ProbeAbsmax("embed", hidden);
            int requestIdx = 0;

            // Retain the actual activation object instead of D2H-copying it into a host output. The next block
            // only reads this tensor, so a requested tap can remain alive until one device-side pack at the end.
            if (layerIndices[requestIdx] == 0)
            {
                captures.Add(hidden);
                hiddenIsCaptured = true;
                requestIdx++;
            }

            for (int i = 0; i < _config.NumLayers && requestIdx < K; i++)
            {
                (Tensor rcos, Tensor rsin) = RopeForLayer(i, constants);
                Tensor next = _blocks[i].Forward(backend, hidden, constants.CausalMask, rcos, rsin, seqLen);
                Tensor previous = hidden;
                hidden = next;
                if (!hiddenIsCaptured) DisposeBestEffort(previous, "previous multi-layer activation");
                hiddenIsCaptured = false;
                if (TeProbe) ProbeAbsmax($"layer {i + 1}", hidden);

                // Block i produces hidden_states[i+1] in HF terms. Strictly ascending indices mean at most
                // one capture per state.
                if (layerIndices[requestIdx] == i + 1)
                {
                    captures.Add(hidden);
                    hiddenIsCaptured = true;
                    requestIdx++;
                }
            }

            if (captures.Count != K)
                throw new InvalidOperationException($"Captured {captures.Count} of {K} requested Llama hidden states.");

            if (K == 1)
            {
                Tensor only = captures[0];
                captures.Clear();
                hidden = null;
                hiddenIsCaptured = false;
                return only;
            }

            TensorShape outputShape = new(batch, seqLen, checked(K * H));
            Tensor tapMajor = new(outputShape, DType.F32);
            try
            {
                backend.Concat(tapMajor, captures.ToArray(), dim: 2);
                if (!interleavedLayout)
                {
                    hidden = null;
                    hiddenIsCaptured = false;
                    return tapMajor;
                }

                Tensor output = new(outputShape, DType.F32);
                try
                {
                    // Physical per-token layout is [K,H]; transpose to [H,K] for Ideogram's hidden-major order.
                    backend.Transpose2D(output, tapMajor, K, H);
                }
                catch
                {
                    DisposeBestEffort(output, "interleaved tap output");
                    throw;
                }
                DisposeBestEffort(tapMajor, "tap-major staging output");
                hidden = null;
                hiddenIsCaptured = false;
                return output;
            }
            catch
            {
                DisposeBestEffort(tapMajor, "tap-major output after pack failure");
                throw;
            }
        }
        finally
        {
            if (!hiddenIsCaptured) DisposeBestEffort(hidden, "multi-layer hidden activation");
            foreach (Tensor capture in captures)
                DisposeBestEffort(capture, "captured hidden activation");
        }
    }

    private static void DisposeBestEffort(Tensor? tensor, string description)
    {
        if (tensor is null) return;
        try { tensor.Dispose(); }
        catch (Exception error) { Logs.Error($"Failed to release Llama encoder {description}.", error); }
    }

    private Tensor EmbeddingLookup(IBackend backend, int[][] tokenIds, int batch, int seqLen)
    {
        int rowCount = checked(batch * seqLen);
        int[] rows = new int[rowCount];
        int rowIndex = 0;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
                rows[rowIndex++] = tokenIds[b][s];

        TensorShape shape = new(batch, seqLen, _config.HiddenSize);
        Tensor gathered = new(shape, DType.F32);
        try
        {
            // Uploading a Qwen/Gemma vocabulary table solely to collect a short prompt can cost 1.5–3.2 GB.
            // Use the kernel only when the recipe/streamer has already made that table resident; otherwise the
            // old host gather is both faster and dramatically cheaper in peak VRAM.
            if (!backend.TryGatherRowsResident(gathered, _embedWeight!, rows))
            {
                DisposeBestEffort(gathered, "unused resident-gather output");
                return EmbeddingLookupHost(tokenIds, batch, seqLen);
            }
            if (_config.EmbeddingScale == 1.0f)
                return gathered;

            Tensor scaled = new(shape, DType.F32);
            try
            {
                backend.Scale(scaled, gathered, _config.EmbeddingScale);
            }
            catch
            {
                DisposeBestEffort(scaled, "scaled embedding output after failure");
                throw;
            }
            DisposeBestEffort(gathered, "unscaled gathered embedding");
            return scaled;
        }
        catch
        {
            DisposeBestEffort(gathered, "gathered embedding after failure");
            throw;
        }
    }

    /// <summary>Host lookup retained for multimodal/audio callers that immediately splice or slice the returned
    /// buffer on the CPU. Moving those callers to the backend requires porting that adjacent host operation too;
    /// uploading here only to read straight back would be strictly worse.</summary>
    private Tensor EmbeddingLookupHost(int[][] tokenIds, int batch, int seqLen)
    {
        TensorShape shape = new TensorShape(batch, seqLen, _config.HiddenSize);
        Tensor output = new Tensor(shape, DType.F32);

        float* embedPtr = (float*)_embedWeight!.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int H = _config.HiddenSize;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int tokenId = tokenIds[b][s];
                if ((uint)tokenId >= (uint)_config.VocabSize)
                    throw new ArgumentOutOfRangeException(nameof(tokenIds),
                        $"Token id {tokenId} at [{b},{s}] out of vocab (size {_config.VocabSize}).");

                long src = (long)tokenId * H;
                long dst = ((long)b * seqLen + s) * H;
                Buffer.MemoryCopy(embedPtr + src, outPtr + dst, H * sizeof(float), H * sizeof(float));
            }
        }

        // Gemma-family "normalizer": scale embeddings by sqrt(HiddenSize). No-op for Llama/Qwen/Mistral
        // (EmbeddingScale defaults to 1.0).
        if (_config.EmbeddingScale != 1.0f)
        {
            float scale = _config.EmbeddingScale;
            long total = (long)batch * seqLen * H;
            for (long i = 0; i < total; i++) outPtr[i] *= scale;
        }
        return output;
    }

    private (int Batch, int Sequence) ValidateTokenBatch(int[][] tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        if (tokenIds.Length == 0)
            throw new ArgumentException("Token batch must contain at least one row.", nameof(tokenIds));
        if (tokenIds[0] is null || tokenIds[0].Length == 0)
            throw new ArgumentException("Token rows must be non-null and non-empty.", nameof(tokenIds));

        int seqLen = tokenIds[0].Length;
        for (int b = 0; b < tokenIds.Length; b++)
        {
            int[]? row = tokenIds[b];
            if (row is null || row.Length != seqLen)
                throw new ArgumentException(
                    $"Token batch must be rectangular with sequence length {seqLen}; row {b} has length {row?.Length ?? 0}.",
                    nameof(tokenIds));
            for (int s = 0; s < row.Length; s++)
            {
                if ((uint)row[s] >= (uint)_config.VocabSize)
                    throw new ArgumentOutOfRangeException(
                        nameof(tokenIds), row[s], $"Token id at [{b},{s}] is outside vocabulary size {_config.VocabSize}.");
            }
        }

        ValidateAttentionGeometry();

        return (tokenIds.Length, seqLen);
    }

    private void ValidateAttentionGeometry()
    {
        if (_config.NumKvHeads <= 0 || _config.NumQueryHeads <= 0
            || _config.NumQueryHeads % _config.NumKvHeads != 0)
        {
            throw new InvalidOperationException(
                $"Invalid grouped-query geometry Hq={_config.NumQueryHeads}, Hkv={_config.NumKvHeads}; Hq must be divisible by positive Hkv.");
        }
        if (_config.HeadDim <= 0 || (_config.HeadDim & 1) != 0)
            throw new InvalidOperationException($"Split-half RoPE requires a positive even head dimension; got {_config.HeadDim}.");
    }

    private static Tensor BuildCausalMask(int seqLen)
    {
        // Mask shape [seqLen, seqLen] (broadcast across batch, heads inside SDPA).
        // Use a large negative number rather than -inf to avoid NaN in mixed-precision softmax.
        const float negInf = -1e30f;
        TensorShape shape = new TensorShape(seqLen, seqLen);
        Tensor mask = new Tensor(shape, DType.F32);
        float* p = (float*)mask.DataPointer;
        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j < seqLen; j++)
                p[i * seqLen + j] = j > i ? negInf : 0f;
        }
        return mask;
    }

    /// <summary>Copies a small request constant through the backend once so every layer reuses its device-resident
    /// activation. A reshape/view is deliberately not used: views have a different tensor identity and therefore
    /// cannot find the producer's activation-cache entry.</summary>
    private static Tensor MakeResidentCopy(IBackend backend, Tensor host)
    {
        Tensor resident = new(host.Shape, host.DType);
        try
        {
            backend.SliceRowsGeneric(resident, host, rowOffset: 0);
            return resident;
        }
        catch
        {
            DisposeBestEffort(resident, "resident request constant after copy failure");
            throw;
        }
        finally
        {
            DisposeBestEffort(host, "host request constant");
        }
    }

    /// <summary>Expands the cached split-half table <c>[S,D/2]</c> to the backend RoPE contract
    /// <c>[B,S,D]</c>, duplicating each frequency into both vector halves and each batch row.</summary>
    private static Tensor BuildFullWidthRopeTable(float[] halfTable, int batch, int seqLen, int headDim)
    {
        ArgumentNullException.ThrowIfNull(halfTable);
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch), "Batch must be positive.");
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen), "Sequence length must be positive.");
        if (headDim <= 0 || (headDim & 1) != 0)
            throw new ArgumentOutOfRangeException(nameof(headDim), "Split-half RoPE requires a positive even head dimension.");

        int half = headDim / 2;
        int required = checked(seqLen * half);
        if (halfTable.Length < required)
            throw new ArgumentException(
                $"RoPE table has {halfTable.Length} values, but sequence {seqLen} and head dimension {headDim} require at least {required}.",
                nameof(halfTable));

        Tensor full = new(new TensorShape(batch, seqLen, headDim), DType.F32);
        float* output = (float*)full.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int sourceBase = s * half;
                long destinationBase = ((long)b * seqLen + s) * headDim;
                for (int i = 0; i < half; i++)
                {
                    float value = halfTable[sourceBase + i];
                    output[destinationBase + i] = value;
                    output[destinationBase + half + i] = value;
                }
            }
        }
        return full;
    }

    private ForwardConstants PrepareForwardConstants(
        IBackend backend,
        int batch,
        int seqLen,
        float[] globalCos,
        float[] globalSin,
        float[]? localCos = null,
        float[]? localSin = null)
    {
        Tensor? mask = null;
        Tensor? gCos = null;
        Tensor? gSin = null;
        Tensor? lCos = null;
        Tensor? lSin = null;
        try
        {
            mask = MakeResidentCopy(backend, BuildCausalMask(seqLen));
            gCos = MakeResidentCopy(backend, BuildFullWidthRopeTable(globalCos, batch, seqLen, _config.HeadDim));
            gSin = MakeResidentCopy(backend, BuildFullWidthRopeTable(globalSin, batch, seqLen, _config.HeadDim));
            if (localCos is not null || localSin is not null)
            {
                if (localCos is null || localSin is null)
                    throw new ArgumentException("Local RoPE cosine and sine tables must either both be supplied or both be absent.");
                lCos = MakeResidentCopy(backend, BuildFullWidthRopeTable(localCos, batch, seqLen, _config.HeadDim));
                lSin = MakeResidentCopy(backend, BuildFullWidthRopeTable(localSin, batch, seqLen, _config.HeadDim));
            }
            return new ForwardConstants(mask, gCos, gSin, lCos, lSin);
        }
        catch
        {
            // Construction is transactional: a failed later upload must not strand the successfully-created prefix.
            Tensor?[] created = [mask, gCos, gSin, lCos, lSin];
            foreach (Tensor? tensor in created)
            {
                try { tensor?.Dispose(); }
                catch (Exception error) { Logs.Error("Failed to release a partially prepared Llama encoder constant.", error); }
            }
            throw;
        }
    }

    private (Tensor Cos, Tensor Sin) RopeForLayer(int layerIndex, ForwardConstants constants)
    {
        if (constants.LocalCos is not null
            && _config.LayerAttentionTypes is { } types && layerIndex < types.Length
            && !string.Equals(types[layerIndex], "full", StringComparison.Ordinal)
            && !string.Equals(types[layerIndex], "full_attention", StringComparison.Ordinal))
        {
            return (constants.LocalCos, constants.LocalSin!);
        }
        return (constants.GlobalCos, constants.GlobalSin);
    }

    /// <summary>Precomputes RoPE cos/sin tables once for any seq_len up to <c>maxLenSeen</c>. Cache grows on demand.</summary>
    private void EnsureRopeTable(int seqLen)
    {
        if (_ropeCos is not null && _ropeBuiltForMaxLen >= seqLen)
            return;

        int halfDim = _config.HeadDim / 2;
        int targetLen = Math.Max(seqLen, _ropeBuiltForMaxLen);
        // Round up to a power of 2 for fewer rebuilds when prompts grow incrementally.
        int rounded = 1;
        while (rounded < targetLen) rounded <<= 1;
        targetLen = Math.Min(rounded, _config.MaxPositionEmbeddings);

        (_ropeCos, _ropeSin) = BuildRopeTable(targetLen, halfDim, _config.RopeTheta);
        // Gemma 3 interleaves LOCAL (sliding) layers that use a smaller rope base frequency than the
        // GLOBAL/full layers. Build the second table only when the config asks for it.
        if (_config.LocalRopeTheta > 0f)
            (_ropeCosLocal, _ropeSinLocal) = BuildRopeTable(targetLen, halfDim, _config.LocalRopeTheta);

        _ropeBuiltForMaxLen = targetLen;
    }

    /// <summary>Precomputes a RoPE cos/sin table for a given base frequency. freqs[k] = 1 / theta^(2k/headDim).</summary>
    private (float[] Cos, float[] Sin) BuildRopeTable(int targetLen, int halfDim, float theta)
    {
        float[] cos = new float[targetLen * halfDim];
        float[] sin = new float[targetLen * halfDim];
        for (int p = 0; p < targetLen; p++)
        {
            for (int k = 0; k < halfDim; k++)
            {
                double freq = 1.0 / Math.Pow(theta, (double)(2 * k) / _config.HeadDim);
                double angle = p * freq;
                cos[p * halfDim + k] = (float)Math.Cos(angle);
                sin[p * halfDim + k] = (float)Math.Sin(angle);
            }
        }
        return (cos, sin);
    }

    private static Tensor CastToF32IfNeeded(Tensor t) =>
        t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    /// <summary>Pre-adds 1.0 to every element of an F32 RMSNorm scale tensor in place. Gemma 2 stores
    /// scales as offsets from 1.0; folding the +1 at load time keeps the runtime <c>RmsNorm</c> path
    /// unchanged.</summary>
    private static unsafe void AddOneInPlace(Tensor scale)
    {
        if (scale.DType != DType.F32)
            throw new InvalidOperationException("AddOneInPlace requires F32 scale.");
        float* p = (float*)scale.DataPointer;
        long count = scale.Shape.ElementCount;
        for (long i = 0; i < count; i++) p[i] += 1.0f;
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _embedWeight = null;
            _finalNormWeight = null;
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Per-layer block: RMSNorm → GQA self-attn (causal, with RoPE + optional
    // per-head Q/K RMSNorm) → residual → RMSNorm → SwiGLU MLP → residual.
    // ──────────────────────────────────────────────────────────────────────
    private sealed unsafe class LlamaBlock
    {
        // Diagnostic: first-block-only stage probes for HARTSY_TE_PROBE.
        private static bool _probedDetail;

        private readonly LlamaStyleEncoderConfig _config;

        // Norms (pre-attn and pre-mlp), F32. For Gemma 2 (HasFfnSandwichNorms=true), `_postAttnNorm`
        // is the post-attention sandwich norm (applied to attention output before residual), and
        // `_preFfnNorm` / `_postFfnNorm` are loaded from `pre_feedforward_layernorm` and
        // `post_feedforward_layernorm`. For Llama / Qwen, `_postAttnNorm` is the pre-MLP norm and
        // the FFN-sandwich slots stay null.
        private Tensor? _inputNorm;
        private Tensor? _postAttnNorm;
        private Tensor? _preFfnNorm;
        private Tensor? _postFfnNorm;

        // Attention projections (kept native for cuBLAS; bias absent for Qwen3/Llama).
        private Tensor? _qProj;
        private Tensor? _kProj;
        private Tensor? _vProj;
        private Tensor? _oProj;

        // Q/K/V biases (Qwen2 / Qwen2.5-VL convention — o_proj has none). Loaded only when
        // config.AttentionBias is set AND the checkpoint ships them.
        private Tensor? _qBias;
        private Tensor? _kBias;
        private Tensor? _vBias;

        // Per-head q/k norms (Qwen3 only). F32 because broadcast-applied via RmsNorm CPU path.
        private Tensor? _qHeadNorm;
        private Tensor? _kHeadNorm;

        // SwiGLU MLP.
        private Tensor? _gateProj;
        private Tensor? _upProj;
        private Tensor? _downProj;

        public LlamaBlock(LlamaStyleEncoderConfig config) { _config = config; }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
        {
            _inputNorm = CastToF32IfNeeded(weights[$"{prefix}.input_layernorm.weight"]);
            _postAttnNorm = CastToF32IfNeeded(weights[$"{prefix}.post_attention_layernorm.weight"]);
            if (_config.RmsNormScalePlusOne)
            {
                AddOneInPlace(_inputNorm);
                AddOneInPlace(_postAttnNorm);
            }
            if (_config.HasFfnSandwichNorms)
            {
                _preFfnNorm = CastToF32IfNeeded(weights[$"{prefix}.pre_feedforward_layernorm.weight"]);
                _postFfnNorm = CastToF32IfNeeded(weights[$"{prefix}.post_feedforward_layernorm.weight"]);
                if (_config.RmsNormScalePlusOne)
                {
                    AddOneInPlace(_preFfnNorm);
                    AddOneInPlace(_postFfnNorm);
                }
            }

            _qProj = weights[$"{prefix}.self_attn.q_proj.weight"];
            _kProj = weights[$"{prefix}.self_attn.k_proj.weight"];
            _vProj = weights[$"{prefix}.self_attn.v_proj.weight"];
            _oProj = weights[$"{prefix}.self_attn.o_proj.weight"];

            if (_config.AttentionBias)
            {
                // TryGetValue keeps repackaged checkpoints that stripped biases loadable.
                weights.TryGetValue($"{prefix}.self_attn.q_proj.bias", out _qBias);
                weights.TryGetValue($"{prefix}.self_attn.k_proj.bias", out _kBias);
                weights.TryGetValue($"{prefix}.self_attn.v_proj.bias", out _vBias);
            }

            if (_config.QkHeadNorm)
            {
                _qHeadNorm = CastToF32IfNeeded(weights[$"{prefix}.self_attn.q_norm.weight"]);
                _kHeadNorm = CastToF32IfNeeded(weights[$"{prefix}.self_attn.k_norm.weight"]);
                // Gemma 3's per-head q/k norms are Gemma RMSNorms (scale stored as offset from 1). Qwen3
                // (RmsNormScalePlusOne=false) stores them directly, so this only fires for the Gemma path.
                if (_config.RmsNormScalePlusOne)
                {
                    AddOneInPlace(_qHeadNorm);
                    AddOneInPlace(_kHeadNorm);
                }
            }

            _gateProj = weights[$"{prefix}.mlp.gate_proj.weight"];
            _upProj = weights[$"{prefix}.mlp.up_proj.weight"];
            _downProj = weights[$"{prefix}.mlp.down_proj.weight"];
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            if (_inputNorm is not null) yield return _inputNorm;
            if (_postAttnNorm is not null) yield return _postAttnNorm;
            if (_preFfnNorm is not null) yield return _preFfnNorm;
            if (_postFfnNorm is not null) yield return _postFfnNorm;
            if (_qProj is not null) yield return _qProj;
            if (_kProj is not null) yield return _kProj;
            if (_vProj is not null) yield return _vProj;
            if (_oProj is not null) yield return _oProj;
            if (_qBias is not null) yield return _qBias;
            if (_kBias is not null) yield return _kBias;
            if (_vBias is not null) yield return _vBias;
            if (_qHeadNorm is not null) yield return _qHeadNorm;
            if (_kHeadNorm is not null) yield return _kHeadNorm;
            if (_gateProj is not null) yield return _gateProj;
            if (_upProj is not null) yield return _upProj;
            if (_downProj is not null) yield return _downProj;
        }

        public Tensor Forward(IBackend backend, Tensor hidden, Tensor causalMask,
            Tensor ropeCos, Tensor ropeSin, int seqLen)
        {
            int batch = (int)hidden.Shape[0];
            int H = _config.HiddenSize;
            int Hq = _config.NumQueryHeads;
            int Hkv = _config.NumKvHeads;
            int D = _config.HeadDim;
            int Qd = _config.QDim;       // Hq * D

            // ── Attention sub-block ──────────────────────────────────────
            // 1. Pre-attention RMSNorm.
            TensorShape hShape = new TensorShape(batch, seqLen, H);
            Tensor preAttn = new Tensor(hShape, DType.F32);
            backend.RmsNorm(preAttn, hidden, _inputNorm!, _config.RmsNormEps);

            // 2. Q/K/V projections.
            TensorShape qShape = new(batch, seqLen, Qd);
            // Linear only requires the same contiguous element count; keeping the logical [B,S,H,D]
            // shape here makes the following device-side permutation explicit without creating a
            // Reshape view (a distinct Tensor identity would miss the activation cache and force D2H).
            TensorShape qTokenShape = new TensorShape(batch, seqLen, Hq, D);
            TensorShape kvTokenShape = new TensorShape(batch, seqLen, Hkv, D);
            Tensor qFlat = new Tensor(qTokenShape, DType.F32);
            Tensor kFlat = new Tensor(kvTokenShape, DType.F32);
            Tensor vFlat = new Tensor(kvTokenShape, DType.F32);
            backend.Linear(qFlat, preAttn, _qProj!, _qBias);
            backend.Linear(kFlat, preAttn, _kProj!, _kBias);
            backend.Linear(vFlat, preAttn, _vProj!, _vBias);
            if (TeProbe && !_probedDetail)
            {
                ProbeAbsmax("blk0 preAttn", preAttn);
                ProbeAbsmax("blk0 q", qFlat);
                ProbeAbsmax("blk0 k", kFlat);
                ProbeAbsmax("blk0 v", vFlat);
                Logs.Debug($"[TE probe] blk0 qProj dtype={_qProj!.DType} scale={_qProj.Fp8ScaleFactor}");
            }
            preAttn.Dispose();

            // 3. Reshape Q to [B, Hq, S, D]; K, V to [B, Hkv, S, D].
            TensorShape qMhShape = new TensorShape(batch, Hq, seqLen, D);
            TensorShape kvMhShape = new TensorShape(batch, Hkv, seqLen, D);
            Tensor qMh = new Tensor(qMhShape, DType.F32);
            Tensor kMh = new Tensor(kvMhShape, DType.F32);
            Tensor vMh = new Tensor(kvMhShape, DType.F32);
            backend.Permute0213(qMh, qFlat, seqLen, Hq, D);
            backend.Permute0213(kMh, kFlat, seqLen, Hkv, D);
            backend.Permute0213(vMh, vFlat, seqLen, Hkv, D);
            qFlat.Dispose();
            kFlat.Dispose();
            vFlat.Dispose();

            // 4. Per-head Q/K RMSNorm (Qwen3-only).
            if (_config.QkHeadNorm)
            {
                Tensor qNormed = new Tensor(qMhShape, DType.F32);
                Tensor kNormed = new Tensor(kvMhShape, DType.F32);
                backend.RmsNorm(qNormed, qMh, _qHeadNorm!, _config.RmsNormEps);
                backend.RmsNorm(kNormed, kMh, _kHeadNorm!, _config.RmsNormEps);
                qMh.Dispose();
                kMh.Dispose();
                qMh = qNormed;
                kMh = kNormed;
            }

            // 5. Apply RoPE in-place to Q and K (split-half rotation, Llama convention).
            backend.ApplyRopeSingleHeadMajor(qMh, ropeCos, ropeSin, D);
            backend.ApplyRopeSingleHeadMajor(kMh, ropeCos, ropeSin, D);

            // 6. GQA: repeat KV from Hkv heads to Hq heads (KvGroupSize copies of each head).
            Tensor kRepeated = kMh;
            Tensor vRepeated = vMh;
            if (Hkv != Hq)
            {
                kRepeated = new Tensor(qMhShape, DType.F32);
                vRepeated = new Tensor(qMhShape, DType.F32);
                backend.RepeatKvHeads(kRepeated, kMh, Hkv, _config.KvGroupSize);
                backend.RepeatKvHeads(vRepeated, vMh, Hkv, _config.KvGroupSize);
                kMh.Dispose();
                vMh.Dispose();
            }

            // 7. Causal SDPA.
            float scale = 1.0f / MathF.Sqrt(D);
            Tensor attnOut = new Tensor(qMhShape, DType.F32);
            backend.ScaledDotProductAttention(attnOut, qMh, kRepeated, vRepeated, causalMask, scale);
            if (TeProbe && !_probedDetail)
            {
                ProbeAbsmax("blk0 qRoped", qMh);
                ProbeAbsmax("blk0 attnOut", attnOut);
                ProbeAbsmax("blk0 mask", causalMask);
            }
            qMh.Dispose();
            kRepeated.Dispose();
            vRepeated.Dispose();

            // 8. Reshape multi-head back to flat [B, S, Qd], then o_proj.
            Tensor attnFlat = new Tensor(qShape, DType.F32);
            // permute(0,2,1,3) is its own inverse when the S/H arguments are swapped.
            backend.Permute0213(attnFlat, attnOut, Hq, seqLen, D);
            attnOut.Dispose();

            Tensor attnProj = new Tensor(hShape, DType.F32);
            backend.Linear(attnProj, attnFlat, _oProj!, null);
            if (TeProbe && !_probedDetail) ProbeAbsmax("blk0 oProj", attnProj);
            attnFlat.Dispose();

            // 9. First residual.
            // Gemma 2 inserts a "sandwich" RMSNorm between attention output and the residual add
            // (post_attention_layernorm). Llama / Qwen skip this step.
            Tensor postAttnInput = attnProj;
            if (_config.HasFfnSandwichNorms)
            {
                Tensor sandwich = new Tensor(hShape, DType.F32);
                backend.RmsNorm(sandwich, attnProj, _postAttnNorm!, _config.RmsNormEps);
                attnProj.Dispose();
                postAttnInput = sandwich;
            }
            Tensor afterAttn = new Tensor(hShape, DType.F32);
            backend.Add(afterAttn, hidden, postAttnInput);
            postAttnInput.Dispose();

            // ── MLP sub-block (SwiGLU or GeGLU) ──────────────────────────
            // 10. Pre-mlp RMSNorm. For Gemma 2, this is `pre_feedforward_layernorm`; for Llama, it's
            // `post_attention_layernorm` (semantically the pre-MLP norm).
            Tensor preMlpScale = _config.HasFfnSandwichNorms ? _preFfnNorm! : _postAttnNorm!;
            Tensor preMlp = new Tensor(hShape, DType.F32);
            backend.RmsNorm(preMlp, afterAttn, preMlpScale, _config.RmsNormEps);

            // 11. Gate + Up projections, SiLU/GeluTanh(gate) * up, then Down projection.
            TensorShape mlpShape = new TensorShape(batch, seqLen, _config.IntermediateSize);
            Tensor gate = new Tensor(mlpShape, DType.F32);
            Tensor up = new Tensor(mlpShape, DType.F32);
            backend.Linear(gate, preMlp, _gateProj!, null);
            backend.Linear(up, preMlp, _upProj!, null);
            preMlp.Dispose();

            Tensor activated = new Tensor(mlpShape, DType.F32);
            if (_config.Activation == MlpActivation.GeluTanh)
                backend.Gelu(activated, gate);
            else
                backend.Silu(activated, gate);
            gate.Dispose();

            Tensor gated = new Tensor(mlpShape, DType.F32);
            backend.Mul(gated, activated, up);
            activated.Dispose();
            up.Dispose();

            Tensor mlpOut = new Tensor(hShape, DType.F32);
            backend.Linear(mlpOut, gated, _downProj!, null);
            if (TeProbe && !_probedDetail) { ProbeAbsmax("blk0 mlpOut", mlpOut); _probedDetail = true; }
            gated.Dispose();

            // 11b. Gemma 2: post-FFN sandwich norm before the residual add.
            Tensor postMlpInput = mlpOut;
            if (_config.HasFfnSandwichNorms)
            {
                Tensor sandwich = new Tensor(hShape, DType.F32);
                backend.RmsNorm(sandwich, mlpOut, _postFfnNorm!, _config.RmsNormEps);
                mlpOut.Dispose();
                postMlpInput = sandwich;
            }

            // 12. Second residual.
            Tensor result = new Tensor(hShape, DType.F32);
            backend.Add(result, afterAttn, postMlpInput);
            afterAttn.Dispose();
            postMlpInput.Dispose();

            return result;
        }

    }
}

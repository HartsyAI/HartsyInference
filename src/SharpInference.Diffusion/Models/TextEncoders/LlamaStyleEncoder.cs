using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.TextEncoders;

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
    private int _ropeBuiltForMaxLen;

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

    /// <summary>Loads all weights from a HuggingFace-style key dict (keys like <c>model.layers.{i}.self_attn.q_proj.weight</c>). Cast to F32 sites are: token embedding (CPU lookup), RMSNorm scales (CPU pointer code expects float*), per-head q/k norms.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        Tensor rawEmbed = weights["model.embed_tokens.weight"];
        _embedWeight = CastToF32IfNeeded(rawEmbed);

        if (_config.HasFinalNorm)
        {
            Tensor rawFinalNorm = weights["model.norm.weight"];
            _finalNormWeight = CastToF32IfNeeded(rawFinalNorm);
            if (_config.RmsNormScalePlusOne) AddOneInPlace(_finalNormWeight);
        }

        for (int i = 0; i < _config.NumLayers; i++)
            _blocks[i].LoadWeights(weights, $"model.layers.{i}");

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

        int batch = tokenIds.Length;
        int seqLen = tokenIds[0].Length;
        if (seqLen > _config.MaxPositionEmbeddings)
            throw new InvalidOperationException(
                $"Prompt length {seqLen} exceeds MaxPositionEmbeddings ({_config.MaxPositionEmbeddings}).");

        EnsureRopeTable(seqLen);

        // 1. Token embedding lookup — CPU code, uses the F32-cast embed table.
        Tensor hidden = EmbeddingLookup(tokenIds, batch, seqLen);

        // 2. Build a causal mask [seqLen, seqLen] with 0 for allowed positions and -inf (large negative) for masked.
        // SDPA backends typically add the mask to scaled QK^T before softmax; -inf masks are clamped via softmax.
        Tensor causalMask = BuildCausalMask(seqLen);

        // 3. Layer loop — each block reads `hidden`, allocates a new tensor, returns it. Old hidden disposed.
        for (int i = 0; i < _config.NumLayers; i++)
        {
            Tensor next = _blocks[i].Forward(backend, hidden, causalMask, _ropeCos!, _ropeSin!, seqLen);
            hidden.Dispose();
            hidden = next;
        }

        causalMask.Dispose();

        // 4. Final RMSNorm (skipped for checkpoints that don't ship one — e.g. Mistral-Small-3
        // distilled for Flux.2 Dev. In that case we return the last block's raw output, matching
        // how the diffusers pipeline treats feature-extractor encoders.)
        if (_config.HasFinalNorm)
        {
            TensorShape outShape = new TensorShape(batch, seqLen, _config.HiddenSize);
            Tensor output = new Tensor(outShape, DType.F32);
            backend.RmsNorm(output, hidden, _finalNormWeight!, _config.RmsNormEps);
            hidden.Dispose();
            return output;
        }

        return hidden;
    }

    /// <summary>Encodes a prompt and concatenates hidden states from selected intermediate layers along the feature axis. Used by Flux.2 Klein (layers 9, 18, 27 → 7680 dim for Qwen3-4B). HuggingFace indexing: <c>k=0</c> is the embedding output (pre-layer-0); <c>k=1..N</c> is post-layer-(k−1). Final RMSNorm is NOT applied to intermediate outputs (matches HF — only the last hidden state passes through <c>model.norm</c>).</summary>
    /// <param name="layerIndices">Layer indices in HuggingFace convention: 0 = embeddings, k = post-layer-(k-1). Must be sorted ascending and within [0, NumLayers].</param>
    /// <returns>F32 tensor of shape <c>[batch, seqLen, layerIndices.Length × hiddenSize]</c>. Channels are arranged as <c>[layer_0_features, layer_1_features, ..., layer_N_features]</c> per token (matching diffusers' <c>permute(0, 2, 1, 3).reshape(B, S, N*H)</c>).</returns>
    public Tensor EncodeMultiLayer(IBackend backend, int[][] tokenIds, int[] layerIndices)
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

        int batch = tokenIds.Length;
        int seqLen = tokenIds[0].Length;
        int H = _config.HiddenSize;
        int K = layerIndices.Length;
        if (seqLen > _config.MaxPositionEmbeddings)
            throw new InvalidOperationException(
                $"Prompt length {seqLen} exceeds MaxPositionEmbeddings ({_config.MaxPositionEmbeddings}).");

        EnsureRopeTable(seqLen);

        Tensor hidden = EmbeddingLookup(tokenIds, batch, seqLen);
        Tensor causalMask = BuildCausalMask(seqLen);

        // Allocate the output tensor up-front: [B, S, K*H]. We'll scatter each requested layer's
        // [B, S, H] hidden state into its slice.
        TensorShape outShape = new TensorShape(batch, seqLen, K * H);
        Tensor output = new Tensor(outShape, DType.F32);
        float* outPtr = (float*)output.DataPointer;
        int requestIdx = 0;

        // Capture layer index 0 (embeddings) before any block runs.
        if (layerIndices[requestIdx] == 0)
        {
            ScatterLayerSlice(hidden, output, requestIdx, batch, seqLen, H, K);
            requestIdx++;
        }

        for (int i = 0; i < _config.NumLayers; i++)
        {
            Tensor next = _blocks[i].Forward(backend, hidden, causalMask, _ropeCos!, _ropeSin!, seqLen);
            hidden.Dispose();
            hidden = next;

            // Block i produces hidden_states[i+1] in HF terms.
            int hfLayerIndex = i + 1;
            while (requestIdx < K && layerIndices[requestIdx] == hfLayerIndex)
            {
                ScatterLayerSlice(hidden, output, requestIdx, batch, seqLen, H, K);
                requestIdx++;
            }
            if (requestIdx >= K) break; // No more layers to capture — early exit.
        }

        causalMask.Dispose();
        hidden.Dispose();
        return output;
    }

    /// <summary>Copies a layer's <c>[B, S, H]</c> hidden state into channels <c>[layerSlot*H .. layerSlot*H + H)</c> of an output tensor of shape <c>[B, S, K*H]</c>.</summary>
    private static void ScatterLayerSlice(Tensor src, Tensor dst, int layerSlot, int batch, int seqLen, int H, int K)
    {
        float* sp = (float*)src.DataPointer;
        float* dp = (float*)dst.DataPointer;
        long bytesPerToken = (long)H * sizeof(float);
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                long srcOff = ((long)b * seqLen + s) * H;
                long dstOff = ((long)b * seqLen + s) * (K * H) + (long)layerSlot * H;
                Buffer.MemoryCopy(sp + srcOff, dp + dstOff, bytesPerToken, bytesPerToken);
            }
        }
    }

    private Tensor EmbeddingLookup(int[][] tokenIds, int batch, int seqLen)
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
        return output;
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

        _ropeCos = new float[targetLen * halfDim];
        _ropeSin = new float[targetLen * halfDim];

        // freqs[k] = 1 / theta^(2k / headDim), k in [0, halfDim)
        for (int p = 0; p < targetLen; p++)
        {
            for (int k = 0; k < halfDim; k++)
            {
                double freq = 1.0 / Math.Pow(_config.RopeTheta, (double)(2 * k) / _config.HeadDim);
                double angle = p * freq;
                _ropeCos[p * halfDim + k] = (float)Math.Cos(angle);
                _ropeSin[p * halfDim + k] = (float)Math.Sin(angle);
            }
        }
        _ropeBuiltForMaxLen = targetLen;
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

            if (_config.QkHeadNorm)
            {
                _qHeadNorm = CastToF32IfNeeded(weights[$"{prefix}.self_attn.q_norm.weight"]);
                _kHeadNorm = CastToF32IfNeeded(weights[$"{prefix}.self_attn.k_norm.weight"]);
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
            if (_qHeadNorm is not null) yield return _qHeadNorm;
            if (_kHeadNorm is not null) yield return _kHeadNorm;
            if (_gateProj is not null) yield return _gateProj;
            if (_upProj is not null) yield return _upProj;
            if (_downProj is not null) yield return _downProj;
        }

        public Tensor Forward(IBackend backend, Tensor hidden, Tensor causalMask,
            float[] ropeCos, float[] ropeSin, int seqLen)
        {
            int batch = (int)hidden.Shape[0];
            int H = _config.HiddenSize;
            int Hq = _config.NumQueryHeads;
            int Hkv = _config.NumKvHeads;
            int D = _config.HeadDim;
            int Qd = _config.QDim;       // Hq * D
            int Kvd = _config.KvDim;     // Hkv * D

            // ── Attention sub-block ──────────────────────────────────────
            // 1. Pre-attention RMSNorm.
            TensorShape hShape = new TensorShape(batch, seqLen, H);
            Tensor preAttn = new Tensor(hShape, DType.F32);
            backend.RmsNorm(preAttn, hidden, _inputNorm!, _config.RmsNormEps);

            // 2. Q/K/V projections.
            TensorShape qShape = new TensorShape(batch, seqLen, Qd);
            TensorShape kvShape = new TensorShape(batch, seqLen, Kvd);
            Tensor qFlat = new Tensor(qShape, DType.F32);
            Tensor kFlat = new Tensor(kvShape, DType.F32);
            Tensor vFlat = new Tensor(kvShape, DType.F32);
            backend.Linear(qFlat, preAttn, _qProj!, null);
            backend.Linear(kFlat, preAttn, _kProj!, null);
            backend.Linear(vFlat, preAttn, _vProj!, null);
            preAttn.Dispose();

            // 3. Reshape Q to [B, Hq, S, D]; K, V to [B, Hkv, S, D].
            TensorShape qMhShape = new TensorShape(batch, Hq, seqLen, D);
            TensorShape kvMhShape = new TensorShape(batch, Hkv, seqLen, D);
            Tensor qMh = new Tensor(qMhShape, DType.F32);
            Tensor kMh = new Tensor(kvMhShape, DType.F32);
            Tensor vMh = new Tensor(kvMhShape, DType.F32);
            ReshapeFlatToMultiHead(qMh, qFlat, batch, seqLen, Hq, D);
            ReshapeFlatToMultiHead(kMh, kFlat, batch, seqLen, Hkv, D);
            ReshapeFlatToMultiHead(vMh, vFlat, batch, seqLen, Hkv, D);
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
            ApplyRopeSplitHalf(qMh, ropeCos, ropeSin, batch, Hq, seqLen, D);
            ApplyRopeSplitHalf(kMh, ropeCos, ropeSin, batch, Hkv, seqLen, D);

            // 6. GQA: repeat KV from Hkv heads to Hq heads (KvGroupSize copies of each head).
            Tensor kRepeated = kMh;
            Tensor vRepeated = vMh;
            if (Hkv != Hq)
            {
                kRepeated = new Tensor(qMhShape, DType.F32);
                vRepeated = new Tensor(qMhShape, DType.F32);
                RepeatKvHeads(kRepeated, kMh, batch, Hkv, _config.KvGroupSize, seqLen, D);
                RepeatKvHeads(vRepeated, vMh, batch, Hkv, _config.KvGroupSize, seqLen, D);
                kMh.Dispose();
                vMh.Dispose();
            }

            // 7. Causal SDPA.
            float scale = 1.0f / MathF.Sqrt(D);
            Tensor attnOut = new Tensor(qMhShape, DType.F32);
            backend.ScaledDotProductAttention(attnOut, qMh, kRepeated, vRepeated, causalMask, scale);
            qMh.Dispose();
            kRepeated.Dispose();
            vRepeated.Dispose();

            // 8. Reshape multi-head back to flat [B, S, Qd], then o_proj.
            Tensor attnFlat = new Tensor(qShape, DType.F32);
            ReshapeMultiHeadToFlat(attnFlat, attnOut, batch, seqLen, Hq, D);
            attnOut.Dispose();

            Tensor attnProj = new Tensor(hShape, DType.F32);
            backend.Linear(attnProj, attnFlat, _oProj!, null);
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

        // [B, S, H*D] → [B, H, S, D]
        private static void ReshapeFlatToMultiHead(Tensor output, Tensor input, int batch, int seqLen, int heads, int headDim)
        {
            float* inPtr = (float*)input.DataPointer;
            float* outPtr = (float*)output.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < seqLen; s++)
                    for (int h = 0; h < heads; h++)
                    {
                        long inOff = ((long)b * seqLen + s) * heads * headDim + (long)h * headDim;
                        long outOff = (((long)b * heads + h) * seqLen + s) * headDim;
                        Buffer.MemoryCopy(inPtr + inOff, outPtr + outOff, headDim * sizeof(float), headDim * sizeof(float));
                    }
        }

        // [B, H, S, D] → [B, S, H*D]
        private static void ReshapeMultiHeadToFlat(Tensor output, Tensor input, int batch, int seqLen, int heads, int headDim)
        {
            float* inPtr = (float*)input.DataPointer;
            float* outPtr = (float*)output.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < seqLen; s++)
                    for (int h = 0; h < heads; h++)
                    {
                        long inOff = (((long)b * heads + h) * seqLen + s) * headDim;
                        long outOff = ((long)b * seqLen + s) * heads * headDim + (long)h * headDim;
                        Buffer.MemoryCopy(inPtr + inOff, outPtr + outOff, headDim * sizeof(float), headDim * sizeof(float));
                    }
        }

        // GQA: replicate each KV head `groupSize` times into Q-head positions.
        // [B, Hkv, S, D] → [B, Hkv*groupSize=Hq, S, D]
        private static void RepeatKvHeads(Tensor output, Tensor input, int batch, int kvHeads, int groupSize, int seqLen, int headDim)
        {
            float* inPtr = (float*)input.DataPointer;
            float* outPtr = (float*)output.DataPointer;
            long perHead = (long)seqLen * headDim;
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < kvHeads; h++)
                {
                    long srcOff = ((long)b * kvHeads + h) * perHead;
                    for (int g = 0; g < groupSize; g++)
                    {
                        int qHead = h * groupSize + g;
                        long dstOff = ((long)b * (kvHeads * groupSize) + qHead) * perHead;
                        Buffer.MemoryCopy(inPtr + srcOff, outPtr + dstOff, perHead * sizeof(float), perHead * sizeof(float));
                    }
                }
            }
        }

        /// <summary>Llama-style RoPE (split-half): for each head vector x[0..D] at position p, produce
        /// y[i] = x[i]*cos[i'] − x[i+D/2]*sin[i'] for i in [0, D/2),
        /// y[i] = x[i]*cos[i']' + x[i−D/2]*sin[i']'  for i in [D/2, D),
        /// where cos/sin are duplicated across the half so cos[i] = cos[i'] = cos[(i mod D/2)].</summary>
        private static void ApplyRopeSplitHalf(Tensor q, float[] cos, float[] sin, int batch, int heads, int seqLen, int headDim)
        {
            int half = headDim / 2;
            float* qPtr = (float*)q.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < heads; h++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        long off = (((long)b * heads + h) * seqLen + s) * headDim;
                        int posOff = s * half;
                        for (int i = 0; i < half; i++)
                        {
                            float c = cos[posOff + i];
                            float si = sin[posOff + i];
                            float a = qPtr[off + i];
                            float b2 = qPtr[off + i + half];
                            qPtr[off + i] = a * c - b2 * si;
                            qPtr[off + i + half] = b2 * c + a * si;
                        }
                    }
                }
            }
        }
    }
}

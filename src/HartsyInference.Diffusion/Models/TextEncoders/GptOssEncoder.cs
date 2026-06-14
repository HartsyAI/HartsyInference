using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>GPT-OSS decoder transformer (<c>openai/gpt-oss-20b</c>, repackaged in <c>microsoft/Lens/text_encoder/</c>). Architecturally divergent from the Llama family on five axes: (1) **Mixture-of-Experts FFN** — 32 experts × top-4 routing per token, with the GPT-OSS-specific clamped gated MLP (<see cref="GptOssMoeFfn"/>); (2) **Attention sinks** — per-head learnable scalars mixed into the softmax denominator as a virtual zero-value key, mitigating attention sink pathology in long contexts; (3) **Alternating sliding/full attention** — layer 0/2/4/... use a 128-token sliding window, 1/3/5/... use unrestricted causal attention; (4) **YaRN-extended RoPE** with theta=150000, factor=32 — extends 4096 → 131072 effective context; (5) **Q/K/V/O biases enabled** by default. Used as a feature-extractor encoder by Microsoft Lens — multi-layer hidden states captured at [5, 11, 17, 23] feed the DiT.</summary>
public sealed unsafe class GptOssEncoder : IDisposable
{
    private readonly LlamaStyleEncoderConfig _config;
    private readonly GptOssBlock[] _blocks;
    private Tensor? _causalMaskFull;
    private Tensor? _causalMaskSliding;
    private int _causalMasksBuiltForLen;

    private Tensor? _embedWeight;
    private Tensor? _finalNormWeight;

    private float[]? _ropeCos;
    private float[]? _ropeSin;
    private int _ropeBuiltForMaxLen;

    private int _disposed;

    /// <summary>Constructs the encoder skeleton. Validates that the config has GPT-OSS-specific flags set
    /// (<see cref="LlamaStyleEncoderConfig.NumLocalExperts"/> &gt; 0,
    /// <see cref="LlamaStyleEncoderConfig.HasAttentionSinks"/>,
    /// <see cref="LlamaStyleEncoderConfig.LayerAttentionTypes"/> length matches
    /// <see cref="LlamaStyleEncoderConfig.NumLayers"/>). Use <see cref="LlamaStyleEncoderConfig.GptOss"/>
    /// for the canonical preset.</summary>
    public GptOssEncoder(LlamaStyleEncoderConfig config)
    {
        if (config.NumLocalExperts <= 0)
            throw new ArgumentException("GptOssEncoder requires NumLocalExperts > 0 (default GPT-OSS = 32).", nameof(config));
        if (config.NumExpertsPerToken <= 0 || config.NumExpertsPerToken > config.NumLocalExperts)
            throw new ArgumentException(
                $"GptOssEncoder requires 0 < NumExpertsPerToken <= NumLocalExperts; got {config.NumExpertsPerToken} of {config.NumLocalExperts}.",
                nameof(config));
        if (config.LayerAttentionTypes is null || config.LayerAttentionTypes.Length != config.NumLayers)
            throw new ArgumentException(
                $"GptOssEncoder requires LayerAttentionTypes of length NumLayers ({config.NumLayers}); got {config.LayerAttentionTypes?.Length.ToString() ?? "null"}.",
                nameof(config));

        _config = config;
        _blocks = new GptOssBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
            _blocks[i] = new GptOssBlock(config, layerIndex: i);
        _causalMasksBuiltForLen = 0;
    }

    /// <summary>Number of transformer layers — convenience accessor (24 for GPT-OSS).</summary>
    public int NumLayers => _config.NumLayers;

    /// <summary>Loads all weights from a HuggingFace-style key dict (<c>model.embed_tokens.weight</c>,
    /// <c>model.layers.{i}.*</c>, <c>model.norm.weight</c>). The MoE experts (<c>mlp.experts.gate_up_proj</c>,
    /// <c>mlp.experts.down_proj</c>) and their biases are expected in F32 — pass MXFP4-packed weights
    /// through <see cref="HartsyInference.ModelHandler.Mxfp4.Mxfp4Codec.DequantToF32"/> at load time.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        Tensor rawEmbed = weights["model.embed_tokens.weight"];
        _embedWeight = CastToF32IfNeeded(rawEmbed);

        if (_config.HasFinalNorm)
        {
            Tensor rawFinalNorm = weights["model.norm.weight"];
            _finalNormWeight = CastToF32IfNeeded(rawFinalNorm);
        }

        for (int i = 0; i < _config.NumLayers; i++)
            _blocks[i].LoadWeights(weights, $"model.layers.{i}");

        Logs.Verbose($"GptOssEncoder loaded: {_config.NumLayers} layers, hidden={_config.HiddenSize}, " +
                  $"q_heads={_config.NumQueryHeads}, kv_heads={_config.NumKvHeads}, head_dim={_config.HeadDim}, " +
                  $"experts={_config.NumLocalExperts}x{_config.NumExpertsPerToken}, " +
                  $"sliding_window={_config.SlidingWindow}, yarn_factor={_config.YarnFactor}");
    }

    /// <summary>Enumerates all weight tensors for GPU-preload paths.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embedWeight is not null) yield return _embedWeight;
        if (_finalNormWeight is not null) yield return _finalNormWeight;
        for (int i = 0; i < _blocks.Length; i++)
            foreach (Tensor w in _blocks[i].EnumerateWeights())
                yield return w;
    }

    /// <summary>Encodes a batch of token id arrays (all rows must have the same length). Returns
    /// <c>last_hidden_state</c> shaped <c>[B, seqLen, hiddenSize]</c> as F32. Applies the post-block
    /// final RMSNorm.</summary>
    public Tensor Encode(IBackend backend, int[][] tokenIds)
    {
        ThrowIfDisposed();
        int batch = tokenIds.Length;
        int seqLen = tokenIds[0].Length;
        if (seqLen > _config.MaxPositionEmbeddings)
            throw new InvalidOperationException(
                $"Prompt length {seqLen} exceeds MaxPositionEmbeddings ({_config.MaxPositionEmbeddings}).");

        EnsureRopeTable(seqLen);
        EnsureMasks(seqLen);

        Tensor hidden = EmbeddingLookup(tokenIds, batch, seqLen);

        for (int i = 0; i < _config.NumLayers; i++)
        {
            Tensor mask = _config.LayerAttentionTypes![i] == "sliding_attention" ? _causalMaskSliding! : _causalMaskFull!;
            Tensor next = _blocks[i].Forward(backend, hidden, mask, _ropeCos!, _ropeSin!, seqLen);
            hidden.Dispose();
            hidden = next;
        }

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

    /// <summary>Captures hidden states at the given decoder-layer indices (0-indexed, matching
    /// <c>model.layers.{i}</c> output AFTER block i runs — distinct from HuggingFace's
    /// <c>hidden_states[i]</c> convention which is the pre-block-i input). For Lens this is the right
    /// indexing: <c>SelectedLayers=[5,11,17,23]</c> captures the output of decoder layers 5/11/17/23.
    /// Early-exits after the largest selected index — does NOT run the final norm. Returns one tensor
    /// per requested layer, each <c>[B, seqLen, hiddenSize]</c>, in the order they were requested.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="tokenIds">Equal-length token id rows.</param>
    /// <param name="selectedLayers">Decoder-layer indices in <c>[0, NumLayers)</c>. Need not be sorted but no duplicates.</param>
    public List<Tensor> EncodeAtLayers(IBackend backend, int[][] tokenIds, int[] selectedLayers)
    {
        ThrowIfDisposed();
        if (selectedLayers is null || selectedLayers.Length == 0)
            throw new ArgumentException("Must request at least one layer.", nameof(selectedLayers));
        HashSet<int> selectedSet = new(selectedLayers.Length);
        int maxSelected = -1;
        for (int i = 0; i < selectedLayers.Length; i++)
        {
            int idx = selectedLayers[i];
            if (idx < 0 || idx >= _config.NumLayers)
                throw new ArgumentOutOfRangeException(nameof(selectedLayers),
                    $"Layer index {idx} out of range [0, {_config.NumLayers}).");
            if (!selectedSet.Add(idx))
                throw new ArgumentException($"Duplicate layer index {idx}.", nameof(selectedLayers));
            if (idx > maxSelected) maxSelected = idx;
        }

        int batch = tokenIds.Length;
        int seqLen = tokenIds[0].Length;
        if (seqLen > _config.MaxPositionEmbeddings)
            throw new InvalidOperationException(
                $"Prompt length {seqLen} exceeds MaxPositionEmbeddings ({_config.MaxPositionEmbeddings}).");

        EnsureRopeTable(seqLen);
        EnsureMasks(seqLen);

        Tensor hidden = EmbeddingLookup(tokenIds, batch, seqLen);

        // Pre-allocate slots in the order the user requested.
        Tensor?[] outputs = new Tensor?[selectedLayers.Length];

        for (int i = 0; i <= maxSelected; i++)
        {
            Tensor mask = _config.LayerAttentionTypes![i] == "sliding_attention" ? _causalMaskSliding! : _causalMaskFull!;
            Tensor next = _blocks[i].Forward(backend, hidden, mask, _ropeCos!, _ropeSin!, seqLen);
            hidden.Dispose();
            hidden = next;

            // Capture if requested. We clone to give each slot its own lifetime; the running hidden
            // continues to mutate via Dispose-and-replace.
            if (selectedSet.Contains(i))
            {
                int destSlot = Array.IndexOf(selectedLayers, i);
                outputs[destSlot] = CloneTensor(hidden);
            }
        }

        hidden.Dispose();

        List<Tensor> result = new(selectedLayers.Length);
        for (int i = 0; i < selectedLayers.Length; i++)
            result.Add(outputs[i] ?? throw new InvalidOperationException($"Failed to capture layer {selectedLayers[i]}."));
        return result;
    }

    private static Tensor CloneTensor(Tensor src)
    {
        Tensor copy = new Tensor(src.Shape, src.DType);
        long bytes = src.Shape.ElementCount * src.DType.SizeInBytes;
        Buffer.MemoryCopy((void*)src.DataPointer, (void*)copy.DataPointer, bytes, bytes);
        return copy;
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

    /// <summary>Builds (or rebuilds, if seqLen grew) both the full-causal and sliding-window causal masks,
    /// each <c>[seqLen, seqLen]</c>. Sliding-window: position q only attends to positions in
    /// <c>[q - SlidingWindow + 1, q]</c>. Cached on the encoder between calls.</summary>
    private void EnsureMasks(int seqLen)
    {
        if (_causalMaskFull is not null && _causalMasksBuiltForLen >= seqLen)
            return;

        _causalMaskFull?.Dispose();
        _causalMaskSliding?.Dispose();

        TensorShape shape = new TensorShape(seqLen, seqLen);
        Tensor full = new Tensor(shape, DType.F32);
        Tensor sliding = new Tensor(shape, DType.F32);

        const float negInf = -1e30f;
        int window = _config.SlidingWindow;
        float* pFull = (float*)full.DataPointer;
        float* pSlide = (float*)sliding.DataPointer;
        for (int q = 0; q < seqLen; q++)
        {
            int slidingMin = q - window + 1;
            if (slidingMin < 0) slidingMin = 0;
            for (int k = 0; k < seqLen; k++)
            {
                pFull[q * seqLen + k] = k > q ? negInf : 0f;
                pSlide[q * seqLen + k] = (k > q || k < slidingMin) ? negInf : 0f;
            }
        }

        _causalMaskFull = full;
        _causalMaskSliding = sliding;
        _causalMasksBuiltForLen = seqLen;
    }

    /// <summary>Precomputes RoPE cos/sin tables. Applies YaRN frequency scaling when configured,
    /// matching HuggingFace's <c>_compute_yarn_parameters</c> exactly: the extrapolation↔interpolation
    /// blend ramps over the <b>dimension index</b> (via <c>find_correction_dim</c>), NOT over wavelength,
    /// and the inferred <c>attention_factor</c> (<c>0.1·ln(factor)+1</c>) is baked into cos/sin as the
    /// YaRN mscale. <c>truncate=false</c> for GPT-OSS — the correction-range bounds are not floored/ceiled.
    /// Validated against <c>transformers.models.gpt_oss</c> (inv_freq diff &lt; 1e-8, attention_factor exact).</summary>
    private void EnsureRopeTable(int seqLen)
    {
        if (_ropeCos is not null && _ropeBuiltForMaxLen >= seqLen)
            return;

        int halfDim = _config.HeadDim / 2;
        int dim = _config.HeadDim;
        int targetLen = Math.Max(seqLen, _ropeBuiltForMaxLen);
        int rounded = 1;
        while (rounded < targetLen) rounded <<= 1;
        targetLen = Math.Min(rounded, _config.MaxPositionEmbeddings);

        // Base (extrapolation) inverse frequencies: inv_freq[k] = base^-(2k/dim).
        double[] invFreq = new double[halfDim];
        for (int k = 0; k < halfDim; k++)
            invFreq[k] = 1.0 / Math.Pow(_config.RopeTheta, (double)(2 * k) / dim);

        double attentionScaling = 1.0;
        if (_config.RopeScaling == RopeScalingType.Yarn)
        {
            double factor = _config.YarnFactor;
            double betaFast = _config.YarnBetaFast;
            double betaSlow = _config.YarnBetaSlow;
            double origMaxPos = _config.YarnOriginalMaxPosition;
            double logBase = Math.Log(_config.RopeTheta);

            // find_correction_dim: dimension index at which a given rotation count completes.
            double FindCorrectionDim(double numRotations) =>
                (dim * Math.Log(origMaxPos / (numRotations * 2.0 * Math.PI))) / (2.0 * logBase);

            // truncate=false for GPT-OSS → no floor/ceil, just clamp into [0, dim-1].
            double low = Math.Max(FindCorrectionDim(betaFast), 0.0);
            double high = Math.Min(FindCorrectionDim(betaSlow), dim - 1);
            double rampDenom = high == low ? 0.001 : high - low;

            for (int k = 0; k < halfDim; k++)
            {
                // linear_ramp_factor over dim//2 indices, clamped to [0, 1].
                double ramp = (k - low) / rampDenom;
                if (ramp < 0.0) ramp = 0.0;
                else if (ramp > 1.0) ramp = 1.0;
                double extrapolationFactor = 1.0 - ramp;       // 1 → keep, 0 → fully scaled
                double extrapolation = invFreq[k];             // original
                double interpolation = invFreq[k] / factor;    // scaled by 1/factor
                invFreq[k] = interpolation * (1.0 - extrapolationFactor) + extrapolation * extrapolationFactor;
            }

            // Inferred attention_factor (mscale) = get_mscale(factor) = 0.1·ln(factor)+1 for factor > 1.
            attentionScaling = factor <= 1.0 ? 1.0 : 0.1 * Math.Log(factor) + 1.0;
        }

        _ropeCos = new float[targetLen * halfDim];
        _ropeSin = new float[targetLen * halfDim];
        for (int p = 0; p < targetLen; p++)
        {
            for (int k = 0; k < halfDim; k++)
            {
                double angle = p * invFreq[k];
                _ropeCos[p * halfDim + k] = (float)(Math.Cos(angle) * attentionScaling);
                _ropeSin[p * halfDim + k] = (float)(Math.Sin(angle) * attentionScaling);
            }
        }
        _ropeBuiltForMaxLen = targetLen;
    }

    private static Tensor CastToF32IfNeeded(Tensor t) =>
        t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _embedWeight = null;
            _finalNormWeight = null;
            if (_causalMasksBuiltForLen > 0)
            {
                _causalMaskFull?.Dispose();
                _causalMaskSliding?.Dispose();
            }
        }
        GC.SuppressFinalize(this);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Per-layer GPT-OSS block: RMSNorm → biased GQA self-attn (RoPE, sinks,
    // sliding-or-full causal mask via passed-in mask, manual SDPA so we can
    // incorporate sinks) → residual → RMSNorm → MoE FFN → residual.
    // ──────────────────────────────────────────────────────────────────────
    private sealed unsafe class GptOssBlock
    {
        private readonly LlamaStyleEncoderConfig _config;
        private readonly int _layerIndex;
        private readonly GptOssMoeFfn _moe;

        private Tensor? _inputNorm;
        private Tensor? _postAttnNorm;

        private Tensor? _qProj;
        private Tensor? _qProjBias;
        private Tensor? _kProj;
        private Tensor? _kProjBias;
        private Tensor? _vProj;
        private Tensor? _vProjBias;
        private Tensor? _oProj;
        private Tensor? _oProjBias;

        private Tensor? _sinks;   // [num_q_heads]

        public GptOssBlock(LlamaStyleEncoderConfig config, int layerIndex)
        {
            _config = config;
            _layerIndex = layerIndex;
            _moe = new GptOssMoeFfn(
                config.HiddenSize, config.IntermediateSize,
                config.NumLocalExperts, config.NumExpertsPerToken,
                config.ClampedSwiGluLimit, config.ClampedSwiGluAlpha);
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
        {
            _inputNorm = CastToF32IfNeeded(weights[$"{prefix}.input_layernorm.weight"]);
            _postAttnNorm = CastToF32IfNeeded(weights[$"{prefix}.post_attention_layernorm.weight"]);

            _qProj = weights[$"{prefix}.self_attn.q_proj.weight"];
            _kProj = weights[$"{prefix}.self_attn.k_proj.weight"];
            _vProj = weights[$"{prefix}.self_attn.v_proj.weight"];
            _oProj = weights[$"{prefix}.self_attn.o_proj.weight"];

            if (_config.AttentionBias)
            {
                _qProjBias = weights[$"{prefix}.self_attn.q_proj.bias"];
                _kProjBias = weights[$"{prefix}.self_attn.k_proj.bias"];
                _vProjBias = weights[$"{prefix}.self_attn.v_proj.bias"];
                _oProjBias = weights[$"{prefix}.self_attn.o_proj.bias"];
            }

            if (_config.HasAttentionSinks)
                _sinks = CastToF32IfNeeded(weights[$"{prefix}.self_attn.sinks"]);

            _moe.LoadWeights(weights, $"{prefix}.mlp");
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            if (_inputNorm is not null) yield return _inputNorm;
            if (_postAttnNorm is not null) yield return _postAttnNorm;
            if (_qProj is not null) yield return _qProj;
            if (_qProjBias is not null) yield return _qProjBias;
            if (_kProj is not null) yield return _kProj;
            if (_kProjBias is not null) yield return _kProjBias;
            if (_vProj is not null) yield return _vProj;
            if (_vProjBias is not null) yield return _vProjBias;
            if (_oProj is not null) yield return _oProj;
            if (_oProjBias is not null) yield return _oProjBias;
            if (_sinks is not null) yield return _sinks;
            foreach (Tensor w in _moe.EnumerateWeights()) yield return w;
        }

        public Tensor Forward(IBackend backend, Tensor hidden, Tensor mask,
            float[] ropeCos, float[] ropeSin, int seqLen)
        {
            int batch = (int)hidden.Shape[0];
            int H = _config.HiddenSize;
            int Hq = _config.NumQueryHeads;
            int Hkv = _config.NumKvHeads;
            int D = _config.HeadDim;
            int Qd = _config.QDim;
            int Kvd = _config.KvDim;

            // ── Attention sub-block ──
            TensorShape hShape = new TensorShape(batch, seqLen, H);
            Tensor preAttn = new Tensor(hShape, DType.F32);
            backend.RmsNorm(preAttn, hidden, _inputNorm!, _config.RmsNormEps);

            TensorShape qShape = new TensorShape(batch, seqLen, Qd);
            TensorShape kvShape = new TensorShape(batch, seqLen, Kvd);
            Tensor qFlat = new Tensor(qShape, DType.F32);
            Tensor kFlat = new Tensor(kvShape, DType.F32);
            Tensor vFlat = new Tensor(kvShape, DType.F32);
            backend.Linear(qFlat, preAttn, _qProj!, _qProjBias);
            backend.Linear(kFlat, preAttn, _kProj!, _kProjBias);
            backend.Linear(vFlat, preAttn, _vProj!, _vProjBias);
            preAttn.Dispose();

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

            ApplyRopeSplitHalf(qMh, ropeCos, ropeSin, batch, Hq, seqLen, D);
            ApplyRopeSplitHalf(kMh, ropeCos, ropeSin, batch, Hkv, seqLen, D);

            // GQA: replicate KV from Hkv heads to Hq heads.
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

            // Manual SDPA-with-sinks. We don't use backend.ScaledDotProductAttention because we need
            // the sink term in the softmax denominator and we don't have access to the per-head log-sum-exp
            // from the standard SDPA primitive.
            Tensor attnOut = new Tensor(qMhShape, DType.F32);
            ManualScaledDotProductAttention(attnOut, qMh, kRepeated, vRepeated, mask, _sinks,
                batch, Hq, seqLen, D, _config.HasAttentionSinks);

            qMh.Dispose();
            kRepeated.Dispose();
            vRepeated.Dispose();

            Tensor attnFlat = new Tensor(qShape, DType.F32);
            ReshapeMultiHeadToFlat(attnFlat, attnOut, batch, seqLen, Hq, D);
            attnOut.Dispose();

            Tensor attnProj = new Tensor(hShape, DType.F32);
            backend.Linear(attnProj, attnFlat, _oProj!, _oProjBias);
            attnFlat.Dispose();

            Tensor afterAttn = new Tensor(hShape, DType.F32);
            backend.Add(afterAttn, hidden, attnProj);
            attnProj.Dispose();

            // ── MoE FFN sub-block ──
            Tensor preMlp = new Tensor(hShape, DType.F32);
            backend.RmsNorm(preMlp, afterAttn, _postAttnNorm!, _config.RmsNormEps);

            Tensor moeOut = _moe.Forward(backend, preMlp);
            preMlp.Dispose();

            Tensor result = new Tensor(hShape, DType.F32);
            backend.Add(result, afterAttn, moeOut);
            afterAttn.Dispose();
            moeOut.Dispose();

            return result;
        }

        /// <summary>Manual scaled-dot-product attention with optional GPT-OSS per-head sinks.
        /// Standard softmax(QK^T/√d + mask)·V if <paramref name="hasSinks"/> is false (matches
        /// <see cref="IBackend.ScaledDotProductAttention"/> semantics). With sinks, the denominator
        /// of the softmax gains a virtual <c>exp(sinks[h] - max_logit)</c> term, equivalent to
        /// prepending one virtual key-value pair with logit = sinks[h] and zero value contribution.
        /// Numerically stable via max-subtract. O(B · H · S² · D) — fine for the encoder's single
        /// forward pass on prompts ≤ 512 tokens.</summary>
        private static void ManualScaledDotProductAttention(Tensor output, Tensor q, Tensor k, Tensor v,
            Tensor mask, Tensor? sinks, int batch, int heads, int seqLen, int headDim, bool hasSinks)
        {
            float* qPtr = (float*)q.DataPointer;
            float* kPtr = (float*)k.DataPointer;
            float* vPtr = (float*)v.DataPointer;
            float* outPtr = (float*)output.DataPointer;
            float* maskPtr = (float*)mask.DataPointer;
            float* sinkPtr = hasSinks ? (float*)sinks!.DataPointer : null;

            float scale = 1.0f / MathF.Sqrt(headDim);

            // Per-(batch, head) attention.
            float[] logits = new float[seqLen];
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < heads; h++)
                {
                    long headOffset = ((long)b * heads + h) * seqLen * headDim;
                    float sinkLogit = hasSinks ? sinkPtr![h] : 0f;

                    for (int qPos = 0; qPos < seqLen; qPos++)
                    {
                        long qOff = headOffset + (long)qPos * headDim;
                        // 1) Logits: QK^T row · scale + mask.
                        float maxLogit = float.NegativeInfinity;
                        for (int kPos = 0; kPos < seqLen; kPos++)
                        {
                            long kOff = headOffset + (long)kPos * headDim;
                            float dot = 0f;
                            for (int d = 0; d < headDim; d++) dot += qPtr[qOff + d] * kPtr[kOff + d];
                            float logit = dot * scale + maskPtr[(long)qPos * seqLen + kPos];
                            logits[kPos] = logit;
                            if (logit > maxLogit) maxLogit = logit;
                        }

                        // 2) Stabilized softmax.
                        float sumExp = 0f;
                        for (int kPos = 0; kPos < seqLen; kPos++)
                        {
                            float e = MathF.Exp(logits[kPos] - maxLogit);
                            logits[kPos] = e;
                            sumExp += e;
                        }
                        // 3) Sink bucket adds exp(sinks[h] - max) to the denominator.
                        float denom = sumExp;
                        if (hasSinks) denom += MathF.Exp(sinkLogit - maxLogit);
                        float invDenom = 1.0f / denom;

                        // 4) Weighted V combine.
                        long outOff = headOffset + (long)qPos * headDim;
                        for (int d = 0; d < headDim; d++) outPtr[outOff + d] = 0f;
                        for (int kPos = 0; kPos < seqLen; kPos++)
                        {
                            float w = logits[kPos] * invDenom;
                            if (w == 0f) continue;
                            long kOff = headOffset + (long)kPos * headDim;
                            for (int d = 0; d < headDim; d++) outPtr[outOff + d] += w * vPtr[kOff + d];
                        }
                    }
                }
            }
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

        // GQA: replicate each KV head `groupSize` times.
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

        // Llama-style split-half RoPE (GPT-OSS matches Llama here).
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

        private static Tensor CastToF32IfNeeded(Tensor t) =>
            t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }
}

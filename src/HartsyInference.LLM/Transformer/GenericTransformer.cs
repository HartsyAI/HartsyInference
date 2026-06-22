using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Config-driven dense decoder transformer — the reusable spine for LLM text generation. One forward
/// path serves Qwen2, Qwen3, and Llama-lineage models; the differences (QKV bias, per-head Q/K RMSNorm,
/// decoupled head dimension, tied head) are <see cref="TransformerConfig"/> flags.
///
/// <para>Built for full GPU residency: every op is an <see cref="IBackend"/> call, no activation is read on
/// the host mid-forward, weights are the raw loaded tensors (stable references) handed to
/// <see cref="IBackend.PreloadWeights"/> so each uploads once and every projection hits the weight cache, and
/// K/V grow on-device via <see cref="KvCache"/>. Validated against the per-family reference decoders.</para>
///
/// <para>Scope: batch = 1, F32, pre-norm + SwiGLU + causal attention.</para></summary>
public sealed unsafe class GenericTransformer : IDisposable
{
    private readonly TransformerConfig _cfg;
    private readonly Layer[] _layers;
    private Tensor? _embed;
    private Tensor? _finalNorm;
    private Tensor? _lmHead;   // null when tied — reuse _embed.
    private int _disposed;

    /// <summary>The architecture this transformer was built for.</summary>
    public TransformerConfig Config => _cfg;

    /// <summary>Creates the transformer for the given architecture (weights loaded separately).</summary>
    public GenericTransformer(TransformerConfig cfg)
    {
        _cfg = cfg;
        _layers = new Layer[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++) _layers[i] = new Layer(cfg);
    }

    // F32 for the host-side embedding gather and the norm/bias vectors. Dequantizes quantized GGUF tensors
    // (rare for norms; possible for the embed table) so loading never throws on a quant dtype.
    /// <summary>Projection dispatch. Float weights always take cuBLAS <see cref="IBackend.Linear"/>. Quantized
    /// weights take the low-VRAM <see cref="IBackend.QuantizedMatMul"/> when <paramref name="lowVram"/> is set
    /// (weight stays compressed, transient dequant), else the faster <see cref="IBackend.Linear"/> path (which
    /// dequants + caches an F16 weight).</summary>
    internal static void Project(IBackend backend, Tensor output, Tensor input, Tensor weight, Tensor? bias, bool lowVram)
    {
        if (weight.DType.IsQuantized && lowVram) backend.QuantizedMatMul(output, input, weight, bias);
        else backend.Linear(output, input, weight, bias);
    }

    private static Tensor EnsureF32(Tensor t)
    {
        if (t.DType == DType.F32) return t;
        if (t.DType.IsQuantized) return HartsyInference.ModelHandler.Gguf.GgufDequantizer.Dequantize(t, DType.F32);
        return t.CastTo(DType.F32);
    }

    /// <summary>Loads weights from an HF-style key dict. <paramref name="prefix"/> is everything up to (not
    /// including) <c>embed_tokens</c> (e.g. <c>"model"</c> for a standalone checkpoint).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, string? lmHeadKey = null)
    {
        ThrowIfDisposed();
        _embed = EnsureF32(w[$"{prefix}.embed_tokens.weight"]);
        _finalNorm = EnsureF32(w[$"{prefix}.norm.weight"]);
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, $"{prefix}.layers.{i}");
        // lm_head stays in its loaded dtype (quant kept for the dequant/fused matmul path); the matmul backend
        // handles quantized weights. Only the embed table is forced to F32 (host gather).
        if (!_cfg.TieWordEmbeddings) _lmHead = w[lmHeadKey ?? "lm_head.weight"];
    }

    /// <summary>Loads a headless transformer body — decoder layers + final RMSNorm only, no
    /// <c>embed_tokens</c> / <c>lm_head</c>. Used when an outer model owns the embedding table(s) and output
    /// head(s) and drives the body via <see cref="ForwardEmbeds"/> (e.g. Sesame CSM, Kyutai, Qwen3-TTS).
    /// Do not call <see cref="Forward"/> or <see cref="ProjectLogits"/> on a headless instance.</summary>
    public void LoadWeightsHeadless(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        ThrowIfDisposed();
        _finalNorm = EnsureF32(w[$"{prefix}.norm.weight"]);
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, $"{prefix}.layers.{i}");
    }

    /// <summary>All weight tensors (stable references) for <see cref="IBackend.PreloadWeights"/> /
    /// <see cref="IBackend.FreeWeights"/>.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embed is not null) yield return _embed;
        if (_finalNorm is not null) yield return _finalNorm;
        if (_lmHead is not null) yield return _lmHead;
        foreach (Layer l in _layers)
            foreach (Tensor t in l.EnumerateWeights()) yield return t;
    }

    /// <summary>Token-IDs-in path: host embedding gather (one tiny H2D), then the resident transformer.
    /// Returns the final <c>[1, T, hidden]</c> hidden state (post final RMSNorm).</summary>
    public Tensor Forward(IBackend backend, ReadOnlySpan<int> tokenIds, int posStart, IKvCache cache)
    {
        ThrowIfDisposed();
        int t = tokenIds.Length;
        Tensor embeds = new(new TensorShape(1, t, _cfg.HiddenSize), DType.F32);
        EmbedLookup(embeds, tokenIds);
        Tensor output = ForwardEmbeds(backend, embeds, t, posStart, cache);
        embeds.Dispose();
        return output;
    }

    /// <summary>Embedding-in path. Runs decoder layers <c>[startLayer, endLayer)</c> (default the full stack)
    /// and returns the <c>[1, T, hidden]</c> hidden state, final-normed when <paramref name="applyFinalNorm"/>
    /// is true (else the raw last-layer hidden). <paramref name="posStart"/> = <see cref="IKvCache.CurrentLength"/>.
    /// The cache advances once per call (after the layers run), matching the reference decoder.</summary>
    public Tensor ForwardEmbeds(IBackend backend, Tensor embeds, int t, int posStart, IKvCache cache,
        bool applyFinalNorm = true, int startLayer = 0, int? endLayer = null)
    {
        ThrowIfDisposed();
        int last = endLayer ?? _layers.Length;
        if (startLayer < 0 || last > _layers.Length || startLayer > last)
            throw new ArgumentException($"Invalid layer range [{startLayer}, {last}) for a stack of {_layers.Length}.");

        int d = _cfg.HeadDim;

        Tensor cos = new(new TensorShape(1, t, d), DType.F32);
        Tensor sin = new(new TensorShape(1, t, d), DType.F32);
        BuildRope(cos, sin, t, posStart, d, _cfg.RopeTheta);

        try
        {
            Tensor hidden = embeds;
            bool ownsHidden = false;
            for (int i = startLayer; i < last; i++)
            {
                Tensor next = _layers[i].Forward(backend, hidden, t, posStart, cache, i, cos, sin);
                if (ownsHidden) hidden.Dispose();
                hidden = next;
                ownsHidden = true;
            }
            cache.AdvanceLength(t);

            if (applyFinalNorm)
            {
                Tensor normed = new(new TensorShape(1, t, _cfg.HiddenSize), DType.F32);
                backend.RmsNorm(normed, hidden, _finalNorm!, _cfg.RmsNormEps);
                if (ownsHidden) hidden.Dispose();
                return normed;
            }
            if (!ownsHidden)
            {
                // No layers ran (or range empty): return an owned copy so the caller never shares the input.
                Tensor copy = new(embeds.Shape, DType.F32);
                backend.CopyTo(copy, embeds);
                return copy;
            }
            return hidden;
        }
        finally
        {
            cos.Dispose();
            sin.Dispose();
        }
    }

    /// <summary>Batched decode step for continuous batching: <paramref name="embeds"/> is <c>[1, B, hidden]</c>
    /// (one decode token per active sequence), <paramref name="positions"/>[b] is sequence b's absolute position
    /// (== its KV length), and <paramref name="caches"/>[b] is sequence b's own KV cache. Returns the post-norm
    /// hidden state <c>[1, B, hidden]</c>; ready for <see cref="ProjectLogits"/> (rows = B). The heavy
    /// projections/MLP run as one batched GEMM over all B tokens; attention is per-sequence (each token attends
    /// only its own prefix). Advances each cache by 1.</summary>
    public Tensor ForwardBatchDecode(IBackend backend, Tensor embeds, ReadOnlySpan<int> positions, FixedKvCache[] caches)
    {
        ThrowIfDisposed();
        int b = (int)embeds.Shape[1];
        if (positions.Length != b || caches.Length != b)
            throw new ArgumentException($"positions ({positions.Length}) / caches ({caches.Length}) must match batch B={b}.");
        int d = _cfg.HeadDim;

        Tensor cos = new(new TensorShape(1, b, d), DType.F32);
        Tensor sin = new(new TensorShape(1, b, d), DType.F32);
        BuildRopeBatched(cos, sin, positions, d, _cfg.RopeTheta);
        try
        {
            Tensor hidden = embeds;
            bool ownsHidden = false;
            for (int i = 0; i < _layers.Length; i++)
            {
                Tensor next = _layers[i].ForwardBatchDecode(backend, hidden, b, positions, cos, sin, caches, i);
                if (ownsHidden) hidden.Dispose();
                hidden = next;
                ownsHidden = true;
            }
            for (int s = 0; s < b; s++) caches[s].AdvanceLength(1);

            Tensor normed = new(new TensorShape(1, b, _cfg.HiddenSize), DType.F32);
            backend.RmsNorm(normed, hidden, _finalNorm!, _cfg.RmsNormEps);
            if (ownsHidden) hidden.Dispose();
            return normed;
        }
        finally
        {
            cos.Dispose();
            sin.Dispose();
        }
    }

    /// <summary>Per-row RoPE table for a ragged decode batch: row b uses absolute position <paramref name="positions"/>[b]
    /// (same split-half layout as <see cref="BuildRope"/>, so both RoPE styles consume it identically).</summary>
    private static void BuildRopeBatched(Tensor cos, Tensor sin, ReadOnlySpan<int> positions, int headDim, float theta)
    {
        int half = headDim / 2;
        float* pc = (float*)cos.DataPointer;
        float* ps = (float*)sin.DataPointer;
        for (int s = 0; s < positions.Length; s++)
        {
            int pos = positions[s];
            long baseOff = (long)s * headDim;
            for (int i = 0; i < half; i++)
            {
                double freq = 1.0 / Math.Pow(theta, (double)(2 * i) / headDim);
                double angle = pos * freq;
                float c = (float)Math.Cos(angle);
                float si = (float)Math.Sin(angle);
                pc[baseOff + i] = c; pc[baseOff + i + half] = c;
                ps[baseOff + i] = si; ps[baseOff + i + half] = si;
            }
        }
    }

    /// <summary>Projects hidden <c>[1, T, hidden]</c> → logits <c>[1, T, vocab]</c> via the (tied) lm_head.</summary>
    public Tensor ProjectLogits(IBackend backend, Tensor hidden, int t)
    {
        ThrowIfDisposed();
        Tensor headW = _lmHead ?? _embed ?? throw new InvalidOperationException("weights not loaded.");
        Tensor logits = new(new TensorShape(1, t, _cfg.VocabSize), DType.F32);
        Project(backend, logits, hidden, headW, bias: null, _cfg.LowVramQuant);
        return logits;
    }

    /// <summary>Host embedding gather into <paramref name="output"/> <c>[1, T, hidden]</c>.</summary>
    public void EmbedLookup(Tensor output, ReadOnlySpan<int> tokenIds)
    {
        ThrowIfDisposed();
        int h = _cfg.HiddenSize;
        float* op = (float*)output.DataPointer;
        float* ep = (float*)_embed!.DataPointer;
        for (int s = 0; s < tokenIds.Length; s++)
        {
            int id = tokenIds[s];
            if ((uint)id >= (uint)_cfg.VocabSize)
                throw new ArgumentException($"token id {id} out of range [0, {_cfg.VocabSize}).");
            Buffer.MemoryCopy(ep + (long)id * h, op + (long)s * h, h * 4, h * 4);
        }
    }

    /// <summary>Builds duplicated-half cos/sin: <c>cos[s,i] = cos[s,i+half] = cos((posStart+s)·freq_i)</c>,
    /// <c>freq_i = theta^(-2i/headDim)</c> — the split-half rotate-half convention of
    /// <see cref="IBackend.ApplyRopeSingle"/> (shared by Qwen2 / Qwen3 / Llama).</summary>
    private static void BuildRope(Tensor cos, Tensor sin, int t, int posStart, int headDim, float theta)
    {
        int half = headDim / 2;
        float* pc = (float*)cos.DataPointer;
        float* ps = (float*)sin.DataPointer;
        for (int s = 0; s < t; s++)
        {
            int pos = posStart + s;
            long baseOff = (long)s * headDim;
            for (int i = 0; i < half; i++)
            {
                double freq = 1.0 / Math.Pow(theta, (double)(2 * i) / headDim);
                double angle = pos * freq;
                float c = (float)Math.Cos(angle);
                float si = (float)Math.Sin(angle);
                pc[baseOff + i] = c; pc[baseOff + i + half] = c;
                ps[baseOff + i] = si; ps[baseOff + i + half] = si;
            }
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(GenericTransformer));
    }

    /// <summary>Releases weight references (GPU copies are freed via <see cref="IBackend.FreeWeights"/>).</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _embed = null; _finalNorm = null; _lmHead = null;
    }

    /// <summary>One resident decoder layer: RMSNorm → GQA self-attn (optional Q/K norm, +KV cache) → residual
    /// → RMSNorm → SwiGLU → residual. All <see cref="IBackend"/> ops.</summary>
    private sealed class Layer
    {
        private readonly TransformerConfig _cfg;
        private Tensor? _inNorm, _postNorm;
        private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW;
        private Tensor? _qNorm, _kNorm;
        private Tensor? _gateW, _upW, _downW;

        public Layer(TransformerConfig cfg) => _cfg = cfg;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
        {
            // Norms and biases are forced to F32; projection weights keep their loaded dtype (quant or float)
            // so the matmul backend can dequant-on-the-fly (or run a fused quantized GEMV) without an F32 copy.
            _inNorm = EnsureF32(w[$"{prefix}.input_layernorm.weight"]);
            _postNorm = EnsureF32(w[$"{prefix}.post_attention_layernorm.weight"]);
            _qW = w[$"{prefix}.self_attn.q_proj.weight"];
            _kW = w[$"{prefix}.self_attn.k_proj.weight"];
            _vW = w[$"{prefix}.self_attn.v_proj.weight"];
            _oW = w[$"{prefix}.self_attn.o_proj.weight"];
            if (_cfg.AttentionBias)
            {
                _qB = EnsureF32(w[$"{prefix}.self_attn.q_proj.bias"]);
                _kB = EnsureF32(w[$"{prefix}.self_attn.k_proj.bias"]);
                _vB = EnsureF32(w[$"{prefix}.self_attn.v_proj.bias"]);
            }
            if (_cfg.QkNorm)
            {
                _qNorm = EnsureF32(w[$"{prefix}.self_attn.q_norm.weight"]);
                _kNorm = EnsureF32(w[$"{prefix}.self_attn.k_norm.weight"]);
            }
            _gateW = w[$"{prefix}.mlp.gate_proj.weight"];
            _upW = w[$"{prefix}.mlp.up_proj.weight"];
            _downW = w[$"{prefix}.mlp.down_proj.weight"];
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] all = [_inNorm, _postNorm, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _qNorm, _kNorm, _gateW, _upW, _downW];
            foreach (Tensor? t in all) if (t is not null) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor hidden, int t, int posStart,
            IKvCache cache, int layerIndex, Tensor cos, Tensor sin)
        {
            int h = _cfg.HiddenSize;
            int hq = _cfg.NumHeads;
            int hkv = _cfg.NumKvHeads;
            int d = _cfg.HeadDim;
            int group = _cfg.KvGroup;
            TensorShape flat = new(1, t, h);

            Tensor pre = new(flat, DType.F32);
            backend.RmsNorm(pre, hidden, _inNorm!, _cfg.RmsNormEps);

            // Q/K/V projections written straight into head layout [1, T, heads, D] (Linear derives N from the
            // weight, not the output rank, so the byte-identical reshape is free and stays resident).
            Tensor q = new(new TensorShape(1, t, hq, d), DType.F32);
            Tensor k = new(new TensorShape(1, t, hkv, d), DType.F32);
            Tensor v = new(new TensorShape(1, t, hkv, d), DType.F32);
            Project(backend, q, pre, _qW!, _qB, _cfg.LowVramQuant);
            Project(backend, k, pre, _kW!, _kB, _cfg.LowVramQuant);
            Project(backend, v, pre, _vW!, _vB, _cfg.LowVramQuant);
            pre.Dispose();

            // Optional per-head Q/K RMSNorm over the head_dim (rows = T·heads), before RoPE (Qwen3).
            if (_cfg.QkNorm)
            {
                Tensor qN = new(new TensorShape(1, t, hq, d), DType.F32);
                Tensor kN = new(new TensorShape(1, t, hkv, d), DType.F32);
                backend.RmsNorm(qN, q, _qNorm!, _cfg.RmsNormEps);
                backend.RmsNorm(kN, k, _kNorm!, _cfg.RmsNormEps);
                q.Dispose(); k.Dispose();
                q = qN; k = kN;
            }

            if (_cfg.Rope == RopeStyle.Interleaved)
            {
                backend.ApplyRopeInterleaved(q, cos, sin);
                backend.ApplyRopeInterleaved(k, cos, sin);
            }
            else
            {
                backend.ApplyRopeSingle(q, cos, sin);
                backend.ApplyRopeSingle(k, cos, sin);
            }

            Tensor qMh = new(new TensorShape(1, hq, t, d), DType.F32);
            Tensor kMh = new(new TensorShape(1, hkv, t, d), DType.F32);
            Tensor vMh = new(new TensorShape(1, hkv, t, d), DType.F32);
            backend.Permute0213(qMh, q, t, hq, d);
            backend.Permute0213(kMh, k, t, hkv, d);
            backend.Permute0213(vMh, v, t, hkv, d);
            q.Dispose(); k.Dispose(); v.Dispose();

            cache.AppendStep(backend, layerIndex, kMh, vMh);
            kMh.Dispose(); vMh.Dispose();
            Tensor kFull = cache.KeyPrefix(layerIndex);   // [1, Hkv, kvLen, D] (cache-owned)
            Tensor vFull = cache.ValuePrefix(layerIndex);

            // FlashAttention: GQA-aware (no K/V replication) online-softmax, causal via the absolute query
            // offset (decode t=1 naturally attends the whole prefix; prefill t>1 is per-row causal). No score
            // matrix, no causal-mask tensor.
            // kFull/vFull may be a fixed-capacity buffer whose seq stride exceeds the valid length, so pass the
            // valid key count (posStart + t) explicitly; the kernel reads the stride from the tensor shape.
            int kvLen = posStart + t;
            float scale = 1f / MathF.Sqrt(d);
            Tensor attn = new(new TensorShape(1, hq, t, d), DType.F32);
            backend.FlashAttention(attn, qMh, kFull, vFull, kvLen, group, causal: true, qOffset: posStart, scale);
            qMh.Dispose();

            Tensor attnFlat = new(new TensorShape(1, t, _cfg.QDim), DType.F32);
            backend.Permute0213(attnFlat, attn, hq, t, d);
            attn.Dispose();
            Tensor attnOut = new(flat, DType.F32);
            Project(backend, attnOut, attnFlat, _oW!, null, _cfg.LowVramQuant);
            attnFlat.Dispose();

            Tensor afterAttn = new(flat, DType.F32);
            backend.Add(afterAttn, hidden, attnOut);
            attnOut.Dispose();

            Tensor preMlp = new(flat, DType.F32);
            backend.RmsNorm(preMlp, afterAttn, _postNorm!, _cfg.RmsNormEps);

            TensorShape ff = new(1, t, _cfg.IntermediateSize);
            Tensor gate = new(ff, DType.F32);
            Project(backend, gate, preMlp, _gateW!, null, _cfg.LowVramQuant);
            Tensor gateAct = new(ff, DType.F32);
            backend.Silu(gateAct, gate);
            gate.Dispose();
            Tensor up = new(ff, DType.F32);
            Project(backend, up, preMlp, _upW!, null, _cfg.LowVramQuant);
            preMlp.Dispose();
            Tensor comb = new(ff, DType.F32);
            backend.Mul(comb, gateAct, up);
            gateAct.Dispose(); up.Dispose();
            Tensor mlpOut = new(flat, DType.F32);
            Project(backend, mlpOut, comb, _downW!, null, _cfg.LowVramQuant);
            comb.Dispose();

            Tensor result = new(flat, DType.F32);
            backend.Add(result, afterAttn, mlpOut);
            afterAttn.Dispose(); mlpOut.Dispose();
            return result;
        }

        /// <summary>Batched decode (one token per sequence): projections/MLP run as a single GEMM over all B
        /// tokens; attention is looped per sequence (each token attends only its own KV prefix via the scalar
        /// FlashAttention). <paramref name="hidden"/> is <c>[1, B, hidden]</c>.</summary>
        public Tensor ForwardBatchDecode(IBackend backend, Tensor hidden, int b, ReadOnlySpan<int> positions,
            Tensor cos, Tensor sin, FixedKvCache[] caches, int layerIndex)
        {
            int h = _cfg.HiddenSize;
            int hq = _cfg.NumHeads;
            int hkv = _cfg.NumKvHeads;
            int d = _cfg.HeadDim;
            int group = _cfg.KvGroup;
            TensorShape flat = new(1, b, h);

            Tensor pre = new(flat, DType.F32);
            backend.RmsNorm(pre, hidden, _inNorm!, _cfg.RmsNormEps);

            Tensor q = new(new TensorShape(1, b, hq, d), DType.F32);
            Tensor k = new(new TensorShape(1, b, hkv, d), DType.F32);
            Tensor v = new(new TensorShape(1, b, hkv, d), DType.F32);
            Project(backend, q, pre, _qW!, _qB, _cfg.LowVramQuant);
            Project(backend, k, pre, _kW!, _kB, _cfg.LowVramQuant);
            Project(backend, v, pre, _vW!, _vB, _cfg.LowVramQuant);
            pre.Dispose();

            if (_cfg.QkNorm)
            {
                Tensor qN = new(new TensorShape(1, b, hq, d), DType.F32);
                Tensor kN = new(new TensorShape(1, b, hkv, d), DType.F32);
                backend.RmsNorm(qN, q, _qNorm!, _cfg.RmsNormEps);
                backend.RmsNorm(kN, k, _kNorm!, _cfg.RmsNormEps);
                q.Dispose(); k.Dispose();
                q = qN; k = kN;
            }

            // RoPE per row (each row b is one token at its own absolute position).
            if (_cfg.Rope == RopeStyle.Interleaved)
            {
                backend.ApplyRopeInterleaved(q, cos, sin);
                backend.ApplyRopeInterleaved(k, cos, sin);
            }
            else
            {
                backend.ApplyRopeSingle(q, cos, sin);
                backend.ApplyRopeSingle(k, cos, sin);
            }

            float scale = 1f / MathF.Sqrt(d);
            Tensor[] segs = new Tensor[b];
            for (int s = 0; s < b; s++)
            {
                // Slice this sequence's single token out of the batched projections. q rows are head-major
                // [B, Hq, D]; [1,Hq,1,D] and [1,1,Hq,D] share memory (Tq=1), so the slice doubles as the
                // multi-head layout FlashAttention wants and the [1,1,Hq,D] piece Concat reassembles.
                Tensor qMh = new(new TensorShape(1, hq, 1, d), DType.F32);
                Tensor kMh = new(new TensorShape(1, hkv, 1, d), DType.F32);
                Tensor vMh = new(new TensorShape(1, hkv, 1, d), DType.F32);
                backend.SliceRows(qMh, q, s * hq);
                backend.SliceRows(kMh, k, s * hkv);
                backend.SliceRows(vMh, v, s * hkv);

                FixedKvCache cache = caches[s];
                cache.AppendStep(backend, layerIndex, kMh, vMh);
                kMh.Dispose(); vMh.Dispose();
                int kvLen = cache.CurrentLength + 1;   // append did not advance; +1 for the just-written token
                Tensor attnSeg = new(new TensorShape(1, 1, hq, d), DType.F32);
                backend.FlashAttention(attnSeg, qMh, cache.KeyPrefix(layerIndex), cache.ValuePrefix(layerIndex),
                    kvLen, group, causal: true, qOffset: cache.CurrentLength, scale);
                qMh.Dispose();
                segs[s] = attnSeg;
            }

            Tensor attnConcat = new(new TensorShape(1, b, hq, d), DType.F32);
            backend.Concat(attnConcat, segs, dim: 1);
            foreach (Tensor seg in segs) seg.Dispose();

            Tensor attnOut = new(flat, DType.F32);   // o_proj reads attnConcat as [B, QDim]
            Project(backend, attnOut, attnConcat, _oW!, null, _cfg.LowVramQuant);
            attnConcat.Dispose();

            Tensor afterAttn = new(flat, DType.F32);
            backend.Add(afterAttn, hidden, attnOut);
            attnOut.Dispose();

            Tensor preMlp = new(flat, DType.F32);
            backend.RmsNorm(preMlp, afterAttn, _postNorm!, _cfg.RmsNormEps);

            TensorShape ff = new(1, b, _cfg.IntermediateSize);
            Tensor gate = new(ff, DType.F32);
            Project(backend, gate, preMlp, _gateW!, null, _cfg.LowVramQuant);
            Tensor gateAct = new(ff, DType.F32);
            backend.Silu(gateAct, gate);
            gate.Dispose();
            Tensor up = new(ff, DType.F32);
            Project(backend, up, preMlp, _upW!, null, _cfg.LowVramQuant);
            preMlp.Dispose();
            Tensor comb = new(ff, DType.F32);
            backend.Mul(comb, gateAct, up);
            gateAct.Dispose(); up.Dispose();
            Tensor mlpOut = new(flat, DType.F32);
            Project(backend, mlpOut, comb, _downW!, null, _cfg.LowVramQuant);
            comb.Dispose();

            Tensor result = new(flat, DType.F32);
            backend.Add(result, afterAttn, mlpOut);
            afterAttn.Dispose(); mlpOut.Dispose();
            return result;
        }
    }
}

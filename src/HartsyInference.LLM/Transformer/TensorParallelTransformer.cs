using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Megatron-style tensor-parallel dense decoder (v1, correctness-first): every rank holds the FULL layer stack but only its slice of each layer — a contiguous range of Q/KV heads plus gate/up FFN columns (column-parallel), with o_proj and down_proj split along IN (row-parallel: each rank's partial output is summed across ranks) via exactly two <see cref="ICollectiveComm.AllReduceSum"/> seams per layer. Final norm, lm_head, and the (host-gathered) embedding table live on rank 0; each rank's KV cache holds only its own KV-head slice.</summary>
/// <remarks>
/// <para>Execution model (v1): ONE driving thread runs the per-rank partial loops sequentially between the collectives — no rendezvous machinery, trivially testable on CPU backends with <see cref="HostStagedComm"/>. Upgrading to one thread per rank (a RankRunner meeting at the collectives) is a deliberate follow-up; the phase-split structure here is shaped so that upgrade changes the driver only. Graph decode, speculative decoding, and the batching scheduler are out of scope — none of those paths can reach this type.</para>
/// <para>Scope: the dense pre-norm GQA/RoPE + gated-FFN decoder shape (Llama/Qwen2/Qwen3, incl. QKV bias and per-head QK-norm). Everything else is refused loudly at construction — see <c>ValidateConfig</c>.</para>
/// </remarks>
public sealed unsafe class TensorParallelTransformer : IDisposable
{
    private readonly TransformerConfig _cfg;
    private readonly TpPlacement _placement;
    private readonly RankLayer[][] _layers;   // [rank][layer]
    private readonly List<Tensor> _owned = [];
    private Tensor? _embed;       // F32 table for the host embedding gather (rank-0-owned)
    private bool _ownsEmbed;
    private Tensor? _head;        // lm_head projection weight (borrowed checkpoint reference; may alias _embed when tied)
    private Tensor? _finalNorm;   // owned F32 copy, rank 0
    private int _disposed;

    /// <summary>The architecture this transformer was built for.</summary>
    public TransformerConfig Config => _cfg;

    /// <summary>Tensor-parallel degree (rank count).</summary>
    public int Degree => _placement.Degree;

    /// <summary>Validates the architecture and geometry against <paramref name="placement"/>'s degree (weights loaded separately via <see cref="LoadWeights"/>).</summary>
    public TensorParallelTransformer(TransformerConfig cfg, TpPlacement placement)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        ArgumentNullException.ThrowIfNull(placement);
        ValidateConfig(cfg, placement.Degree);
        _cfg = cfg;
        _placement = placement;
        _layers = new RankLayer[placement.Degree][];
        for (int r = 0; r < placement.Degree; r++)
        {
            _layers[r] = new RankLayer[cfg.NumLayers];
        }
    }

    /// <summary>Refuses architectures outside the v1 dense shape and geometries that don't tile: every rank needs a whole number of Q heads, KV heads, and FFN columns.</summary>
    private static void ValidateConfig(TransformerConfig cfg, int degree)
    {
        List<string> unsupported = [];
        if (cfg.Mla is not null) unsupported.Add("MLA");
        if (cfg.Moe is not null) unsupported.Add("MoE");
        if (cfg.CrossAttnLayers.Count > 0) unsupported.Add("cross-attention layers");
        if (cfg.AttnSink) unsupported.Add("attention sink");
        if (cfg.AlibiMaxBias > 0f) unsupported.Add("ALiBi");
        if (cfg.ParallelResidual) unsupported.Add("parallel residual");
        if (cfg.AbsolutePositionEmbeddings) unsupported.Add("absolute position embeddings");
        if (cfg.EmbeddingLayerNorm) unsupported.Add("embedding LayerNorm");
        if (cfg.NormPlacement != NormPlacement.PreNorm) unsupported.Add("post-norm placement");
        if (cfg.UseLayerNorm) unsupported.Add("LayerNorm");
        if (cfg.SandwichNorm) unsupported.Add("sandwich norms");
        if (cfg.QkNorm && cfg.QkNormFullDim) unsupported.Add("full-dim QK-norm");
        if (cfg.VNorm) unsupported.Add("V-norm");
        if (cfg.PerLayerEmbeddingDim > 0) unsupported.Add("per-layer embeddings");
        if (cfg.KvSharedFromLayer > 0) unsupported.Add("KV sharing");
        if (!cfg.GatedFfn) unsupported.Add("non-gated FFN");
        if (cfg.FfnBias) unsupported.Add("FFN bias");
        if (cfg.HeadDimSwa > 0) unsupported.Add("per-layer SWA head dim");
        if (cfg.RopeLocalTheta > 0) unsupported.Add("dual local/global RoPE");
        if (cfg.SlidingWindow > 0) unsupported.Add("sliding-window attention");
        if (cfg.NoRopeOnGlobalLayers) unsupported.Add("NoPE global layers");
        if (cfg.ResidualMultiplier != 1f) unsupported.Add("residual multiplier");
        if (unsupported.Count > 0)
        {
            throw new NotSupportedException(
                "Tensor parallelism v1 supports the dense pre-norm GQA/RoPE + gated-FFN decoder shape only; "
                + $"unsupported here: {string.Join(", ", unsupported)}.");
        }
        if (cfg.NumHeads % degree != 0)
        {
            throw new ArgumentException(
                $"NumHeads ({cfg.NumHeads}) must divide evenly by the TP degree ({degree}) — every rank owns a whole Q-head range.",
                nameof(degree));
        }
        if (cfg.NumKvHeads % degree != 0)
        {
            throw new ArgumentException(
                $"NumKvHeads ({cfg.NumKvHeads}) must divide evenly by the TP degree ({degree}) — every rank's KV cache holds a whole KV-head slice.",
                nameof(degree));
        }
        if (cfg.IntermediateSize % degree != 0)
        {
            throw new ArgumentException(
                $"IntermediateSize ({cfg.IntermediateSize}) must divide evenly by the TP degree ({degree}) — every rank owns a whole gate/up column range.",
                nameof(degree));
        }
    }

    /// <summary>Loads and partitions weights from an HF-style key dict (same keys/prefix contract as <see cref="GenericTransformer.LoadWeights"/>). Per-rank slices are COPIED sub-tensors (not views) so each tensor has exactly one owning rank backend, letting a caller free the full checkpoint tensors afterwards. Norms are copied per rank; the embedding table, final norm, and lm_head stay rank-0-only since the residual stream is replicated after the last reduce.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, string? lmHeadKey = null)
    {
        ThrowIfDisposed();
        int deg = Degree;
        bool addOne = _cfg.RmsNormAddOne;
        Tensor embedRaw = w[$"{prefix}.embed_tokens.weight"];
        _embed = GenericTransformer.EnsureF32(embedRaw);
        _ownsEmbed = !ReferenceEquals(_embed, embedRaw);
        // Same head-dtype choice as GenericTransformer.LoadWeights: a tied quant/16-bit embed doubles as the
        // fused-GEMV head; a tied F32 embed is the head directly; untied models ship their own lm_head.
        if (_cfg.TieWordEmbeddings)
        {
            _head = embedRaw.DType.IsQuantized || embedRaw.DType == DType.BF16 || embedRaw.DType == DType.F16
                ? embedRaw
                : _embed;
        }
        else
        {
            _head = w[lmHeadKey ?? "lm_head.weight"];
        }
        _finalNorm = Own(CopyNormF32(w[$"{prefix}.norm.weight"], addOne));

        for (int l = 0; l < _cfg.NumLayers; l++)
        {
            string p = $"{prefix}.layers.{l}";
            Tensor qW = w[$"{p}.self_attn.q_proj.weight"];
            Tensor kW = w[$"{p}.self_attn.k_proj.weight"];
            Tensor vW = w[$"{p}.self_attn.v_proj.weight"];
            Tensor oW = w[$"{p}.self_attn.o_proj.weight"];
            Tensor gateW = w[$"{p}.mlp.gate_proj.weight"];
            Tensor upW = w[$"{p}.mlp.up_proj.weight"];
            Tensor downW = w[$"{p}.mlp.down_proj.weight"];
            for (int r = 0; r < deg; r++)
            {
                RankLayer rl = new()
                {
                    InNorm = Own(CopyNormF32(w[$"{p}.input_layernorm.weight"], addOne)),
                    PostNorm = Own(CopyNormF32(w[$"{p}.post_attention_layernorm.weight"], addOne)),
                    QW = Own(SliceOutRows(qW, r, deg, $"{p}.self_attn.q_proj.weight")),
                    KW = Own(SliceOutRows(kW, r, deg, $"{p}.self_attn.k_proj.weight")),
                    VW = Own(SliceOutRows(vW, r, deg, $"{p}.self_attn.v_proj.weight")),
                    OW = Own(SliceInCols(oW, r, deg, $"{p}.self_attn.o_proj.weight")),
                    GateW = Own(SliceOutRows(gateW, r, deg, $"{p}.mlp.gate_proj.weight")),
                    UpW = Own(SliceOutRows(upW, r, deg, $"{p}.mlp.up_proj.weight")),
                    DownW = Own(SliceInCols(downW, r, deg, $"{p}.mlp.down_proj.weight")),
                };
                if (_cfg.AttentionBias)
                {
                    rl.QB = Own(SliceBiasRows(w[$"{p}.self_attn.q_proj.bias"], r, deg));
                    rl.KB = Own(SliceBiasRows(w[$"{p}.self_attn.k_proj.bias"], r, deg));
                    rl.VB = Own(SliceBiasRows(w[$"{p}.self_attn.v_proj.bias"], r, deg));
                    // Row-parallel bias: the AllReduce SUMS the per-rank partials, so a replicated o bias would
                    // be added `degree` times — rank 0 alone carries it and the sum contains it exactly once.
                    if (r == 0 && w.TryGetValue($"{p}.self_attn.o_proj.bias", out Tensor? ob))
                    {
                        rl.OB = Own(CopyNormF32(ob, addOne: false));
                    }
                }
                if (_cfg.QkNorm)
                {
                    rl.QNorm = Own(CopyNormF32(w[$"{p}.self_attn.q_norm.weight"], addOne));
                    rl.KNorm = Own(CopyNormF32(w[$"{p}.self_attn.k_norm.weight"], addOne));
                }
                _layers[r][l] = rl;
            }
        }
    }

    /// <summary>One device KV cache per rank, sized to that rank's KV-head slice (<c>NumKvHeads / Degree</c> heads).</summary>
    public KvCache[] CreateKvCaches(int batch = 1)
    {
        ThrowIfDisposed();
        KvCache[] caches = new KvCache[Degree];
        for (int r = 0; r < Degree; r++)
        {
            caches[r] = new KvCache(_cfg.NumLayers, batch, _cfg.NumKvHeads / Degree, _cfg.HeadDim);
        }
        return caches;
    }

    /// <summary>Rank <paramref name="rank"/>'s resident weight set (its sliced projections + per-rank norm copies; rank 0 additionally the final norm and lm_head) for <see cref="IBackend.PreloadWeights"/> on that rank's backend. The union over all ranks covers every device-consumed tensor exactly once — the embedding table is host-gathered and never uploaded (unless it doubles as the tied F32 head).</summary>
    public IEnumerable<Tensor> EnumerateRankWeights(int rank)
    {
        ThrowIfDisposed();
        if (rank < 0 || rank >= Degree)
        {
            throw new ArgumentOutOfRangeException(nameof(rank), rank, $"rank must be in [0, {Degree}).");
        }
        if (rank == 0)
        {
            if (_finalNorm is not null) yield return _finalNorm;
            if (_head is not null) yield return _head;
        }
        foreach (RankLayer rl in _layers[rank])
        {
            yield return rl.InNorm;
            yield return rl.PostNorm;
            yield return rl.QW;
            yield return rl.KW;
            yield return rl.VW;
            yield return rl.OW;
            yield return rl.GateW;
            yield return rl.UpW;
            yield return rl.DownW;
            if (rl.QB is not null) yield return rl.QB;
            if (rl.KB is not null) yield return rl.KB;
            if (rl.VB is not null) yield return rl.VB;
            if (rl.OB is not null) yield return rl.OB;
            if (rl.QNorm is not null) yield return rl.QNorm;
            if (rl.KNorm is not null) yield return rl.KNorm;
        }
    }

    /// <summary>Token-IDs-in convenience over <see cref="ForwardEmbedsTp"/> (host embedding gather on rank 0's table), mirroring <see cref="GenericTransformer.Forward"/>.</summary>
    public Tensor ForwardTp(ReadOnlySpan<int> tokenIds, int posStart, IReadOnlyList<IKvCache> caches)
    {
        ThrowIfDisposed();
        int t = tokenIds.Length;
        Tensor embeds = new(new TensorShape(1, t, _cfg.HiddenSize), DType.F32);
        EmbedLookup(embeds, tokenIds);
        Tensor output = ForwardEmbedsTp(embeds, t, posStart, caches);
        embeds.Dispose();
        return output;
    }

    /// <summary>Runs the full stack tensor-parallel and returns the final-normed <c>[1, T, hidden]</c> hidden state on rank 0's backend; <paramref name="caches"/> is one per-rank KV cache (see <see cref="CreateKvCaches"/>), each advancing by <paramref name="t"/> exactly once per call. The input <paramref name="embeds"/> is host-materialized once and cloned per rank, so it may live on any backend; the caller keeps ownership.</summary>
    public Tensor ForwardEmbedsTp(Tensor embeds, int t, int posStart, IReadOnlyList<IKvCache> caches)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(embeds);
        ArgumentNullException.ThrowIfNull(caches);
        if (_finalNorm is null)
        {
            throw new InvalidOperationException("weights not loaded.");
        }
        if (caches.Count != Degree)
        {
            throw new ArgumentException($"Expected one KV cache per rank ({Degree}), got {caches.Count}.", nameof(caches));
        }
        int deg = Degree;
        int h = _cfg.HiddenSize;
        int d = _cfg.HeadDim;
        TensorShape flat = new(1, t, h);
        if (embeds.ElementCount != flat.ElementCount)
        {
            throw new ArgumentException($"embeds must be [1, {t}, {h}], got {embeds.Shape}.", nameof(embeds));
        }
        // This read IS the input handoff: it host-materializes a device-resident input so every rank's clone
        // uploads from the authoritative host copy (same pattern as ForwardEmbedsStaged's stage boundary).
        _ = embeds.DataPointer;

        Tensor[] hidden = new Tensor[deg];
        Tensor[] cos = new Tensor[deg];
        Tensor[] sin = new Tensor[deg];
        Tensor?[] partial = new Tensor?[deg];
        Tensor?[] afterAttn = new Tensor?[deg];
        try
        {
            for (int r = 0; r < deg; r++)
            {
                hidden[r] = HostClone(embeds);
                // Per-rank RoPE tables (identical host-built values) so no tensor is consumed by two backends.
                cos[r] = new Tensor(new TensorShape(1, t, d), DType.F32);
                sin[r] = new Tensor(new TensorShape(1, t, d), DType.F32);
                GenericTransformer.BuildRope(cos[r], sin[r], t, posStart, d, _cfg.RotaryDim, _cfg.RopeTheta, _cfg.RopeScaling);
            }

            for (int l = 0; l < _cfg.NumLayers; l++)
            {
                for (int r = 0; r < deg; r++)
                {
                    partial[r] = AttnPartial(r, l, hidden[r], t, posStart, caches[r], cos[r], sin[r]);
                }
                _placement.Comm.AllReduceSum(NonNull(partial));
                for (int r = 0; r < deg; r++)
                {
                    IBackend backend = _placement.RankBackends[r];
                    afterAttn[r] = new Tensor(flat, DType.F32);
                    backend.Add(afterAttn[r]!, hidden[r], partial[r]!);
                    partial[r]!.Dispose();
                    partial[r] = MlpPartial(r, l, afterAttn[r]!, t);
                }
                _placement.Comm.AllReduceSum(NonNull(partial));
                for (int r = 0; r < deg; r++)
                {
                    IBackend backend = _placement.RankBackends[r];
                    Tensor next = new(flat, DType.F32);
                    backend.Add(next, afterAttn[r]!, partial[r]!);
                    afterAttn[r]!.Dispose();
                    afterAttn[r] = null;
                    partial[r]!.Dispose();
                    partial[r] = null;
                    hidden[r].Dispose();
                    hidden[r] = next;
                }
            }
            for (int r = 0; r < deg; r++)
            {
                caches[r].AdvanceLength(t);
            }

            Tensor normed = new(flat, DType.F32);
            _placement.RankBackends[0].RmsNorm(normed, hidden[0], _finalNorm, _cfg.RmsNormEps);
            return normed;
        }
        finally
        {
            for (int r = 0; r < deg; r++)
            {
                hidden[r]?.Dispose();
                cos[r]?.Dispose();
                sin[r]?.Dispose();
                partial[r]?.Dispose();
                afterAttn[r]?.Dispose();
            }
        }
    }

    /// <summary>Projects rank 0's hidden <c>[1, T, hidden]</c> → logits <c>[1, T, vocab]</c> via the (tied) lm_head on rank 0's backend, with Granite logit scale and the Gemma-2 final soft-cap.</summary>
    public Tensor ProjectLogits(Tensor hidden, int t)
    {
        ThrowIfDisposed();
        Tensor headW = _head ?? throw new InvalidOperationException("weights not loaded.");
        IBackend backend = _placement.RankBackends[0];
        Tensor logits = new(new TensorShape(1, t, _cfg.VocabSize), DType.F32);
        GenericTransformer.Project(backend, logits, hidden, headW, bias: null, _cfg.LowVramQuant);
        if (_cfg.LogitScale != 1f)
        {
            backend.Scale(logits, logits, 1f / _cfg.LogitScale);
        }
        if (_cfg.FinalLogitSoftcap > 0f)
        {
            float cap = _cfg.FinalLogitSoftcap;
            backend.Scale(logits, logits, 1f / cap);
            backend.Tanh(logits, logits);
            backend.Scale(logits, logits, cap);
        }
        return logits;
    }

    /// <summary>Host embedding gather into <paramref name="output"/> <c>[1, T, hidden]</c> (rank-0 table), mirroring <see cref="GenericTransformer.EmbedLookup"/> including <see cref="TransformerConfig.EmbeddingScale"/>.</summary>
    public void EmbedLookup(Tensor output, ReadOnlySpan<int> tokenIds)
    {
        ThrowIfDisposed();
        if (_embed is null)
        {
            throw new InvalidOperationException("weights not loaded.");
        }
        int h = _cfg.HiddenSize;
        float* op = (float*)output.DataPointer;
        float* ep = (float*)_embed.DataPointer;
        for (int s = 0; s < tokenIds.Length; s++)
        {
            int id = tokenIds[s];
            if ((uint)id >= (uint)_cfg.VocabSize)
            {
                throw new ArgumentException($"token id {id} out of range [0, {_cfg.VocabSize}).");
            }
            Buffer.MemoryCopy(ep + (long)id * h, op + (long)s * h, h * 4, h * 4);
        }
        if (_cfg.EmbeddingScale != 1f)
        {
            float scale = _cfg.EmbeddingScale;
            long total = (long)tokenIds.Length * h;
            for (long i = 0; i < total; i++)
            {
                op[i] *= scale;
            }
        }
    }

    /// <summary>Rank <paramref name="rank"/>'s attention partial for one layer: pre-norm → its Q/KV head slice's projections (+QK-norm, RoPE, per-rank KV append, FlashAttention over its heads) → its o_proj column slice. The return is a PARTIAL <c>[1, T, hidden]</c> sum — callers AllReduce before the residual add.</summary>
    private Tensor AttnPartial(int rank, int layer, Tensor hidden, int t, int posStart, IKvCache cache, Tensor cos, Tensor sin)
    {
        RankLayer rl = _layers[rank][layer];
        IBackend backend = _placement.RankBackends[rank];
        int deg = Degree;
        int hqR = _cfg.NumHeads / deg;
        int hkvR = _cfg.NumKvHeads / deg;
        int d = _cfg.HeadDim;
        TensorShape flat = new(1, t, _cfg.HiddenSize);

        Tensor pre = new(flat, DType.F32);
        backend.RmsNorm(pre, hidden, rl.InNorm, _cfg.RmsNormEps);

        Tensor q = new(new TensorShape(1, t, hqR, d), DType.F32);
        Tensor k = new(new TensorShape(1, t, hkvR, d), DType.F32);
        Tensor v = new(new TensorShape(1, t, hkvR, d), DType.F32);
        GenericTransformer.Project(backend, q, pre, rl.QW, rl.QB, _cfg.LowVramQuant);
        GenericTransformer.Project(backend, k, pre, rl.KW, rl.KB, _cfg.LowVramQuant);
        GenericTransformer.Project(backend, v, pre, rl.VW, rl.VB, _cfg.LowVramQuant);
        pre.Dispose();

        // Per-head QK-norm: the [headDim] weight is head-shared, so the replicated copy serves every rank's
        // head slice unchanged.
        if (_cfg.QkNorm)
        {
            Tensor qN = new(new TensorShape(1, t, hqR, d), DType.F32);
            backend.RmsNorm(qN, q, rl.QNorm!, _cfg.RmsNormEps);
            q.Dispose();
            q = qN;
            Tensor kN = new(new TensorShape(1, t, hkvR, d), DType.F32);
            backend.RmsNorm(kN, k, rl.KNorm!, _cfg.RmsNormEps);
            k.Dispose();
            k = kN;
        }

        if (_cfg.Rope == RopeStyle.Interleaved)
        {
            backend.ApplyRopeInterleaved(q, cos, sin, _cfg.RotaryDim);
            backend.ApplyRopeInterleaved(k, cos, sin, _cfg.RotaryDim);
        }
        else
        {
            backend.ApplyRopeSingle(q, cos, sin, _cfg.RotaryDim);
            backend.ApplyRopeSingle(k, cos, sin, _cfg.RotaryDim);
        }

        Tensor qMh = new(new TensorShape(1, hqR, t, d), DType.F32);
        backend.Permute0213(qMh, q, t, hqR, d);
        q.Dispose();
        Tensor kMh = new(new TensorShape(1, hkvR, t, d), DType.F32);
        Tensor vMh = new(new TensorShape(1, hkvR, t, d), DType.F32);
        backend.Permute0213(kMh, k, t, hkvR, d);
        backend.Permute0213(vMh, v, t, hkvR, d);
        k.Dispose();
        v.Dispose();
        cache.AppendStep(backend, layer, kMh, vMh);
        kMh.Dispose();
        vMh.Dispose();
        Tensor kFull = cache.KeyPrefix(layer);
        Tensor vFull = cache.ValuePrefix(layer);

        // GQA group is degree-invariant: (hq/deg) / (hkv/deg) == KvGroup, so each rank's Q-head slice reads
        // exactly its own KV-head slice — no cross-rank attention traffic in the dense TP shape.
        int kvLen = posStart + t;
        Tensor attn = new(new TensorShape(1, hqR, t, d), DType.F32);
        backend.FlashAttention(attn, qMh, kFull, vFull, kvLen, _cfg.KvGroup, causal: true, qOffset: posStart, _cfg.AttnScale);
        qMh.Dispose();

        Tensor attnFlat = new(new TensorShape(1, t, hqR * d), DType.F32);
        backend.Permute0213(attnFlat, attn, hqR, t, d);
        attn.Dispose();
        Tensor outp = new(flat, DType.F32);
        GenericTransformer.Project(backend, outp, attnFlat, rl.OW, rl.OB, _cfg.LowVramQuant);
        attnFlat.Dispose();
        return outp;
    }

    /// <summary>Rank <paramref name="rank"/>'s FFN partial for one layer: pre-MLP norm → its gate/up column slice → activation·up → its down_proj column slice. Returns a PARTIAL <c>[1, T, hidden]</c> sum — callers AllReduce before the residual add.</summary>
    private Tensor MlpPartial(int rank, int layer, Tensor afterAttn, int t)
    {
        RankLayer rl = _layers[rank][layer];
        IBackend backend = _placement.RankBackends[rank];
        TensorShape flat = new(1, t, _cfg.HiddenSize);
        TensorShape ff = new(1, t, (int)rl.UpW.Shape[0]);

        Tensor preMlp = new(flat, DType.F32);
        backend.RmsNorm(preMlp, afterAttn, rl.PostNorm, _cfg.RmsNormEps);
        Tensor gate = new(ff, DType.F32);
        Tensor up = new(ff, DType.F32);
        GenericTransformer.Project(backend, gate, preMlp, rl.GateW, bias: null, _cfg.LowVramQuant);
        GenericTransformer.Project(backend, up, preMlp, rl.UpW, bias: null, _cfg.LowVramQuant);
        preMlp.Dispose();
        Tensor act = new(ff, DType.F32);
        Activate(backend, act, gate);
        gate.Dispose();
        Tensor comb = new(ff, DType.F32);
        backend.Mul(comb, act, up);
        act.Dispose();
        up.Dispose();
        Tensor outp = new(flat, DType.F32);
        GenericTransformer.Project(backend, outp, comb, rl.DownW, bias: null, _cfg.LowVramQuant);
        comb.Dispose();
        return outp;
    }

    /// <summary>Gate activation (elementwise, so column-parallel slices activate identically to the full tensor) — mirrors <c>GenericTransformer.Layer.Activate</c>.</summary>
    private void Activate(IBackend backend, Tensor outp, Tensor inp)
    {
        switch (_cfg.Activation)
        {
            case ActivationKind.GeluTanh: backend.Gelu(outp, inp); break;
            case ActivationKind.Relu: backend.Clamp(outp, inp, 0f, float.PositiveInfinity); break;
            case ActivationKind.ReluSquared:
                backend.Clamp(outp, inp, 0f, float.PositiveInfinity);
                backend.Mul(outp, outp, outp);
                break;
            default: backend.Silu(outp, inp); break;
        }
    }

    /// <summary>Column-parallel slice: rank <paramref name="rank"/>'s contiguous OUT-row range of a row-major <c>[out, in]</c> weight (or <c>[out]</c> bias-shaped vector); a contiguous byte copy for any dtype since GGUF quant blocks run along IN, so every OUT row is a whole number of blocks and a row range never splits a block.</summary>
    internal static Tensor SliceOutRows(Tensor w, int rank, int degree, string name)
    {
        long rows = w.Shape[0];
        if (rows % degree != 0)
        {
            throw new ArgumentException($"{name}: {rows} output rows do not tile across TP degree {degree}.");
        }
        long rowsPer = rows / degree;
        long elementsPerRow = w.ElementCount / rows;
        long rowBytes = w.DType.ComputeByteCount(elementsPerRow);
        TensorShape shape = w.Shape.Rank == 1
            ? new TensorShape(rowsPer)
            : new TensorShape(rowsPer, w.Shape[1]);
        Tensor slice = new(shape, w.DType);
        long bytes = rowsPer * rowBytes;
        Buffer.MemoryCopy((byte*)w.DataPointer + rank * bytes, (void*)slice.DataPointer, bytes, bytes);
        return slice;
    }

    /// <summary>Row-parallel slice: rank <paramref name="rank"/>'s contiguous IN-column range of a row-major <c>[out, in]</c> weight — a strided per-row byte copy. For block-quantized dtypes the per-rank column count must be a whole number of blocks; refused loudly otherwise so a misaligned shard can never silently corrupt a projection.</summary>
    internal static Tensor SliceInCols(Tensor w, int rank, int degree, string name)
    {
        long outRows = w.Shape[0];
        long inCols = w.Shape[1];
        if (inCols % degree != 0)
        {
            throw new ArgumentException($"{name}: {inCols} input columns do not tile across TP degree {degree}.");
        }
        long colsPer = inCols / degree;
        int blockElems = w.DType.BlockElementCount;
        if (colsPer % blockElems != 0)
        {
            throw new NotSupportedException(
                $"{name}: TP degree {degree} gives {colsPer} input columns per rank, which is not a whole number of "
                + $"{w.DType.Name} blocks ({blockElems} elements) — this checkpoint cannot be row-parallel-sharded at this degree.");
        }
        long srcRowBytes = w.DType.ComputeByteCount(inCols);
        long dstRowBytes = w.DType.ComputeByteCount(colsPer);
        long srcColOffset = (long)rank * dstRowBytes;
        Tensor slice = new(new TensorShape(outRows, colsPer), w.DType);
        byte* src = (byte*)w.DataPointer;
        byte* dst = (byte*)slice.DataPointer;
        for (long row = 0; row < outRows; row++)
        {
            Buffer.MemoryCopy(src + row * srcRowBytes + srcColOffset, dst + row * dstRowBytes, dstRowBytes, dstRowBytes);
        }
        return slice;
    }

    /// <summary>Rank's F32 slice of a per-output-row bias vector (Q/K/V bias rows split with their heads).</summary>
    private static Tensor SliceBiasRows(Tensor bias, int rank, int degree)
    {
        Tensor f = GenericTransformer.EnsureF32(bias);
        Tensor slice = SliceOutRows(f, rank, degree, "bias");
        if (!ReferenceEquals(f, bias))
        {
            f.Dispose();
        }
        return slice;
    }

    /// <summary>Always-owned F32 copy of a norm/bias vector, optionally baking Gemma's <c>(1 + weight)</c> offset — a copy even when the source is already F32, so every rank's tensor has exactly one owner.</summary>
    private static Tensor CopyNormF32(Tensor t, bool addOne)
    {
        Tensor f = GenericTransformer.EnsureF32(t);
        Tensor copy = new(f.Shape, DType.F32);
        float offset = addOne ? 1f : 0f;
        float* src = (float*)f.DataPointer;
        float* dst = (float*)copy.DataPointer;
        for (long i = 0; i < f.ElementCount; i++)
        {
            dst[i] = src[i] + offset;
        }
        if (!ReferenceEquals(f, t))
        {
            f.Dispose();
        }
        return copy;
    }

    /// <summary>Fresh host clone of <paramref name="src"/> (the DataPointer read demotes a device copy first, so the clone starts from the authoritative bytes).</summary>
    private static Tensor HostClone(Tensor src)
    {
        Tensor copy = new(src.Shape, src.DType);
        long bytes = src.DType.ComputeByteCount(src.ElementCount);
        Buffer.MemoryCopy((void*)src.DataPointer, (void*)copy.DataPointer, bytes, bytes);
        return copy;
    }

    private static Tensor[] NonNull(Tensor?[] tensors)
    {
        Tensor[] result = new Tensor[tensors.Length];
        for (int i = 0; i < tensors.Length; i++)
        {
            result[i] = tensors[i] ?? throw new InvalidOperationException("rank partial missing.");
        }
        return result;
    }

    private Tensor Own(Tensor t)
    {
        _owned.Add(t);
        return t;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0)
        {
            throw new ObjectDisposedException(nameof(TensorParallelTransformer));
        }
    }

    /// <summary>Frees the per-rank weight slices and norm copies (checkpoint tensors stay caller-owned).</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        foreach (Tensor t in _owned)
        {
            t.Dispose();
        }
        _owned.Clear();
        if (_ownsEmbed)
        {
            _embed?.Dispose();
        }
        _embed = null;
        _head = null;
        _finalNorm = null;
    }

    /// <summary>One rank's slice of one decoder layer: sliced projections plus per-rank norm copies.</summary>
    private sealed class RankLayer
    {
        public required Tensor InNorm { get; init; }
        public required Tensor PostNorm { get; init; }
        public required Tensor QW { get; init; }
        public required Tensor KW { get; init; }
        public required Tensor VW { get; init; }
        public required Tensor OW { get; init; }
        public required Tensor GateW { get; init; }
        public required Tensor UpW { get; init; }
        public required Tensor DownW { get; init; }
        public Tensor? QB { get; set; }
        public Tensor? KB { get; set; }
        public Tensor? VB { get; set; }
        public Tensor? OB { get; set; }
        public Tensor? QNorm { get; set; }
        public Tensor? KNorm { get; set; }
    }
}

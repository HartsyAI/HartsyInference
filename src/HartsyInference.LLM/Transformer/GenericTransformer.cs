using HartsyInference.Core.Backends;
using HartsyInference.Core.Rope;
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
    private Tensor? _posEmbed;   // GPT-2/StarCoder absolute position table [maxPos, hidden] (host gather); null otherwise
    private Tensor? _embedNorm, _embedNormBias;   // BLOOM word-embedding LayerNorm (applied before the first block)
    private Tensor? _finalNorm;
    private Tensor? _finalNormBias;   // zero bias for the LayerNorm path (Cohere)
    private Tensor? _lmHead;   // null when tied — reuse _embed.
    // Tied models: the original QUANTIZED embed, kept for the lm_head GEMV. The F32 _embed is host-only (gather);
    // this quantized copy is what goes to the GPU + the fused decode GEMV — 4-5× less HBM/token than the F32 head,
    // and ~0.5 GB less VRAM (we preload this instead of the F32 table).
    private Tensor? _lmHeadQuant;
    private int _disposed;
    // Graph-decode RoPE table (see EnsureRopeTableForGraphDecode): built once, lazily, at the first capacity a
    // caller needs; rebuilt only if a LATER call needs more positions than currently built. The old table is
    // disposed on rebuild (not leaked) — safe by construction, see that method's doc.
    private Tensor? _graphDecodeCos, _graphDecodeSin;
    // Gemma-3 dual-RoPE: local (sliding-window) layers rotate with a different base frequency. Built alongside
    // the global tables when RopeLocalTheta > 0; ForwardGraphDecodeStep selects per layer via IsGlobalLayer.
    private Tensor? _graphDecodeCosLocal, _graphDecodeSinLocal;
    private int _graphDecodeCapacity;
    // Granite/MiniCPM-family embedding scale, pre-applied once to a dedicated GPU copy of the embed table for the
    // graph-decode gather kernel (see EnsureEmbedResidentForGraphDecode) — EmbedGatherDecodeStep has no scale
    // parameter, unlike EmbedLookup's host gather which multiplies per call.
    private Tensor? _graphDecodeEmbedScaled;
    // Gemma-4 per-layer embeddings (PLE): a top-level small embedding table + projection, shared across every
    // layer (each layer additionally has its own gate/proj/post-norm — see Layer). Null when PerLayerEmbeddingDim=0.
    private Tensor? _perLayerTokEmbd, _perLayerModelProj, _perLayerProjNorm;
    private Tensor? _perLayerTokEmbdRaw;   // quantized original — device-resident PLE gather for graph decode

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

    /// <summary>F32 view-or-copy: dequantizes quantized tensors, casts 16-bit floats, returns the SAME
    /// reference when already F32 (callers check <c>ReferenceEquals</c> before disposing).</summary>
    internal static Tensor EnsureF32(Tensor t)
    {
        if (t.DType == DType.F32) return t;
        if (t.DType.IsQuantized) return HartsyInference.ModelAssets.Gguf.GgufDequantizer.Dequantize(t, DType.F32);
        return t.CastTo(DType.F32);
    }

    /// <summary>Concatenates weight (<c>[N,K]</c>) or bias (<c>[N]</c>) tensors along dim 0 (output rows) via a
    /// plain byte-level copy — correct for any dtype, including block-quantized formats, because a GGUF/our
    /// quant block never spans two output rows (each row is an independent whole number of blocks; confirmed
    /// against every fused-GEMV kernel's block-layout assumption). Load-time only (called once per layer while
    /// building the layer's weights, not on the decode hot path) — used to fuse separate Q/K/V or gate/up
    /// projections into a single larger GEMV dispatch. All parts must share dtype and (for 2D) K.</summary>
    private static Tensor ConcatRows(params Tensor[] parts)
    {
        DType dt = parts[0].DType;
        long elementsPerRow = parts[0].ElementCount / parts[0].Shape[0];
        long totalRows = 0;
        foreach (Tensor p in parts) totalRows += p.Shape[0];
        TensorShape outShape = parts[0].Shape.Rank == 1
            ? new TensorShape(totalRows)
            : new TensorShape(totalRows, parts[0].Shape[1]);
        Tensor result = new(outShape, dt);
        long rowBytes = dt.ComputeByteCount(elementsPerRow);
        byte* dst = (byte*)result.DataPointer;
        long offset = 0;
        foreach (Tensor p in parts)
        {
            byte* src = (byte*)p.DataPointer;
            long bytes = p.Shape[0] * rowBytes;
            Buffer.MemoryCopy(src, dst + offset, bytes, bytes);
            offset += bytes;
        }
        return result;
    }

    /// <summary>Loads an RMSNorm weight to F32, optionally baking Gemma's <c>(1 + weight)</c> offset into a
    /// fresh tensor (so the runtime norm op is the standard one and no borrowed GGUF mmap / cached weight is
    /// mutated in place).</summary>
    private static Tensor LoadNorm(Tensor t, bool addOne)
    {
        Tensor f = EnsureF32(t);
        if (!addOne) return f;
        Tensor outp = new(f.Shape, DType.F32);
        long n = f.ElementCount;
        float* src = (float*)f.DataPointer;
        float* dst = (float*)outp.DataPointer;
        for (long i = 0; i < n; i++) dst[i] = src[i] + 1f;
        if (!ReferenceEquals(f, t)) f.Dispose();   // free the transient cast/dequant
        return outp;
    }

    /// <summary>A zeroed bias vector of length <paramref name="n"/> for the LayerNorm path (models that use
    /// LayerNorm but ship no norm bias, e.g. Cohere).</summary>
    private static Tensor ZeroBias(int n)
    {
        Tensor b = new(new TensorShape(n), DType.F32);
        float* p = (float*)b.DataPointer;
        for (int i = 0; i < n; i++) p[i] = 0f;
        return b;
    }

    /// <summary>Loads a LayerNorm bias (F32) under <paramref name="key"/>, or a zeroed length-<paramref name="n"/>
    /// vector when the checkpoint ships none (Cohere has no norm bias; StableLM / GPT-2-lineage do).</summary>
    private static Tensor LoadBiasOrZero(IReadOnlyDictionary<string, Tensor> w, string key, int n)
        => w.TryGetValue(key, out Tensor? b) ? EnsureF32(b) : ZeroBias(n);

    /// <summary>A zeroed tensor of the given shape (used to pad MLA's value heads up to the q/k head dim so the
    /// shared FlashAttention kernel, which assumes equal k/v dims, can be reused).</summary>
    private static Tensor ZeroTensor(TensorShape shape)
    {
        Tensor z = new(shape, DType.F32);
        long n = z.ElementCount;
        float* p = (float*)z.DataPointer;
        for (long i = 0; i < n; i++) p[i] = 0f;
        return z;
    }

    /// <summary>Returns an owned copy of <paramref name="hidden"/> (used to pass a layer through unchanged, e.g. an
    /// mllama cross-attention layer with no image present).</summary>
    private static Tensor CopyHidden(IBackend backend, Tensor hidden)
    {
        Tensor copy = new(hidden.Shape, DType.F32);
        backend.CopyTo(copy, hidden);
        return copy;
    }

    /// <summary>Normalizes <paramref name="input"/> with <paramref name="weight"/> using LayerNorm (mean-centered,
    /// Cohere) or RMSNorm (everything else).</summary>
    private static void Normalize(IBackend backend, Tensor output, Tensor input, Tensor weight, Tensor? bias, bool layerNorm, float eps)
    {
        if (layerNorm) backend.LayerNorm(output, input, weight, bias!, eps);
        else backend.RmsNorm(output, input, weight, eps);
    }

    /// <summary>Loads weights from an HF-style key dict. <paramref name="prefix"/> is everything up to (not
    /// including) <c>embed_tokens</c> (e.g. <c>"model"</c> for a standalone checkpoint).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, string? lmHeadKey = null)
    {
        ThrowIfDisposed();
        Tensor embedRaw = w[$"{prefix}.embed_tokens.weight"];
        _embed = EnsureF32(embedRaw);
        // Keep the raw (quantized OR 16-bit-float) tied embed for the lm_head so the decode projection runs the
        // fused GEMV over the native weight instead of a full-size F32 GEMM. For a large vocab this is the decode
        // bottleneck: Orpheus's 156,940×3072 head was 221 ms/token (~90% of decode) as an F32 GEMM vs ~a few ms
        // as a BF16 fused GEMV. (Qwen 151k×1024 quant head was ~28% of decode time — same fix, quant dtype.)
        if (_cfg.TieWordEmbeddings && (embedRaw.DType.IsQuantized || embedRaw.DType == DType.BF16 || embedRaw.DType == DType.F16))
            _lmHeadQuant = embedRaw;
        if (_cfg.AbsolutePositionEmbeddings) _posEmbed = EnsureF32(w[$"{prefix}.embed_positions.weight"]);
        if (_cfg.EmbeddingLayerNorm)
        {
            _embedNorm = EnsureF32(w[$"{prefix}.embed_norm.weight"]);
            _embedNormBias = EnsureF32(w[$"{prefix}.embed_norm.bias"]);
        }
        _finalNorm = LoadNorm(w[$"{prefix}.norm.weight"], _cfg.RmsNormAddOne);
        if (_cfg.UseLayerNorm) _finalNormBias = LoadBiasOrZero(w, $"{prefix}.norm.bias", _cfg.HiddenSize);
        if (_cfg.PerLayerEmbeddingDim > 0)
        {
            _perLayerTokEmbdRaw = w[$"{prefix}.per_layer_token_embd.weight"];
            _perLayerTokEmbd = EnsureF32(_perLayerTokEmbdRaw);
            _perLayerModelProj = w[$"{prefix}.per_layer_model_proj.weight"];
            _perLayerProjNorm = LoadNorm(w[$"{prefix}.per_layer_proj_norm.weight"], addOne: false);
        }
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, $"{prefix}.layers.{i}", i);
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
        _finalNorm = LoadNorm(w[$"{prefix}.norm.weight"], _cfg.RmsNormAddOne);
        if (_cfg.UseLayerNorm) _finalNormBias = LoadBiasOrZero(w, $"{prefix}.norm.bias", _cfg.HiddenSize);
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, $"{prefix}.layers.{i}", i);
    }

    /// <summary>All weight tensors (stable references) for <see cref="IBackend.PreloadWeights"/> /
    /// <see cref="IBackend.FreeWeights"/>.</summary>
    public IEnumerable<Tensor> EnumerateWeights(bool includeRedundantSplits = true)
    {
        // The embedding table is gathered host-side. Upload it to the GPU only when it doubles as the (tied)
        // lm_head; when the model has a separate lm_head, _embed is host-only — skip it (saves real VRAM, e.g.
        // ~0.8 GB for DeepSeek-V2-Lite's 102k×2048 F32 table).
        // Tied lm_head: upload the QUANTIZED head (fused GEMV, ~0.5 GB less VRAM) when available; else the F32
        // table. Untied: _embed is host-only (gather), skip it — the separate _lmHead covers the projection.
        if (_cfg.TieWordEmbeddings)
        {
            if (_lmHeadQuant is not null) yield return _lmHeadQuant;
            else if (_embed is not null) yield return _embed;
        }
        if (_embedNorm is not null) { yield return _embedNorm; yield return _embedNormBias!; }
        if (_finalNorm is not null) yield return _finalNorm;
        if (_finalNormBias is not null) yield return _finalNormBias;
        // _perLayerTokEmbd is gathered host-side (ComputePerLayerInputs), like _embed — never uploaded. It can be
        // huge (Gemma-4 E2B: [8960, 262144] — 6x wider than the main embed table) so skipping the GPU upload
        // matters, not just style: forcing it resident nearly OOM'd a 12GB card on a 2GB checkpoint.
        if (_perLayerModelProj is not null) { yield return _perLayerModelProj; yield return _perLayerProjNorm!; }
        if (_lmHead is not null) yield return _lmHead;
        foreach (Layer l in _layers)
            foreach (Tensor t in l.EnumerateWeights(includeRedundantSplits)) yield return t;
    }

    /// <summary>The weight subset one pipeline stage needs resident on ITS backend: the layer range's tensors,
    /// plus (first stage) the embedding norm and (last stage) the tied/untied head, final norm, and PLE
    /// projection. The union over a full contiguous stage tiling equals <see cref="EnumerateWeights"/> exactly —
    /// unit-asserted, since a dropped tensor here silently becomes a per-op PCIe re-upload.</summary>
    public IEnumerable<Tensor> EnumerateStageWeights(int startLayer, int endLayer, bool isFirstStage, bool isLastStage,
        bool includeRedundantSplits = true)
    {
        ThrowIfDisposed();
        if (startLayer < 0 || endLayer > _layers.Length || startLayer >= endLayer)
            throw new ArgumentException($"Invalid stage range [{startLayer}, {endLayer}) for a stack of {_layers.Length}.");
        if (isLastStage && _cfg.TieWordEmbeddings)
        {
            if (_lmHeadQuant is not null) yield return _lmHeadQuant;
            else if (_embed is not null) yield return _embed;
        }
        if (isFirstStage && _embedNorm is not null) { yield return _embedNorm; yield return _embedNormBias!; }
        if (isLastStage)
        {
            if (_finalNorm is not null) yield return _finalNorm;
            if (_finalNormBias is not null) yield return _finalNormBias;
            if (_perLayerModelProj is not null) { yield return _perLayerModelProj; yield return _perLayerProjNorm!; }
            if (_lmHead is not null) yield return _lmHead;
        }
        for (int i = startLayer; i < endLayer; i++)
            foreach (Tensor t in _layers[i].EnumerateWeights(includeRedundantSplits)) yield return t;
    }

    /// <summary>Runs the full stack across a layer-split placement: one <see cref="ForwardEmbeds"/> call per
    /// stage on that stage's backend, final norm and cache advance only on the last, and a HOST-staged hidden
    /// handoff between stages (the read fires the producing backend's lazy D2H; the next stage re-uploads —
    /// ~T×hidden F32 per boundary, 16-32 KB at decode). Works on any hardware; a direct
    /// <c>CopyFromPeer</c> boundary is a measured follow-up. Not supported for Gemma-4 PLE (per-layer inputs are
    /// computed against the full stack). <paramref name="crossStates"/> (mllama gated cross-attention) is
    /// produced once by the caller — presumed resident/computed on stage 0's backend — and peer-copied fresh
    /// onto every OTHER stage that owns a cross-attention layer via <see cref="IBackend.CopyFromPeer"/> (mirrors
    /// <c>QwenImageTransformer.ForwardSharded</c>'s <c>temb</c> handoff: copied fresh from the master at each
    /// boundary, not chained stage-to-stage, since it never changes within one call). Paid on every call
    /// (prefill + each decode step) that reaches a cross-attention-owning stage — a persistent per-stage cache is
    /// a measured follow-up, same status as the host-staged hidden handoff above.</summary>
    public Tensor ForwardEmbedsStaged(LlmPlacement placement, Tensor embeds, int t, int posStart, IKvCache cache,
        ReadOnlySpan<int> tokenIds = default, Tensor? crossStates = null, int crossLen = 0)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(placement);
        if (placement.LayerCount != _layers.Length)
            throw new ArgumentException($"Placement covers {placement.LayerCount} layers; model has {_layers.Length}.");
        if (placement.IsSingle)
            return ForwardEmbeds(placement.Stages[0].Backend, embeds, t, posStart, cache,
                crossStates: crossStates, crossLen: crossLen, tokenIds: tokenIds);
        if (_cfg.PerLayerEmbeddingDim > 0)
            throw new NotSupportedException("Layer-split placement does not support Gemma-4 per-layer embeddings yet.");

        IBackend crossStatesHome = placement.Stages[0].Backend;
        Tensor hidden = embeds;
        bool owns = false;
        for (int s = 0; s < placement.Stages.Count; s++)
        {
            LlmStage stage = placement.Stages[s];
            bool last = s == placement.Stages.Count - 1;

            Tensor? stageCross = null;
            bool ownsStageCross = false;
            if (crossStates is not null && StageHasCrossAttn(stage))
            {
                if (ReferenceEquals(stage.Backend, crossStatesHome))
                {
                    stageCross = crossStates;
                }
                else
                {
                    stageCross = new Tensor(crossStates.Shape, crossStates.DType);
                    stage.Backend.CopyFromPeer(stageCross, crossStates, crossStatesHome);
                    ownsStageCross = true;
                }
            }

            Tensor next = ForwardEmbeds(stage.Backend, hidden, t, posStart, cache,
                applyFinalNorm: last, startLayer: stage.StartLayer, endLayer: stage.EndLayer,
                crossStates: stageCross, crossLen: crossLen, tokenIds: tokenIds, advanceCache: last);
            if (ownsStageCross) stageCross!.Dispose();
            if (!last)
            {
                // LOAD-BEARING host materialization: this read IS the stage boundary — it syncs the hidden state
                // off the producing stage's device so the next stage's backend uploads from the host copy.
                _ = next.DataPointer;
            }
            if (owns) hidden.Dispose();
            hidden = next;
            owns = true;
        }
        return hidden;
    }

    /// <summary>True when any layer in <paramref name="stage"/>'s range is a gated cross-attention layer
    /// (mllama) — determines whether <see cref="ForwardEmbedsStaged"/> needs to peer-copy the vision features
    /// onto that stage's backend.</summary>
    private bool StageHasCrossAttn(LlmStage stage)
    {
        for (int i = stage.StartLayer; i < stage.EndLayer; i++)
        {
            if (_cfg.IsCrossAttnLayer(i)) return true;
        }
        return false;
    }

    /// <summary>Token-IDs-in convenience over <see cref="ForwardEmbedsStaged"/>, mirroring <see cref="Forward"/>.</summary>
    public Tensor ForwardStaged(LlmPlacement placement, ReadOnlySpan<int> tokenIds, int posStart, IKvCache cache)
    {
        ThrowIfDisposed();
        int t = tokenIds.Length;
        Tensor embeds = new(new TensorShape(1, t, _cfg.HiddenSize), DType.F32);
        EmbedLookup(embeds, tokenIds);
        Tensor output = ForwardEmbedsStaged(placement, embeds, t, posStart, cache, tokenIds: tokenIds);
        embeds.Dispose();
        return output;
    }

    /// <summary>Token-IDs-in path: host embedding gather (one tiny H2D), then the resident transformer.
    /// Returns the final <c>[1, T, hidden]</c> hidden state (post final RMSNorm).</summary>
    public Tensor Forward(IBackend backend, ReadOnlySpan<int> tokenIds, int posStart, IKvCache cache)
    {
        ThrowIfDisposed();
        int t = tokenIds.Length;
        Tensor embeds = new(new TensorShape(1, t, _cfg.HiddenSize), DType.F32);
        EmbedLookup(embeds, tokenIds);
        Tensor output = ForwardEmbeds(backend, embeds, t, posStart, cache, tokenIds: tokenIds);
        embeds.Dispose();
        return output;
    }

    /// <summary>Embedding-in path. Runs decoder layers <c>[startLayer, endLayer)</c> (default the full stack)
    /// and returns the <c>[1, T, hidden]</c> hidden state, final-normed when <paramref name="applyFinalNorm"/>
    /// is true (else the raw last-layer hidden). <paramref name="posStart"/> = <see cref="IKvCache.CurrentLength"/>.
    /// The cache advances once per call (after the layers run), matching the reference decoder — UNLESS
    /// <paramref name="advanceCache"/> is false: a staged (layer-split) driver calls this once per stage over ONE
    /// shared cache, and only the final stage may advance, or the write cursor moves stages× per token.
    /// <paramref name="tokenIds"/> is required (throws otherwise) when <see cref="TransformerConfig.PerLayerEmbeddingDim"/>
    /// is set (Gemma-4): its per-layer embedding mixing needs the actual token ids, not just their embedding —
    /// every other architecture ignores it.</summary>
    public Tensor ForwardEmbeds(IBackend backend, Tensor embeds, int t, int posStart, IKvCache cache,
        bool applyFinalNorm = true, int startLayer = 0, int? endLayer = null,
        Tensor? crossStates = null, int crossLen = 0, ReadOnlySpan<int> tokenIds = default,
        bool advanceCache = true)
    {
        ThrowIfDisposed();
        int last = endLayer ?? _layers.Length;
        if (startLayer < 0 || last > _layers.Length || startLayer > last)
            throw new ArgumentException($"Invalid layer range [{startLayer}, {last}) for a stack of {_layers.Length}.");

        // Absolute position embeddings (GPT-2/StarCoder): add posEmbed[posStart+s] to each token embedding,
        // host-side (the embeds buffer is host F32 on the token path). These models use no RoPE.
        // startLayer gate: on a staged (layer-split) run only the FIRST stage sees token embeddings — for
        // stages > 0 `embeds` is the previous stage's hidden state, and re-adding position rows there would
        // silently corrupt the output once per extra stage (EnumerateStageWeights already tiles _posEmbed
        // and _embedNorm to the first stage only; this makes the forward match that contract).
        if (_posEmbed is not null && startLayer == 0)
        {
            int h = _cfg.HiddenSize;
            float* ep = (float*)embeds.DataPointer;
            float* pp = (float*)_posEmbed.DataPointer;
            for (int s = 0; s < t; s++)
            {
                long src = (long)(posStart + s) * h;
                long dst = (long)s * h;
                for (int j = 0; j < h; j++) ep[dst + j] += pp[src + j];
            }
        }

        int d = _cfg.HeadDim;
        // MLA ropes only its decoupled rope-part, so the cos/sin table is sized to that (full rotary over it).
        int ropeTableDim = _cfg.Mla is not null ? _cfg.Mla.QkRopeHeadDim : d;

        // Global RoPE table (all layers when single-base; the full-attention layers under Gemma-3 dual-RoPE).
        Tensor cos = new(new TensorShape(1, t, ropeTableDim), DType.F32);
        Tensor sin = new(new TensorShape(1, t, ropeTableDim), DType.F32);
        BuildRope(cos, sin, t, posStart, ropeTableDim, _cfg.RotaryDim, _cfg.RopeTheta, _cfg.RopeScaling);
        // Gemma-3 dual-RoPE: local (sliding-window) layers use a smaller base frequency. Gemma-4 additionally
        // narrows the local head dimension itself (HeadDimSwa) — not just the RoPE base. Built once, reused.
        Tensor? cosLocal = null, sinLocal = null;
        if (_cfg.RopeLocalTheta > 0)
        {
            int dLocal = _cfg.HeadDimSwa > 0 ? _cfg.HeadDimSwa : d;
            int rotaryLocal = _cfg.HeadDimSwa > 0 ? _cfg.RotaryDimSwa : _cfg.RotaryDim;
            cosLocal = new(new TensorShape(1, t, dLocal), DType.F32);
            sinLocal = new(new TensorShape(1, t, dLocal), DType.F32);
            BuildRope(cosLocal, sinLocal, t, posStart, dLocal, rotaryLocal, _cfg.RopeLocalTheta, _cfg.RopeScaling);
        }

        // BLOOM applies a LayerNorm to the token embeddings before the first block (same startLayer gate as
        // _posEmbed above — first stage only).
        Tensor work = embeds;
        bool ownsWork = false;
        if (_embedNorm is not null && startLayer == 0)
        {
            work = new(embeds.Shape, DType.F32);
            backend.LayerNorm(work, embeds, _embedNorm, _embedNormBias!, _cfg.RmsNormEps);
            ownsWork = true;
        }

        // Gemma-4 per-layer embeddings: computed once per call (needs the real token ids, not just their
        // embedding), then each layer consumes (disposes) its own slice inside Layer.Forward.
        Tensor[]? perLayerInputs = null;
        if (_cfg.PerLayerEmbeddingDim > 0)
        {
            if (tokenIds.Length != t)
                throw new NotSupportedException("Gemma-4 per-layer embeddings require token ids; raw-embedding input (e.g. multimodal) is not yet supported.");
            perLayerInputs = ComputePerLayerInputs(backend, embeds, tokenIds, t);
        }

        try
        {
            Tensor hidden = work;
            bool ownsHidden = ownsWork;
            for (int i = startLayer; i < last; i++)
            {
                bool global = _cfg.IsGlobalLayer(i);
                Tensor lc = global ? cos : cosLocal ?? cos;
                Tensor ls = global ? sin : sinLocal ?? sin;
                Tensor next;
                if (_cfg.IsCrossAttnLayer(i))
                {
                    // mllama: cross-attend the vision features. With no image present the layer is skipped (HF masks
                    // it out), so the hidden state passes through unchanged.
                    next = crossStates is not null
                        ? _layers[i].CrossForward(backend, hidden, t, crossStates, crossLen)
                        : CopyHidden(backend, hidden);
                }
                else
                {
                    next = _cfg.Mla is not null
                        ? _layers[i].MlaForward(backend, hidden, t, posStart, cache, i, cos, sin)
                        : _layers[i].Forward(backend, hidden, t, posStart, cache, i, lc, ls, perLayerInputs?[i]);
                }
                if (ownsHidden) hidden.Dispose();
                hidden = next;
                ownsHidden = true;
            }
            if (advanceCache)
            {
                cache.AdvanceLength(t);
            }

            if (applyFinalNorm)
            {
                Tensor normed = new(new TensorShape(1, t, _cfg.HiddenSize), DType.F32);
                Normalize(backend, normed, hidden, _finalNorm!, _finalNormBias, _cfg.UseLayerNorm, _cfg.RmsNormEps);
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
            cosLocal?.Dispose();
            sinLocal?.Dispose();
        }
    }

    /// <summary>Gemma-4 per-layer embeddings (PLE): for each layer, mixes a small extra embedding derived two
    /// ways — a direct per-token gather from the top-level <see cref="_perLayerTokEmbd"/> table, and a projection
    /// of the (already embedding-scaled) main hidden state through <see cref="_perLayerModelProj"/> — averaged
    /// (via a fixed 1/√2 scale, not learned) after each is independently normalized/scaled. Returns one
    /// <c>[1, T, PerLayerEmbeddingDim]</c> tensor per layer; each is consumed (disposed) by its layer's
    /// <see cref="Layer.Forward"/>. Ported from llama.cpp's <c>build_inp_per_layer</c> / <c>project_per_layer_inputs</c>.</summary>
    private unsafe Tensor[] ComputePerLayerInputs(IBackend backend, Tensor embeds, ReadOnlySpan<int> tokenIds, int t)
    {
        int ple = _cfg.PerLayerEmbeddingDim;
        int layers = _layers.Length;
        int h = _cfg.HiddenSize;

        // Direct per-token gather from the per-layer token table (row width ple*layers per vocab id, laid out as
        // `layers` consecutive ple-sized chunks — same flat row-major layout as the main embed table), scaled by
        // sqrt(ple) (llama.cpp's tok_embd_scale).
        Tensor tokDirect = new(new TensorShape(1, t, ple * layers), DType.F32);
        float* td = (float*)tokDirect.DataPointer;
        float* te = (float*)_perLayerTokEmbd!.DataPointer;
        int width = ple * layers;
        float tokScale = MathF.Sqrt(ple);
        for (int s = 0; s < t; s++)
        {
            int id = tokenIds[s];
            float* row = te + (long)id * width;
            float* outRow = td + (long)s * width;
            for (int j = 0; j < width; j++) outRow[j] = row[j] * tokScale;
        }

        // Projected path: per_layer_model_proj(embeds) / sqrt(hidden) — one [1,T,ple] chunk per layer, packed
        // layer-major in the last dim (chunk `il` occupies [il*ple, (il+1)*ple)).
        Tensor projFlat = new(new TensorShape(1, t, ple * layers), DType.F32);
        Project(backend, projFlat, embeds, _perLayerModelProj!, null, _cfg.LowVramQuant);
        backend.Scale(projFlat, projFlat, 1f / MathF.Sqrt(h));

        Tensor[] result = new Tensor[layers];
        float invSqrt2 = 1f / MathF.Sqrt(2f);
        TensorShape chunkShape = new(1, t, ple);
        for (int il = 0; il < layers; il++)
        {
            Tensor projChunk = new(chunkShape, DType.F32);
            backend.SliceLastDim(projChunk, projFlat, il * ple);
            Tensor projNormed = new(chunkShape, DType.F32);
            backend.RmsNorm(projNormed, projChunk, _perLayerProjNorm!, _cfg.RmsNormEps);
            projChunk.Dispose();

            Tensor tokChunk = new(chunkShape, DType.F32);
            backend.SliceLastDim(tokChunk, tokDirect, il * ple);

            Tensor summed = new(chunkShape, DType.F32);
            backend.Add(summed, projNormed, tokChunk);
            projNormed.Dispose(); tokChunk.Dispose();
            backend.Scale(summed, summed, invSqrt2);
            result[il] = summed;
        }
        projFlat.Dispose();
        tokDirect.Dispose();
        return result;
    }

    /// <summary>True when this architecture is simple enough for CUDA-graph decode capture: the pre-norm
    /// GQA/RoPE decoder shape (Llama/Qwen/Mistral family — most dense models), INCLUDING (since 2026-07-22)
    /// the Gemma-2/3 trio of sliding-window attention, attention-logit soft-cap, and dual local/global RoPE.
    /// Excludes anything <see cref="Layer.ForwardGraphStep"/> doesn't implement: MLA, MoE, cross-attention,
    /// attention sink, ALiBi, parallel residual, absolute-position embeddings, post-norm placement (OLMo-2),
    /// BLOOM's embedding LayerNorm, Gemma-4's per-layer SWA head dim. Non-1.0 embedding scale
    /// (Gemma/Granite/MiniCPM) is NOT excluded — <see cref="EnsureEmbedResidentForGraphDecode"/> pre-applies it
    /// once to a dedicated GPU copy of the embed table, since the graph-decode gather kernel itself has no
    /// scale param.</summary>
    public bool SupportsGraphDecode(IBackend backend)
    {
        bool ok = SupportsGraphDecodeCore(backend);
        if (!ok && Environment.GetEnvironmentVariable("HARTSY_GRAPH_GATE_LOG") == "1")
            HartsyInference.Core.Logging.Logs.Info(
                $"[GraphGate] backend={backend.GraphDecodeSupported} mla={_cfg.Mla is null} moe={_cfg.Moe is null} " +
                $"cross={_cfg.CrossAttnLayers.Count == 0} sink={!_cfg.AttnSink} alibi={_cfg.AlibiMaxBias <= 0f} " +
                $"parRes={!_cfg.ParallelResidual} absPos={!_cfg.AbsolutePositionEmbeddings} " +
                $"noRopeGlobal={!_cfg.NoRopeOnGlobalLayers} preNorm={_cfg.NormPlacement == NormPlacement.PreNorm} " +
                $"embedLn={!_cfg.EmbeddingLayerNorm} kvShared={_cfg.KvSharedFromLayer == 0} " +
                $"ple={_cfg.PerLayerEmbeddingDim == 0 || _perLayerTokEmbdRaw?.DType == DType.Q5_K} " +
                $"(pleDim={_cfg.PerLayerEmbeddingDim} pleRaw={_perLayerTokEmbdRaw?.DType.ToString() ?? "null"})");
        return ok;
    }

    private bool SupportsGraphDecodeCore(IBackend backend) =>
        backend.GraphDecodeSupported
        && _cfg.Mla is null
        && _cfg.Moe is null
        && _cfg.CrossAttnLayers.Count == 0
        // Sliding-window, attention soft-cap, and dual local/global RoPE (the Gemma-2/3 trio) are supported
        // as of 2026-07-22: the split-K decode-attention kernel handles window/softcap with a device position,
        // FlashAttentionDev plumbs them through as per-layer capture constants, and ForwardGraphDecodeStep
        // selects the local-theta rope table per layer. Gemma-4's per-layer SWA head-dim (HeadDimSwa) is NOT
        // supported — the graph rope tables are built at a single head dim.
        && !_cfg.AttnSink
        && _cfg.AlibiMaxBias <= 0f
        && !_cfg.ParallelResidual
        && !_cfg.AbsolutePositionEmbeddings
        && !_cfg.NoRopeOnGlobalLayers
        && _cfg.NormPlacement == NormPlacement.PreNorm
        && !_cfg.EmbeddingLayerNorm
        // Gemma-4 (2026-07-23 bring-up): per-layer SWA head dim (dual-DIM rope tables), the weightless
        // V-norm, PLE (device-side Q5_K row gather by the device token id), and KV-SHARING (shared layers
        // run a q-only path against the donor layer's cache) are all graph-supported now.
        && (_cfg.PerLayerEmbeddingDim == 0 || _perLayerTokEmbdRaw?.DType == DType.Q5_K);

    /// <summary>Ensures the embedding table is GPU-resident (permanent weight-cache, like any other weight) and
    /// returns it — <see cref="ForwardGraphDecodeStep"/>'s on-device embed gather needs this; untied models
    /// normally skip uploading <c>_embed</c> since <see cref="EmbedLookup"/>'s host gather doesn't need it there.
    /// When <see cref="TransformerConfig.EmbeddingScale"/> isn't 1.0 (Granite/MiniCPM), returns a separate,
    /// lazily-built, pre-scaled GPU copy instead — the plain <c>_embed</c> table stays unscaled for
    /// <see cref="EmbedLookup"/>'s own per-call host scale. Call once before entering a graph-decode session.</summary>
    public Tensor EnsureEmbedResidentForGraphDecode(IBackend backend)
    {
        ThrowIfDisposed();
        backend.PreloadWeights([_embed!]);
        if (_cfg.EmbeddingScale == 1f) return _embed!;
        if (_graphDecodeEmbedScaled is null)
        {
            _graphDecodeEmbedScaled = new Tensor(_embed!.Shape, DType.F32);
            backend.Scale(_graphDecodeEmbedScaled, _embed!, _cfg.EmbeddingScale);
        }
        return _graphDecodeEmbedScaled;
    }

    /// <summary>Uploads the QUANTIZED per-layer token embedding table for graph decode's device-side PLE
    /// gather (Q5_K: 1.6 GB for E2B vs 9.4 GB F32 — the F32 host copy stays host-only for the eager path).</summary>
    public void EnsurePleResidentForGraphDecode(IBackend backend)
    {
        ThrowIfDisposed();
        if (_perLayerTokEmbdRaw is not null) backend.PreloadWeights([_perLayerTokEmbdRaw, _perLayerModelProj!, _perLayerProjNorm!]);
    }

    /// <summary>Returns the graph-decode RoPE table, building (or rebuilding, if <paramref name="minCapacity"/>
    /// exceeds what's already built) it via <see cref="IBackend.BuildRopeTableDevice"/>. Cached on this
    /// instance. A rebuild disposes the orphaned old tensors instead of leaking them — safe because a rebuild
    /// only replaces THIS field; any graph that already captured the old tensors' address keeps them alive by
    /// reference regardless of what this field points to next, and (for the scheduler's graph-decode retrofit
    /// specifically) a new capture can only happen while zero other graph-sessioned sequences are active — see
    /// <see cref="Generation.DynamicBatchScheduler"/>'s admission-time solo gate — so there is never a LIVE
    /// captured graph still depending on the table being replaced here.</summary>
    public (Tensor cos, Tensor sin) EnsureRopeTableForGraphDecode(IBackend backend, int minCapacity)
    {
        ThrowIfDisposed();
        if (_graphDecodeCos is null || _graphDecodeCapacity < minCapacity)
        {
            Tensor? oldCos = _graphDecodeCos, oldSin = _graphDecodeSin;
            Tensor? oldCosL = _graphDecodeCosLocal, oldSinL = _graphDecodeSinLocal;
            (_graphDecodeCos, _graphDecodeSin) = backend.BuildRopeTableDevice(minCapacity, _cfg.HeadDim, _cfg.RotaryDim, _cfg.RopeTheta, _cfg.RopeScaling);
            if (_cfg.RopeLocalTheta > 0)
            {
                // Same dims/rotary span/scaling as the eager dual-RoPE build (Forward's cosLocal/sinLocal):
                // Gemma-4's local layers use a NARROWER head dim (HeadDimSwa) — their table must be built at
                // that width, not the global HeadDim (rows are [pos, headDim]-indexed by the rope kernels).
                int dLocal = _cfg.HeadDimSwa > 0 ? _cfg.HeadDimSwa : _cfg.HeadDim;
                int rotaryLocal = _cfg.HeadDimSwa > 0 ? _cfg.RotaryDimSwa : _cfg.RotaryDim;
                (_graphDecodeCosLocal, _graphDecodeSinLocal) =
                    backend.BuildRopeTableDevice(minCapacity, dLocal, rotaryLocal, _cfg.RopeLocalTheta, _cfg.RopeScaling);
            }
            _graphDecodeCapacity = minCapacity;
            oldCos?.Dispose();
            oldSin?.Dispose();
            oldCosL?.Dispose();
            oldSinL?.Dispose();
        }
        return (_graphDecodeCos!, _graphDecodeSin!);
    }

    /// <summary>One full decode step (embed → every layer → final norm → logits → [repetition penalty] →
    /// greedy argmax) using the device-indexed graph-decode ops throughout, so the entire step is capturable
    /// into one CUDA graph and replayed with a single launch. <paramref name="deviceTokenId"/> is read at the
    /// start (this step's input token) and written at the end (the sampled next token) — the SAME buffer, so
    /// replaying the captured graph repeatedly chains greedy decode entirely on-device (no D2H between steps;
    /// the caller reads it back only when it needs the token for streaming/detokenization/stop-checking).
    /// When <paramref name="repetitionPenalty"/> != 1.0 (and <paramref name="history"/>/<paramref name="historyCount"/>
    /// are non-zero device buffers), the current input token is appended to the on-device history and repetition
    /// penalty is applied to the logits before argmax — replicating <c>RepetitionPenaltyStep</c>, the only sampler
    /// stage that can change greedy's picked token (temperature/top-k/top-p/min-p never remove the argmax
    /// survivor, so graph decode doesn't need them). Caller guarantees <see cref="SupportsGraphDecode"/> and that
    /// <paramref name="devicePos"/>/<paramref name="deviceTokenId"/> were refreshed (outside any capture region)
    /// for this step.</summary>
    public void ForwardGraphDecodeStep(IBackend backend, Tensor embedTable, IKvCache cache,
        Tensor cosTable, Tensor sinTable, ulong devicePos, ulong deviceTokenId,
        ulong history = 0, ulong historyCount = 0, float repetitionPenalty = 1.0f)
    {
        ThrowIfDisposed();
        bool applyRepPenalty = repetitionPenalty != 1.0f && history != 0 && historyCount != 0;
        if (applyRepPenalty) backend.AppendTokenHistoryStep(history, historyCount, deviceTokenId);

        Tensor embeds = new(new TensorShape(1, 1, _cfg.HiddenSize), DType.F32);
        backend.EmbedGatherDecodeStep(embeds, embedTable, deviceTokenId);

        // Gemma-4 PLE (graph flavor of ComputePerLayerInputs): device Q5_K row gather by the device token
        // id replaces the host gather; the projected path and per-layer mixing use the same ops in the same
        // order as eager, so the math matches the eager reference exactly.
        Tensor[]? perLayerInputs = null;
        if (_cfg.PerLayerEmbeddingDim > 0)
        {
            int ple = _cfg.PerLayerEmbeddingDim;
            int layers = _layers.Length;
            Tensor tokDirect = new(new TensorShape(1, 1, ple * layers), DType.F32);
            backend.PleGatherDecodeStep(tokDirect, _perLayerTokEmbdRaw!, MathF.Sqrt(ple), deviceTokenId);
            Tensor projFlat = new(new TensorShape(1, 1, ple * layers), DType.F32);
            Project(backend, projFlat, embeds, _perLayerModelProj!, null, _cfg.LowVramQuant);
            backend.Scale(projFlat, projFlat, 1f / MathF.Sqrt(_cfg.HiddenSize));
            perLayerInputs = new Tensor[layers];
            float invSqrt2 = 1f / MathF.Sqrt(2f);
            TensorShape chunkShape = new(1, 1, ple);
            for (int il = 0; il < layers; il++)
            {
                Tensor projChunk = new(chunkShape, DType.F32);
                backend.SliceLastDim(projChunk, projFlat, il * ple);
                Tensor projNormed = new(chunkShape, DType.F32);
                backend.RmsNorm(projNormed, projChunk, _perLayerProjNorm!, _cfg.RmsNormEps);
                projChunk.Dispose();
                Tensor tokChunk = new(chunkShape, DType.F32);
                backend.SliceLastDim(tokChunk, tokDirect, il * ple);
                Tensor summed = new(chunkShape, DType.F32);
                backend.Add(summed, projNormed, tokChunk);
                projNormed.Dispose(); tokChunk.Dispose();
                backend.Scale(summed, summed, invSqrt2);
                perLayerInputs[il] = summed;
            }
            projFlat.Dispose();
            tokDirect.Dispose();
        }

        Tensor hidden = embeds;
        for (int i = 0; i < _layers.Length; i++)
        {
            // Gemma-3 dual-RoPE: local (sliding-window) layers rotate with the local-theta table.
            bool localRope = _graphDecodeCosLocal is not null && !_cfg.IsGlobalLayer(i);
            Tensor next = _layers[i].ForwardGraphStep(backend, hidden, cache, i,
                localRope ? _graphDecodeCosLocal! : cosTable, localRope ? _graphDecodeSinLocal! : sinTable, devicePos,
                perLayerInputs?[i]);
            hidden.Dispose();
            hidden = next;
        }

        Tensor normed = new(new TensorShape(1, 1, _cfg.HiddenSize), DType.F32);
        Normalize(backend, normed, hidden, _finalNorm!, _finalNormBias, _cfg.UseLayerNorm, _cfg.RmsNormEps);
        hidden.Dispose();

        Tensor logits = ProjectLogits(backend, normed, 1);
        normed.Dispose();
        if (applyRepPenalty) backend.ApplyRepetitionPenaltyStep(logits, history, historyCount, repetitionPenalty);
        backend.ArgMaxInto(deviceTokenId, logits);
        logits.Dispose();
    }

    /// <summary>Graph-capturable single-token body step for embedding-driven callers (Sesame CSM's backbone and
    /// depth decoder), where the input is a host-built embedding — not a token id — and the output hidden feeds a
    /// separate head + a second transformer rather than an on-device argmax. Reads the step's input from the
    /// fixed-address <paramref name="inEmbed"/> <c>[1,1,H]</c> (the caller refreshes its contents OUTSIDE the
    /// capture region, like <see cref="ForwardGraphDecodeStep"/>'s device token id) and writes the post-final-norm
    /// hidden into the fixed-address <paramref name="outHidden"/> <c>[1,1,H]</c> (read back after the launch). Every
    /// layer uses the device-position graph-step ops, so the whole call captures into one graph and replays with a
    /// single launch. Neither <paramref name="inEmbed"/> nor <paramref name="outHidden"/> is disposed — both are the
    /// caller's persistent buffers whose baked addresses the graph reuses across replays. Caller guarantees
    /// <see cref="SupportsGraphDecode"/> and a refreshed <paramref name="devicePos"/> for this step.</summary>
    public void ForwardGraphDecodeStepEmbeds(IBackend backend, Tensor inEmbed, IKvCache cache,
        Tensor cosTable, Tensor sinTable, ulong devicePos, Tensor outHidden)
    {
        ThrowIfDisposed();
        Tensor hidden = inEmbed;
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].ForwardGraphStep(backend, hidden, cache, i, cosTable, sinTable, devicePos);
            if (!ReferenceEquals(hidden, inEmbed)) hidden.Dispose();   // never dispose the caller's fixed input buffer
            hidden = next;
        }
        Normalize(backend, outHidden, hidden, _finalNorm!, _finalNormBias, _cfg.UseLayerNorm, _cfg.RmsNormEps);
        if (!ReferenceEquals(hidden, inEmbed)) hidden.Dispose();
    }

    /// <summary>True when this architecture can run the two-stream shared-position graph decode step
    /// (<see cref="ForwardGraphDecodeStepDualEmbeds"/>): everything <see cref="SupportsGraphDecode"/> requires,
    /// minus the features <see cref="Layer.ForwardGraphStepDual"/> doesn't implement (per-layer SWA head dims,
    /// dual local/global RoPE tables, KV-sharing, per-layer embeddings, V-norm, LayerNorm/full-dim QK-norm).
    /// CSM/HeartMuLa's Llama backbone satisfies all of these.</summary>
    public bool SupportsDualGraphDecode(IBackend backend) =>
        SupportsGraphDecodeCore(backend)
        && _cfg.KvSharedFromLayer == 0
        && _cfg.PerLayerEmbeddingDim == 0
        && _cfg.HeadDimSwa == 0
        && _cfg.RopeLocalTheta <= 0
        && !_cfg.VNorm
        && !(_cfg.QkNorm && (_cfg.QkNormFullDim || _cfg.UseLayerNorm));

    /// <summary>Two-stream variant of <see cref="ForwardGraphDecodeStepEmbeds"/> for CSM/HeartMuLa's CFG decode:
    /// <paramref name="inEmbed"/> is <c>[1,2,H]</c> (row 0 = conditional, row 1 = unconditional), the rows are
    /// POSITION-ALIGNED (both streams append at the same absolute position, so ONE shared <paramref name="devicePos"/>
    /// serves RoPE, both KV scatters, and both attention calls), and each row writes into its OWN cache
    /// (<paramref name="cacheA"/>/<paramref name="cacheB"/>). Deliberately a separate method — NOT a batch parameter
    /// on <see cref="ForwardGraphDecodeStepEmbeds"/> — because the text-LLM fleet replays that path's captured
    /// graphs and it is verified byte-stable; it must not gain new dispatch branches. Same fixed-buffer contract:
    /// refresh <paramref name="inEmbed"/> and <paramref name="devicePos"/> outside any capture, read both rows'
    /// post-final-norm hiddens from <paramref name="outHidden"/> <c>[1,2,H]</c> after the launch. Caller guarantees
    /// <see cref="SupportsDualGraphDecode"/>.</summary>
    public void ForwardGraphDecodeStepDualEmbeds(IBackend backend, Tensor inEmbed, IKvCache cacheA, IKvCache cacheB,
        Tensor cosTable, Tensor sinTable, ulong devicePos, Tensor outHidden)
    {
        ThrowIfDisposed();
        Tensor hidden = inEmbed;
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].ForwardGraphStepDual(backend, hidden, cacheA, cacheB, i, cosTable, sinTable, devicePos);
            if (!ReferenceEquals(hidden, inEmbed)) hidden.Dispose();   // never dispose the caller's fixed input buffer
            hidden = next;
        }
        Normalize(backend, outHidden, hidden, _finalNorm!, _finalNormBias, _cfg.UseLayerNorm, _cfg.RmsNormEps);
        if (!ReferenceEquals(hidden, inEmbed)) hidden.Dispose();
    }

    /// <summary>Batched decode step for continuous batching: <paramref name="embeds"/> is <c>[1, B, hidden]</c>
    /// (one decode token per active sequence), <paramref name="positions"/>[b] is sequence b's absolute position
    /// (== its KV length), and <paramref name="caches"/>[b] is sequence b's own KV cache. Returns the post-norm
    /// hidden state <c>[1, B, hidden]</c>; ready for <see cref="ProjectLogits"/> (rows = B). The heavy
    /// projections/MLP run as one batched GEMM over all B tokens; attention is per-sequence (each token attends
    /// only its own prefix). Advances each cache by 1.</summary>
    public Tensor ForwardBatchDecode(IBackend backend, Tensor embeds, ReadOnlySpan<int> positions, IKvCache[] caches)
    {
        ThrowIfDisposed();
        int b = (int)embeds.Shape[1];
        if (positions.Length != b || caches.Length != b)
            throw new ArgumentException($"positions ({positions.Length}) / caches ({caches.Length}) must match batch B={b}.");
        int d = _cfg.HeadDim;

        Tensor cos = new(new TensorShape(1, b, d), DType.F32);
        Tensor sin = new(new TensorShape(1, b, d), DType.F32);
        BuildRopeBatched(cos, sin, positions, d, _cfg.RotaryDim, _cfg.RopeTheta, _cfg.RopeScaling);
        Tensor? cosLocal = null, sinLocal = null;
        if (_cfg.RopeLocalTheta > 0)
        {
            cosLocal = new(new TensorShape(1, b, d), DType.F32);
            sinLocal = new(new TensorShape(1, b, d), DType.F32);
            BuildRopeBatched(cosLocal, sinLocal, positions, d, _cfg.RotaryDim, _cfg.RopeLocalTheta, _cfg.RopeScaling);
        }
        try
        {
            Tensor hidden = embeds;
            bool ownsHidden = false;
            for (int i = 0; i < _layers.Length; i++)
            {
                bool global = _cfg.IsGlobalLayer(i);
                Tensor lc = global ? cos : cosLocal ?? cos;
                Tensor ls = global ? sin : sinLocal ?? sin;
                Tensor next = _layers[i].ForwardBatchDecode(backend, hidden, b, positions, lc, ls, caches, i);
                if (ownsHidden) hidden.Dispose();
                hidden = next;
                ownsHidden = true;
            }
            for (int s = 0; s < b; s++) caches[s].AdvanceLength(1);

            Tensor normed = new(new TensorShape(1, b, _cfg.HiddenSize), DType.F32);
            Normalize(backend, normed, hidden, _finalNorm!, _finalNormBias, _cfg.UseLayerNorm, _cfg.RmsNormEps);
            if (ownsHidden) hidden.Dispose();
            return normed;
        }
        finally
        {
            cos.Dispose();
            sin.Dispose();
            cosLocal?.Dispose();
            sinLocal?.Dispose();
        }
    }

    /// <summary>Per-row RoPE table for a ragged decode batch: row b uses absolute position <paramref name="positions"/>[b]
    /// (same split-half layout as <see cref="BuildRope"/>, so both RoPE styles consume it identically).</summary>
    private static void BuildRopeBatched(Tensor cos, Tensor sin, ReadOnlySpan<int> positions, int headDim, int rotaryDim, float theta, RopeScaling scaling)
    {
        int rdim = rotaryDim > 0 && rotaryDim < headDim ? rotaryDim : headDim;
        int half = rdim / 2;
        int maxPos = 0;
        for (int s = 0; s < positions.Length; s++) maxPos = Math.Max(maxPos, positions[s]);
        (double[] invFreq, double mscale) = RopeFrequencyBuilder.Build(rdim, theta, scaling, maxPos + 1);
        float* pc = (float*)cos.DataPointer;
        float* ps = (float*)sin.DataPointer;
        for (int s = 0; s < positions.Length; s++)
        {
            int pos = positions[s];
            long baseOff = (long)s * headDim;
            for (int i = 0; i < half; i++)
            {
                double angle = pos * invFreq[i];
                float c = (float)(Math.Cos(angle) * mscale);
                float si = (float)(Math.Sin(angle) * mscale);
                pc[baseOff + i] = c; pc[baseOff + i + half] = c;
                ps[baseOff + i] = si; ps[baseOff + i + half] = si;
            }
        }
    }

    /// <summary>Projects hidden <c>[1, T, hidden]</c> → logits <c>[1, T, vocab]</c> via the (tied) lm_head.</summary>
    public Tensor ProjectLogits(IBackend backend, Tensor hidden, int t)
    {
        ThrowIfDisposed();
        Tensor headW = _lmHead ?? _lmHeadQuant ?? _embed ?? throw new InvalidOperationException("weights not loaded.");
        Tensor logits = new(new TensorShape(1, t, _cfg.VocabSize), DType.F32);
        // Fused decode GEMV supports the K-quants (K % 256) and the 32-block quants (K % 32) directly from the
        // F32 hidden — no F16 cast, no whole-weight dequant, no F32 staging (so no large-vocab OOM). This is the
        // fast path for the (tied or not) lm_head. The divisibility gate MUST mirror CudaBackend.LinearImpl's
        // per-dtype dispatch: a blanket %256 here silently sent every model with hidden ≡ 128 (mod 256) — e.g.
        // qwen2.5-0.5b (896) and gemma3-1b (1152), whose heads are Q8_0 — down the dequant-to-F16-then-cuBLAS
        // route EVERY token; nsys measured that at 55% of qwen2.5-0.5b's total decode GPU time (2026-07-22).
        bool fusedHead = t <= 8
            && (((headW.DType == DType.Q4_K || headW.DType == DType.Q5_K || headW.DType == DType.Q6_K)
                    && _cfg.HiddenSize % 256 == 0)
                || ((headW.DType == DType.Q8_0 || headW.DType == DType.Q4_0 || headW.DType == DType.Q5_0)
                    && _cfg.HiddenSize % 32 == 0));
        if (fusedHead)
        {
            Project(backend, logits, hidden, headW, bias: null, _cfg.LowVramQuant);
        }
        else if (_cfg.LowVramQuant && headW.DType.IsQuantized)
        {
            // Large-vocab heads (GLM-4 = 151k, Qwen = 152k) are the biggest single weight. With an F32 hidden the
            // GEMM dtype resolves to BF16, whose dequant stages a full-size F32 temp (≈2.4 GB for a 151k×4096 head)
            // on top of the F16 dequant — OOM on a 12 GB card. Casting the tiny hidden to F16 makes the GEMM resolve
            // to F16 directly (one ≈1.2 GB dequant, no F32 staging). Logits never reach the SwiGLU-intermediate
            // magnitudes that motivate BF16 elsewhere, so F16 is numerically safe here.
            Tensor hidF16 = new(hidden.Shape, DType.F16);
            backend.CastToF16(hidF16, hidden);
            backend.QuantizedMatMul(logits, hidF16, headW, bias: null);
            hidF16.Dispose();
        }
        else
        {
            Project(backend, logits, hidden, headW, bias: null, _cfg.LowVramQuant);
        }
        // Granite divides the logits by logit_scale.
        if (_cfg.LogitScale != 1f) backend.Scale(logits, logits, 1f / _cfg.LogitScale);
        // Gemma-2 final-logit soft-cap: cap·tanh(logit/cap), elementwise in place.
        if (_cfg.FinalLogitSoftcap > 0f)
        {
            float cap = _cfg.FinalLogitSoftcap;
            backend.Scale(logits, logits, 1f / cap);
            backend.Tanh(logits, logits);
            backend.Scale(logits, logits, cap);
        }
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
        // Gemma scales token embeddings by sqrt(hidden) (the embedding normalizer); 1.0 for Qwen/Llama.
        if (_cfg.EmbeddingScale != 1f)
        {
            float scale = _cfg.EmbeddingScale;
            long total = (long)tokenIds.Length * h;
            for (long i = 0; i < total; i++) op[i] *= scale;
        }
    }

    /// <summary>Builds duplicated-half cos/sin: <c>cos[s,i] = cos[s,i+half] = cos((posStart+s)·freq_i)</c>,
    /// <c>freq_i = theta^(-2i/headDim)</c> — the split-half rotate-half convention of
    /// <see cref="IBackend.ApplyRopeSingle"/> (shared by Qwen2 / Qwen3 / Llama).</summary>
    internal static void BuildRope(Tensor cos, Tensor sin, int t, int posStart, int headDim, int rotaryDim, float theta, RopeScaling scaling)
    {
        // Partial rotary: build the table for the first rotaryDim dims (half = rotaryDim/2 duplicated), leaving the
        // rest of each headDim-strided row untouched (the kernel never reads it). 0/full → the whole head.
        int rdim = rotaryDim > 0 && rotaryDim < headDim ? rotaryDim : headDim;
        int half = rdim / 2;
        (double[] invFreq, double mscale) = RopeFrequencyBuilder.Build(rdim, theta, scaling, posStart + t);
        float* pc = (float*)cos.DataPointer;
        float* ps = (float*)sin.DataPointer;
        for (int s = 0; s < t; s++)
        {
            int pos = posStart + s;
            long baseOff = (long)s * headDim;
            for (int i = 0; i < half; i++)
            {
                double angle = pos * invFreq[i];
                float c = (float)(Math.Cos(angle) * mscale);
                float si = (float)(Math.Sin(angle) * mscale);
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
        _embed = null; _posEmbed = null; _embedNorm = null; _embedNormBias = null; _finalNorm = null; _lmHead = null; _lmHeadQuant = null;
        _perLayerTokEmbd = null; _perLayerModelProj = null; _perLayerProjNorm = null;
        _graphDecodeEmbedScaled?.Dispose(); _graphDecodeEmbedScaled = null;
    }

    /// <summary>One resident decoder layer: RMSNorm → GQA self-attn (optional Q/K norm, +KV cache) → residual
    /// → RMSNorm → SwiGLU → residual. All <see cref="IBackend"/> ops.</summary>
    private sealed class Layer
    {
        private readonly TransformerConfig _cfg;
        private Tensor? _inNorm, _postNorm;       // pre-attn norm; pre-MLP norm (Gemma: pre_feedforward)
        private Tensor? _postAttnNorm, _postFfnNorm;   // Gemma sandwich norms (post-attn / post-FFN, pre-residual)
        private Tensor? _normBias, _postNormBias;   // LayerNorm biases (input / pre-MLP); zero for Cohere, real for StableLM
        private Tensor? _parFfnNorm, _parFfnNormBias;   // GPT-NeoX parallel-residual: a SECOND norm feeding the FFN from the raw residual (Cohere has none → reuses the attn norm)
        private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB;
        // Fused Q+K+V projection weight/bias, built once at load time by concatenating _qW/_kW/_vW row-wise
        // (byte-level — safe for any dtype including block-quantized formats, since quant blocks never span
        // rows). Non-null only when this layer has its own K/V (not Gemma-4 KV-sharing). One GEMV dispatch
        // instead of three: at decode (N small), each of Q/K/V independently under-occupies the GPU (measured
        // via ncu: the K/V shapes specifically hit as few as 0.38 full waves across all SMs on a 28-SM GPU) —
        // fusing raises the effective N so the SAME kernel fills the GPU properly. QK-norm (Qwen3) is applied
        // AFTER slicing the fused output back into q/k/v, exactly as it was applied after the separate
        // projections before — fusion only changes the projection dispatch, not anything downstream.
        private Tensor? _qkvW, _qkvB;
        // Partial fused Q+K projection for layers where v's dtype differs (mixed-quant GGUF schemes put
        // Q6_K/Q8_0 on attn_v in roughly half the layers of a Q4_K_M/Q5_0-style file, which blocks the full
        // QKV fusion above). Q and K always share a dtype in those schemes, so [q|k] still fuses: one GEMV
        // (and one activation-quantize) launch replaces two, and V keeps its own projection.
        private Tensor? _qkW, _qkB;
        private Tensor? _qNorm, _kNorm, _qNormBias, _kNormBias;
        private Tensor? _sink;   // GPT-OSS: per-head attention-sink logits [Hq]
        private Tensor? _alibiSlopes;   // ALiBi: per-head slopes [Hq] (MPT/BLOOM/Falcon-classic); these models use no RoPE
        private Tensor? _kvAProj, _kvANorm, _kvBProj;   // MLA: KV down-proj (+latent norm) and up-proj
        private Tensor? _qAProj, _qANorm, _qBProj;      // MLA q-LoRA: Q down-proj (+norm) and up-proj (DeepSeek-V3/Kimi)
        private Tensor? _gateW, _upW, _downW;
        private Tensor? _gateB, _upB, _downB;   // FFN biases (GPT-2 / Falcon / BLOOM / MPT lineage)
        // Fused gate+up projection weight/bias (dense SwiGLU/GeGLU layers only — MoE routes separately and
        // never reaches DenseFfn's non-MoE path). Same rationale/technique as _qkvW above.
        private Tensor? _gateUpW, _gateUpB;
        private MoeFeedForward? _moe;   // non-null on MoE layers (replaces the dense SwiGLU above)
        private Multimodal.MllamaCrossAttentionLayer? _crossAttn;   // non-null on mllama gated cross-attention layers
        // Gemma-4 per-layer embeddings (PLE): this layer's own gate/proj/post-norm (the top-level token
        // table/projection/norm shared across all layers lives on GenericTransformer).
        private Tensor? _pleInpGate, _pleProj, _plePostNorm;
        // Gemma-4: optional learned per-layer output scale (a single scalar, read to host once — TENSOR_NOT_REQUIRED,
        // absent on most checkpoints), and the weightless V RMSNorm's constant (all-ones) "weight".
        private float? _outScale;
        private Tensor? _vNormOnes;
        // Gemma-4 parallel dense+MoE branch (26B-A4B only): the MoE branch's own pre-norm and post-norm, and the
        // dense/"shared" branch's own post-norm (its pre-norm reuses the plain _postNorm/ffn_norm slot). The
        // router's per-channel input scale is pre-multiplied by 1/sqrt(hidden) at load time (see LoadWeights) so
        // a single RmsNorm call against it reproduces llama.cpp's rms_norm→scale(1/√h)→mul(scale) chain exactly.
        private Tensor? _ffnPreNorm2, _ffnPostNorm1, _ffnPostNorm2, _moeGateInpScale;

        public Layer(TransformerConfig cfg) => _cfg = cfg;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, int layerIndex)
        {
            // Norms and biases are forced to F32; projection weights keep their loaded dtype (quant or float)
            // so the matmul backend can dequant-on-the-fly (or run a fused quantized GEMV) without an F32 copy.
            bool addOne = _cfg.RmsNormAddOne;
            // mllama gated cross-attention layer (Llama-3.2-Vision): a self-contained block that reads vision
            // features instead of the causal text K/V. Loads its own weights and skips the self-attn/FFN path.
            if (_cfg.IsCrossAttnLayer(layerIndex))
            {
                _crossAttn = new Multimodal.MllamaCrossAttentionLayer(
                    _cfg.HiddenSize, _cfg.NumHeads, _cfg.NumKvHeads, _cfg.HeadDim, _cfg.IntermediateSize, _cfg.RmsNormEps, _cfg.LowVramQuant);
                _crossAttn.LoadWeights(w, prefix);
                return;
            }
            if (_cfg.NormPlacement == NormPlacement.PostNorm)
            {
                // OLMo-2: no input / pre-FFN norm. The attention and FFN outputs are each RMSNorm'd before being
                // added to the residual (post_attention_norm / post_ffw_norm, mapped to the sandwich-norm slots).
                _postAttnNorm = LoadNorm(w[$"{prefix}.post_attention_layernorm.weight"], addOne);
                _postFfnNorm = LoadNorm(w[$"{prefix}.post_feedforward_layernorm.weight"], addOne);
            }
            else
            {
            _inNorm = LoadNorm(w[$"{prefix}.input_layernorm.weight"], addOne);
            if (_cfg.UseLayerNorm) _normBias = LoadBiasOrZero(w, $"{prefix}.input_layernorm.bias", _cfg.HiddenSize);
            if (_cfg.ParallelResidual)
            {
                // Cohere: one norm for both sublayers (attn + FFN read the same normed input). GPT-NeoX instead
                // carries a SEPARATE FFN norm (post_attention_layernorm) applied to the raw residual — load it when
                // present so the FFN reads its own norm rather than reusing the attention norm.
                if (w.ContainsKey($"{prefix}.post_attention_layernorm.weight"))
                {
                    _parFfnNorm = LoadNorm(w[$"{prefix}.post_attention_layernorm.weight"], addOne);
                    if (_cfg.UseLayerNorm) _parFfnNormBias = LoadBiasOrZero(w, $"{prefix}.post_attention_layernorm.bias", _cfg.HiddenSize);
                }
            }
            else if (_cfg.SandwichNorm)
            {
                // Gemma layout: input → attn → post_attention_layernorm → +res → pre_feedforward_layernorm →
                // mlp → post_feedforward_layernorm → +res. The pre-MLP norm reuses the _postNorm slot.
                _postAttnNorm = LoadNorm(w[$"{prefix}.post_attention_layernorm.weight"], addOne);
                _postNorm = LoadNorm(w[$"{prefix}.pre_feedforward_layernorm.weight"], addOne);
                _postFfnNorm = LoadNorm(w[$"{prefix}.post_feedforward_layernorm.weight"], addOne);
            }
            else
            {
                // Sequential pre-norm: a pre-MLP norm (StableLM/GPT-2-lineage carry a LayerNorm bias here too).
                _postNorm = LoadNorm(w[$"{prefix}.post_attention_layernorm.weight"], addOne);
                if (_cfg.UseLayerNorm) _postNormBias = LoadBiasOrZero(w, $"{prefix}.post_attention_layernorm.bias", _cfg.HiddenSize);
            }
            }
            if (_cfg.Mla is not null)
            {
                // MLA: KV down (a) + latent norm + KV up (b), o_proj. The query is either a direct projection
                // (V2-Lite, q_lora_rank=0) or compressed via q-LoRA: q_a_proj → q_a_norm → q_b_proj (V3 / Kimi-K2).
                if (_cfg.Mla.QLoraRank > 0)
                {
                    _qAProj = w[$"{prefix}.self_attn.q_a_proj.weight"];
                    _qANorm = LoadNorm(w[$"{prefix}.self_attn.q_a_norm.weight"], addOne);
                    _qBProj = w[$"{prefix}.self_attn.q_b_proj.weight"];
                }
                else
                {
                    _qW = w[$"{prefix}.self_attn.q_proj.weight"];
                }
                _kvAProj = w[$"{prefix}.self_attn.kv_a_proj.weight"];
                _kvANorm = LoadNorm(w[$"{prefix}.self_attn.kv_a_norm.weight"], addOne);
                _kvBProj = w[$"{prefix}.self_attn.kv_b_proj.weight"];
                _oW = w[$"{prefix}.self_attn.o_proj.weight"];
                LoadFfn(w, prefix, layerIndex);
                return;
            }
            _qW = w[$"{prefix}.self_attn.q_proj.weight"];
            // Gemma-4 KV-sharing layers compute no K/V of their own (see KvSharedFromLayer) — those tensors are
            // simply absent from the checkpoint for these layers; Forward reads an earlier donor layer's cache.
            bool hasOwnKv = _cfg.HasOwnKv(layerIndex);
            if (hasOwnKv)
            {
                _kW = w[$"{prefix}.self_attn.k_proj.weight"];
                _vW = w[$"{prefix}.self_attn.v_proj.weight"];
            }
            _oW = w[$"{prefix}.self_attn.o_proj.weight"];
            if (_cfg.AttentionBias)
            {
                _qB = EnsureF32(w[$"{prefix}.self_attn.q_proj.bias"]);
                if (hasOwnKv)
                {
                    _kB = EnsureF32(w[$"{prefix}.self_attn.k_proj.bias"]);
                    _vB = EnsureF32(w[$"{prefix}.self_attn.v_proj.bias"]);
                }
                // GPT-2 / Falcon / BLOOM also bias the output projection (Qwen2 does not — load only if present).
                if (w.TryGetValue($"{prefix}.self_attn.o_proj.bias", out Tensor? ob)) _oB = EnsureF32(ob);
            }
            // Fused Q+K+V dispatch (see _qkvW's doc comment) — only when this layer has its own K/V and all
            // three share a dtype (mixed-quant GGUF schemes occasionally substitute a different quant type per
            // tensor based on shape; K is shared across Q/K/V so this is expected to always hold in practice,
            // but the check keeps an edge case safely falling back to the existing separate-projection path
            // instead of building a nonsensical concatenation).
            if (hasOwnKv && _qW.DType == _kW!.DType && _qW.DType == _vW!.DType)
            {
                _qkvW = ConcatRows(_qW, _kW, _vW);
                if (_qB is not null && _kB is not null && _vB is not null)
                    _qkvB = ConcatRows(_qB, _kB, _vB);
            }
            else if (hasOwnKv && _qW.DType == _kW!.DType
                && Environment.GetEnvironmentVariable("HARTSY_QK_FUSION") != "0")
            {
                // Mixed-dtype v (see _qkW's doc comment): fuse what still matches.
                _qkW = ConcatRows(_qW, _kW);
                if (_qB is not null && _kB is not null)
                    _qkB = ConcatRows(_qB, _kB);
            }
            if (_cfg.QkNorm)
            {
                _qNorm = LoadNorm(w[$"{prefix}.self_attn.q_norm.weight"], _cfg.RmsNormAddOne);
                _kNorm = LoadNorm(w[$"{prefix}.self_attn.k_norm.weight"], _cfg.RmsNormAddOne);
                // StableLM's per-head q/k norm is LayerNorm (with an optional bias), not RMSNorm.
                if (_cfg.UseLayerNorm)
                {
                    _qNormBias = LoadBiasOrZero(w, $"{prefix}.self_attn.q_norm.bias", (int)_qNorm.ElementCount);
                    _kNormBias = LoadBiasOrZero(w, $"{prefix}.self_attn.k_norm.bias", (int)_kNorm.ElementCount);
                }
            }
            if (_cfg.AttnSink) _sink = EnsureF32(w[$"{prefix}.self_attn.sinks.weight"]);
            // ALiBi slopes are a config constant (geometric per-head schedule), not a checkpoint tensor — materialize
            // a small [Hq] device tensor so the attention kernel can add the per-head distance bias.
            if (_cfg.AlibiMaxBias > 0f && _cfg.AlibiSlopes.Count >= _cfg.NumHeads)
            {
                _alibiSlopes = new Tensor(new TensorShape(_cfg.NumHeads), DType.F32);
                float* ap = (float*)_alibiSlopes.DataPointer;
                for (int i = 0; i < _cfg.NumHeads; i++) ap[i] = _cfg.AlibiSlopes[i];
            }
            if (_cfg.VNorm)
            {
                // Sized to THIS layer's own head dim (local/SWA layers are narrower than global — HeadDimFor).
                int vNormWidth = _cfg.HeadDimFor(layerIndex);
                _vNormOnes = new Tensor(new TensorShape(vNormWidth), DType.F32);
                float* vp = (float*)_vNormOnes.DataPointer;
                for (int i = 0; i < vNormWidth; i++) vp[i] = 1f;
            }
            if (w.TryGetValue($"{prefix}.layer_out_scale.weight", out Tensor? outScaleT)) _outScale = ReadScalarF32(outScaleT);
            if (_cfg.PerLayerEmbeddingDim > 0)
            {
                _pleInpGate = w[$"{prefix}.per_layer_inp_gate.weight"];
                _pleProj = w[$"{prefix}.per_layer_proj.weight"];
                _plePostNorm = LoadNorm(w[$"{prefix}.per_layer_post_norm.weight"], addOne: false);
            }
            LoadFfn(w, prefix, layerIndex);
        }

        /// <summary>Reads a single-element tensor's value to the host (Gemma-4's optional per-layer output scale).</summary>
        private static unsafe float ReadScalarF32(Tensor t)
        {
            Tensor f32 = EnsureF32(t);
            float v = *(float*)f32.DataPointer;
            if (!ReferenceEquals(f32, t)) f32.Dispose();
            return v;
        }

        /// <summary>Loads the per-layer FFN: the MoE block on MoE layers, the dense gated/non-gated projections
        /// otherwise. Gemma-4's <see cref="TransformerConfig.ParallelDenseMoeBranch"/> MoE layers load BOTH — the
        /// dense projections back the "shared"/dense branch (reusing the plain <c>ffn_gate</c>/<c>ffn_up</c>/
        /// <c>ffn_down</c> tensor names, unlike every other MoE arch's dedicated <c>shared_expert.*</c> tensors)
        /// plus that branch's own post-norm, and the MoE branch's own pre-norm/post-norm/router-input-scale.</summary>
        private void LoadFfn(IReadOnlyDictionary<string, Tensor> w, string prefix, int layerIndex)
        {
            bool moeLayer = _cfg.IsMoeLayer(layerIndex);
            bool dualBranch = moeLayer && _cfg.ParallelDenseMoeBranch;
            if (moeLayer)
            {
                _moe = new MoeFeedForward(_cfg.Moe!, _cfg.HiddenSize, _cfg.LowVramQuant);
                _moe.LoadWeights(w, prefix);
            }
            if (dualBranch)
            {
                _ffnPreNorm2 = LoadNorm(w[$"{prefix}.mlp.ffn_pre_norm_2.weight"], addOne: false);
                _ffnPostNorm1 = LoadNorm(w[$"{prefix}.mlp.ffn_post_norm_1.weight"], addOne: false);
                _ffnPostNorm2 = LoadNorm(w[$"{prefix}.mlp.ffn_post_norm_2.weight"], addOne: false);
                // Pre-bake the router's 1/sqrt(hidden) scale into its per-channel weight so a single RmsNorm(attnOut,
                // this) call reproduces llama.cpp's rms_norm(attn_out) -> scale(1/sqrt(h)) -> mul(gate_inp_s) chain.
                Tensor rawScale = EnsureF32(w[$"{prefix}.mlp.gate_inp_scale.weight"]);
                _moeGateInpScale = new Tensor(rawScale.Shape, DType.F32);
                float invSqrtH = 1f / MathF.Sqrt(_cfg.HiddenSize);
                unsafe
                {
                    float* src = (float*)rawScale.DataPointer;
                    float* dst = (float*)_moeGateInpScale.DataPointer;
                    for (long i = 0; i < rawScale.ElementCount; i++) dst[i] = src[i] * invSqrtH;
                }
                if (!ReferenceEquals(rawScale, w[$"{prefix}.mlp.gate_inp_scale.weight"])) rawScale.Dispose();
            }
            if (!moeLayer || dualBranch)
            {
                // Non-gated FFN (GPT-2 / Falcon / Nemotron) has no gate_proj — only up (fc1) and down (fc2).
                if (_cfg.GatedFfn) _gateW = w[$"{prefix}.mlp.gate_proj.weight"];
                _upW = w[$"{prefix}.mlp.up_proj.weight"];
                _downW = w[$"{prefix}.mlp.down_proj.weight"];
                // FFN biases (GPT-2 / Falcon / BLOOM / MPT). Loaded only when present.
                if (_cfg.FfnBias)
                {
                    if (_cfg.GatedFfn && w.TryGetValue($"{prefix}.mlp.gate_proj.bias", out Tensor? gb)) _gateB = EnsureF32(gb);
                    if (w.TryGetValue($"{prefix}.mlp.up_proj.bias", out Tensor? ub)) _upB = EnsureF32(ub);
                    if (w.TryGetValue($"{prefix}.mlp.down_proj.bias", out Tensor? db)) _downB = EnsureF32(db);
                }
                // Fused gate+up dispatch — same rationale/technique as _qkvW (see its doc comment). Dense
                // SwiGLU/GeGLU only (non-gated FFNs have no gate_proj to fuse with); dtype-match guard for the
                // same mixed-quant-scheme edge case.
                if (_cfg.GatedFfn && _gateW is not null && _gateW.DType == _upW!.DType)
                {
                    _gateUpW = ConcatRows(_gateW, _upW);
                    if (_gateB is not null && _upB is not null)
                        _gateUpB = ConcatRows(_gateB, _upB);
                }
            }
        }

        /// <summary>Applies the configured FFN activation to <paramref name="inp"/> into <paramref name="outp"/>
        /// (same shape): SiLU (SwiGLU), tanh-GELU (GeGLU / GPT-2-lineage), ReLU, or ReLU² (Nemotron).</summary>
        private void Activate(IBackend backend, Tensor outp, Tensor inp)
        {
            switch (_cfg.Activation)
            {
                case ActivationKind.GeluTanh: backend.Gelu(outp, inp); break;
                case ActivationKind.Relu: backend.Clamp(outp, inp, 0f, float.PositiveInfinity); break;
                case ActivationKind.ReluSquared:
                    backend.Clamp(outp, inp, 0f, float.PositiveInfinity);
                    backend.Mul(outp, outp, outp);   // relu(x)² (elementwise, alias-safe)
                    break;
                default: backend.Silu(outp, inp); break;   // SiLU / SwiGLU
            }
        }

        /// <summary>FFN: routes to the dense SwiGLU/GeGLU, the non-gated MLP, or the MoE block. Consumes (disposes)
        /// <paramref name="preMlp"/> <c>[1, n, hidden]</c> and returns the FFN output <c>[1, n, hidden]</c>. On a
        /// Gemma-4 <see cref="TransformerConfig.ParallelDenseMoeBranch"/> layer this is still the MoE block ONLY
        /// (the parallel dense branch is handled separately by <see cref="GemmaMoeFfn"/>, which calls
        /// <see cref="DenseFfn"/> directly instead of going through here).</summary>
        private Tensor Mlp(IBackend backend, Tensor preMlp, int n, bool emitQ = false)
        {
            if (_moe is not null)
            {
                Tensor moeOut = _moe.Forward(backend, preMlp, n);
                preMlp.Dispose();
                return moeOut;
            }
            return DenseFfn(backend, preMlp, n, emitQ);
        }

        /// <summary>The dense gated (SwiGLU/GeGLU) or non-gated FFN only — never the MoE block. Consumes
        /// (disposes) <paramref name="preMlp"/>. Used by <see cref="Mlp"/> for every non-MoE layer, and directly
        /// by <see cref="GemmaMoeFfn"/> for Gemma-4's parallel dense/"shared" branch (which has a live
        /// <see cref="_moe"/> on the SAME layer, so it cannot go through <see cref="Mlp"/>'s dispatch).</summary>
        private Tensor DenseFfn(IBackend backend, Tensor preMlp, int n, bool emitQ = false)
        {
            // Derived from the loaded weight's own shape, not _cfg.IntermediateSize — Gemma-4's KV-sharing
            // layers use a WIDER FFN than the rest (E2B: 12288 vs 6144), so a single global config value would
            // under/over-allocate the intermediate buffer for those layers regardless of what governs the split.
            // Shape[0] is outDim (matches CudaBackend.LinearImpl's own N = weight.Shape[0] reading).
            TensorShape ff = new(1, n, (int)_upW!.Shape[0]);
            Tensor up = new(ff, DType.F32);
            Tensor comb;
            if (_cfg.GatedFfn)
            {
                // Gated SwiGLU/GeGLU: down(act(gate(x)) · up(x)). When the load-time fused gate|up weight
                // exists and the activation is SiLU or GELU-tanh, the whole epilogue (2× SliceLastDim +
                // activation + Mul, four kernels and three intermediates) collapses into ONE fused
                // GluActivate pass over the concatenated projection output.
                bool gluFusable = _gateUpW is not null
                    && _gateW!.Shape[0] == _upW!.Shape[0]
                    && _cfg.Activation is not (ActivationKind.Relu or ActivationKind.ReluSquared);
                if (gluFusable)
                {
                    int ffDim = (int)_gateW!.Shape[0];
                    Tensor gateUpOut = new(new TensorShape(1, n, ffDim * 2), DType.F32);
                    Project(backend, gateUpOut, preMlp, _gateUpW!, _gateUpB, _cfg.LowVramQuant);
                    comb = new(ff, DType.F32);
                    if (emitQ) backend.GluActivateEmitQ8(comb, gateUpOut, ffDim, gelu: _cfg.Activation == ActivationKind.GeluTanh);
                    else backend.GluActivate(comb, gateUpOut, ffDim, gelu: _cfg.Activation == ActivationKind.GeluTanh);
                    gateUpOut.Dispose();
                    up.Dispose();
                }
                else
                {
                    Tensor gate = new(ff, DType.F32);
                    ProjectGateUp(backend, gate, up, preMlp, n, _cfg.LowVramQuant);
                    Tensor gateAct = new(ff, DType.F32);
                    Activate(backend, gateAct, gate);
                    gate.Dispose();
                    comb = new(ff, DType.F32);
                    backend.Mul(comb, gateAct, up);
                    gateAct.Dispose(); up.Dispose();
                }
            }
            else
            {
                // Non-gated MLP (GPT-2 / Falcon / BLOOM / MPT / Nemotron): down(act(up(x))).
                Project(backend, up, preMlp, _upW!, _upB, _cfg.LowVramQuant);
                comb = new(ff, DType.F32);
                Activate(backend, comb, up);
                up.Dispose();
            }
            preMlp.Dispose();
            Tensor mlpOut = new(new TensorShape(1, n, _cfg.HiddenSize), DType.F32);
            Project(backend, mlpOut, comb, _downW!, _downB, _cfg.LowVramQuant);
            comb.Dispose();
            return mlpOut;
        }

        /// <summary>Projects gate+up, using the fused <see cref="_gateUpW"/> dispatch when available (one GEMV
        /// instead of two — see its doc comment), falling back to two separate calls otherwise. Splitting back
        /// apart is <see cref="IBackend.SliceLastDim"/> (gate offset 0, up offset nGate) — a real device copy,
        /// not a zero-cost view, but tiny relative to the GEMV bandwidth the fusion saves.</summary>
        private void ProjectGateUp(IBackend backend, Tensor gate, Tensor up, Tensor preMlp, int n, bool lowVram)
        {
            if (_gateUpW is not null)
            {
                int nGate = (int)_gateW!.Shape[0], nUp = (int)_upW!.Shape[0];
                Tensor combined = new(new TensorShape(1, n, nGate + nUp), DType.F32);
                Project(backend, combined, preMlp, _gateUpW, _gateUpB, lowVram);
                backend.SliceLastDim(gate, combined, 0);
                backend.SliceLastDim(up, combined, nGate);
                combined.Dispose();
                return;
            }
            Project(backend, gate, preMlp, _gateW!, _gateB, lowVram);
            Project(backend, up, preMlp, _upW!, _upB, lowVram);
        }

        /// <summary>Gemma-4's parallel dense+MoE FFN (only on <see cref="TransformerConfig.ParallelDenseMoeBranch"/>
        /// layers): the dense/"shared" branch and the routed-expert branch each read <paramref name="attnOut"/>
        /// through their OWN pre-norm, are each normed again after their own FFN, then summed. Neither branch's
        /// norm is <see cref="_postFfnNorm"/> — that shared norm is applied by the caller (<see cref="Forward"/>)
        /// to the sum, exactly like every other architecture's single FFN output. Does not consume
        /// <paramref name="attnOut"/> (the caller still needs it for the residual add).</summary>
        private Tensor GemmaMoeFfn(IBackend backend, Tensor attnOut, int n)
        {
            TensorShape flat = new(1, n, _cfg.HiddenSize);

            // Branch 1: dense ("shared") FFN — pre-norm reuses the plain ffn_norm slot (_postNorm), own post-norm.
            Tensor dNorm = new(flat, DType.F32);
            backend.RmsNorm(dNorm, attnOut, _postNorm!, _cfg.RmsNormEps);
            Tensor dOut = DenseFfn(backend, dNorm, n);   // consumes dNorm
            Tensor dNormed = new(flat, DType.F32);
            backend.RmsNorm(dNormed, dOut, _ffnPostNorm1!, _cfg.RmsNormEps);
            dOut.Dispose();

            // Branch 2: routed MoE — its own pre-norm feeds the experts; the router reads a SEPARATELY normalized
            // view of attn_out (RmsNorm against the pre-scaled gate_inp_scale "weight" — see LoadWeights).
            Tensor mIn = new(flat, DType.F32);
            backend.RmsNorm(mIn, attnOut, _ffnPreNorm2!, _cfg.RmsNormEps);
            Tensor routerIn = new(flat, DType.F32);
            backend.RmsNorm(routerIn, attnOut, _moeGateInpScale!, _cfg.RmsNormEps);
            Tensor routerLogits = _moe!.ComputeRouterLogits(backend, routerIn, n);
            routerIn.Dispose();
            Tensor mOut = _moe.Forward(backend, mIn, n, routerLogits);
            routerLogits.Dispose();
            mIn.Dispose();
            Tensor mNormed = new(flat, DType.F32);
            backend.RmsNorm(mNormed, mOut, _ffnPostNorm2!, _cfg.RmsNormEps);
            mOut.Dispose();

            Tensor summed = new(flat, DType.F32);
            backend.Add(summed, dNormed, mNormed);
            dNormed.Dispose(); mNormed.Dispose();
            return summed;
        }

        /// <summary>Gemma-4 per-layer embedding mixing (see <see cref="GenericTransformer.ComputePerLayerInputs"/>
        /// for how <paramref name="perLayerInput"/> is built). Consumes (disposes) both <paramref name="cur"/>
        /// (this layer's output before mixing) and <paramref name="perLayerInput"/>; returns the mixed result.</summary>
        private Tensor ApplyPerLayerEmbedding(IBackend backend, Tensor cur, Tensor perLayerInput, int n)
        {
            TensorShape flat = new(1, n, _cfg.HiddenSize);
            TensorShape ffShape = new(1, n, _cfg.PerLayerEmbeddingDim);
            Tensor gate = new(ffShape, DType.F32);
            Project(backend, gate, cur, _pleInpGate!, null, _cfg.LowVramQuant);
            Tensor gateAct = new(ffShape, DType.F32);
            backend.Gelu(gateAct, gate);
            gate.Dispose();
            Tensor mixed = new(ffShape, DType.F32);
            backend.Mul(mixed, gateAct, perLayerInput);
            gateAct.Dispose();
            perLayerInput.Dispose();
            Tensor projected = new(flat, DType.F32);
            Project(backend, projected, mixed, _pleProj!, null, _cfg.LowVramQuant);
            mixed.Dispose();
            Tensor projNormed = new(flat, DType.F32);
            backend.RmsNorm(projNormed, projected, _plePostNorm!, _cfg.RmsNormEps);
            projected.Dispose();
            Tensor result = new(flat, DType.F32);
            backend.Add(result, cur, projNormed);
            cur.Dispose(); projNormed.Dispose();
            return result;
        }

        public IEnumerable<Tensor> EnumerateWeights(bool includeRedundantSplits = true)
        {
            // REDUNDANT SPLIT ORIGINALS (whose load-time fused copies serve every single-sequence decode
            // and prefill path) are enumerated LAST, and the device-residency preloader excludes them
            // outright (includeRedundantSplits: false) — on GLM-4 the duplicated gate/up + q/k splits are
            // ~2.9 GB that crowded real weights out of residency (43.8 → 27.9 tok/s via per-token graph
            // memcpy nodes). Their one consumer (the batch scheduler's split-projection path on
            // mixed-dtype layers) lazily auto-promotes them on first use instead.
            bool qkvFused = _qkvW is not null;
            bool qkFused = _qkW is not null;
            bool gluFused = _gateUpW is not null;
            Tensor?[] primary = [_inNorm, _postNorm, _postAttnNorm, _postFfnNorm, _normBias, _postNormBias, _parFfnNorm, _parFfnNormBias,
                qkvFused || qkFused ? null : _qW, _qB, qkvFused || qkFused ? null : _kW, _kB, qkvFused ? null : _vW, _vB, _oW, _oB,
                _qkvW, _qkvB, _qkW, _qkB, _qNorm, _kNorm, _qNormBias, _kNormBias, _sink, _alibiSlopes, _vNormOnes,
                _kvAProj, _kvANorm, _kvBProj, _qAProj, _qANorm, _qBProj,
                gluFused ? null : _gateW, _gateB, gluFused ? null : _upW, _upB, _gateUpW, _gateUpB, _downW, _downB,
                _pleInpGate, _pleProj, _plePostNorm, _ffnPreNorm2, _ffnPostNorm1, _ffnPostNorm2, _moeGateInpScale];
            Tensor?[] redundant = [qkvFused || qkFused ? _qW : null, qkvFused || qkFused ? _kW : null,
                qkvFused ? _vW : null, gluFused ? _gateW : null, gluFused ? _upW : null];
            foreach (Tensor? t in primary) if (t is not null) yield return t;
            if (_moe is not null) foreach (Tensor t in _moe.EnumerateWeights()) yield return t;
            if (_crossAttn is not null) foreach (Tensor t in _crossAttn.EnumerateWeights()) yield return t;
            if (includeRedundantSplits)
                foreach (Tensor? t in redundant) if (t is not null) yield return t;
        }

        /// <summary>mllama gated cross-attention forward: reads the encoded <paramref name="vision"/> features
        /// (<c>[1, L, hidden]</c>) instead of the causal text K/V. Used at the cross-attention layer indices; the
        /// self-attention <see cref="Forward"/> path is bypassed for these layers.</summary>
        public Tensor CrossForward(IBackend backend, Tensor hidden, int t, Tensor vision, int visionLen)
            => _crossAttn!.Forward(backend, hidden, t, vision, visionLen);

        /// <summary>Pre-sublayer norm: for pre-norm models (Llama/Qwen/Gemma) normalizes <paramref name="inp"/>
        /// into <paramref name="outp"/>; for post-norm models (OLMo-2) the sublayer reads the raw residual, so this
        /// copies through unchanged (the norm is applied to the sublayer <i>output</i> instead, via PostSublayer).</summary>
        private void PreSublayer(IBackend backend, Tensor outp, Tensor inp, Tensor? norm, Tensor? bias)
        {
            if (_cfg.NormPlacement == NormPlacement.PostNorm) backend.CopyTo(outp, inp);
            else Normalize(backend, outp, inp, norm!, bias, _cfg.UseLayerNorm, _cfg.RmsNormEps);
        }

        /// <summary>Projects Q/K/V, using the fused <see cref="_qkvW"/> dispatch when available (one GEMV
        /// instead of three — see its doc comment) — QK-norm, if any, is applied by the caller AFTER this
        /// returns, exactly as it was applied after the separate projections before; fusion only changes how
        /// the projection itself dispatches. Falls back to three separate calls when there's no fused weight
        /// (KV-sharing layers, MLA, the mixed-dtype edge case) or no K/V this layer owns. <paramref name="q"/>/
        /// <paramref name="k"/>/<paramref name="v"/> must already be allocated at their target shape by the
        /// caller (unchanged from the pre-fusion calling convention).</summary>
        private void ProjectQkv(IBackend backend, Tensor q, Tensor? k, Tensor? v, Tensor pre, int t, bool hasOwnKv, bool lowVram)
        {
            if (_qkvW is not null && hasOwnKv && k is not null && v is not null)
            {
                int nq = (int)_qW!.Shape[0], nk = (int)_kW!.Shape[0], nv = (int)_vW!.Shape[0];
                Tensor qkvOut = new(new TensorShape(1, t, nq + nk + nv), DType.F32);
                Project(backend, qkvOut, pre, _qkvW, _qkvB, lowVram);
                backend.SliceLastDim(q, qkvOut, 0);
                backend.SliceLastDim(k, qkvOut, nq);
                backend.SliceLastDim(v, qkvOut, nq + nk);
                qkvOut.Dispose();
                return;
            }
            if (_qkW is not null && hasOwnKv && k is not null && v is not null && t == 1)
            {
                // Partial fusion (mixed-dtype v): one [q|k] GEMV, v projects separately. DECODE ONLY (t=1):
                // the fusion's entire win is per-launch overhead, which prefill's single big GEMM already
                // amortizes — and fusing prefill would change the cuBLAS GEMM shape (one [q|k] GEMM instead
                // of two), shifting prefill numerics at float-rounding level. The decode GEMV is bit-exact
                // row-for-row (row-independent kernel), so t=1 fusion preserves outputs exactly.
                int nq = (int)_qW!.Shape[0], nk = (int)_kW!.Shape[0];
                Tensor qkOut = new(new TensorShape(1, t, nq + nk), DType.F32);
                Project(backend, qkOut, pre, _qkW, _qkB, lowVram);
                backend.SliceLastDim(q, qkOut, 0);
                backend.SliceLastDim(k, qkOut, nq);
                qkOut.Dispose();
                Project(backend, v, pre, _vW!, _vB, lowVram);
                return;
            }
            Project(backend, q, pre, _qW!, _qB, lowVram);
            if (hasOwnKv && k is not null && v is not null)
            {
                Project(backend, k, pre, _kW!, _kB, lowVram);
                Project(backend, v, pre, _vW!, _vB, lowVram);
            }
        }

        public Tensor Forward(IBackend backend, Tensor hidden, int t, int posStart,
            IKvCache cache, int layerIndex, Tensor cos, Tensor sin, Tensor? perLayerInput = null)
        {
            int h = _cfg.HiddenSize;
            int hq = _cfg.NumHeads;
            int hkv = _cfg.NumKvHeads;
            int d = _cfg.HeadDimFor(layerIndex);   // Gemma-4: local/SWA layers use a narrower head dim than global
            int group = _cfg.KvGroup;
            TensorShape flat = new(1, t, h);

            Tensor pre = new(flat, DType.F32);
            PreSublayer(backend, pre, hidden, _inNorm, _normBias);

            // Q/K/V projections written straight into head layout [1, T, heads, D] (Linear derives N from the
            // weight, not the output rank, so the byte-identical reshape is free and stays resident). For OLMoE's
            // whole-vector Q/K norm the projection output is shaped [1, T, QDim] so the following RMSNorm reduces
            // over the full vector (RmsNorm reduces over the input's last dim).
            // Gemma-4 KV-sharing layers (see KvSharedFromLayer) have no k/v of their own — k/v stay null and the
            // layer reads an earlier donor layer's already-populated cache slot below instead.
            bool hasOwnKv = _cfg.HasOwnKv(layerIndex);
            bool fullNorm = _cfg.QkNorm && _cfg.QkNormFullDim;
            Tensor q = new(fullNorm ? new TensorShape(1, t, _cfg.QDim) : new TensorShape(1, t, hq, d), DType.F32);
            Tensor? k = null, v = null;
            if (hasOwnKv)
            {
                k = new(fullNorm ? new TensorShape(1, t, _cfg.KvDim) : new TensorShape(1, t, hkv, d), DType.F32);
                v = new(new TensorShape(1, t, hkv, d), DType.F32);
            }
            ProjectQkv(backend, q, k, v, pre, t, hasOwnKv, _cfg.LowVramQuant);
            if (!_cfg.ParallelResidual) pre.Dispose();   // parallel residual reuses `pre` for the FFN below

            // Gemma-4: weightless RMSNorm on V (v / rms(v), no learned scale — the "weight" is a constant ones
            // vector materialized at load time). Every other architecture skips this (_vNormOnes is null).
            if (_cfg.VNorm && v is not null)
            {
                Tensor vN = new(v.Shape, DType.F32);
                backend.RmsNorm(vN, v, _vNormOnes!, _cfg.RmsNormEps);
                v.Dispose();
                v = vN;
            }

            // Q/K RMSNorm before RoPE: per-head over head_dim (Qwen3, q is [1,T,Hq,D]) or whole-vector over QDim
            // (OLMoE, q is [1,T,QDim]) — same call, the input's last dim sets the reduction width. Output is
            // always head-shaped for RoPE. k may be null (Gemma-4 KV-sharing layer) — nothing to normalize then.
            if (_cfg.QkNorm)
            {
                Tensor qN = new(new TensorShape(1, t, hq, d), DType.F32);
                if (_cfg.UseLayerNorm) backend.LayerNorm(qN, q, _qNorm!, _qNormBias!, _cfg.RmsNormEps);   // StableLM: q/k LayerNorm (with optional bias), not RMSNorm
                else backend.RmsNorm(qN, q, _qNorm!, _cfg.RmsNormEps);
                q.Dispose();
                q = qN;
                if (k is not null)
                {
                    Tensor kN = new(new TensorShape(1, t, hkv, d), DType.F32);
                    if (_cfg.UseLayerNorm) backend.LayerNorm(kN, k, _kNorm!, _kNormBias!, _cfg.RmsNormEps);
                    else backend.RmsNorm(kN, k, _kNorm!, _cfg.RmsNormEps);
                    k.Dispose();
                    k = kN;
                }
            }

            // ALiBi / absolute-position models use no RoPE (bias added in attention / positions added to embeds);
            // Cohere2's global (full-attention) layers also skip RoPE; all other layers/models rope.
            if (_cfg.AlibiMaxBias <= 0f && !_cfg.AbsolutePositionEmbeddings && !(_cfg.NoRopeOnGlobalLayers && _cfg.IsGlobalLayer(layerIndex)))
            {
                if (_cfg.Rope == RopeStyle.Interleaved)
                {
                    int rotaryI = _cfg.RotaryDimFor(layerIndex);
                    backend.ApplyRopeInterleaved(q, cos, sin, rotaryI);
                    if (k is not null) backend.ApplyRopeInterleaved(k, cos, sin, rotaryI);
                }
                else
                {
                    int rotary = _cfg.RotaryDimFor(layerIndex);
                    backend.ApplyRopeSingle(q, cos, sin, rotary);
                    if (k is not null) backend.ApplyRopeSingle(k, cos, sin, rotary);
                }
            }

            Tensor qMh = new(new TensorShape(1, hq, t, d), DType.F32);
            backend.Permute0213(qMh, q, t, hq, d);
            q.Dispose();

            Tensor kFull, vFull;
            if (hasOwnKv)
            {
                Tensor kMh = new(new TensorShape(1, hkv, t, d), DType.F32);
                Tensor vMh = new(new TensorShape(1, hkv, t, d), DType.F32);
                backend.Permute0213(kMh, k!, t, hkv, d);
                backend.Permute0213(vMh, v!, t, hkv, d);
                k!.Dispose(); v!.Dispose();
                cache.AppendStep(backend, layerIndex, kMh, vMh);
                kMh.Dispose(); vMh.Dispose();
                kFull = cache.KeyPrefix(layerIndex);   // [1, Hkv, kvLen, D] (cache-owned)
                vFull = cache.ValuePrefix(layerIndex);
            }
            else
            {
                // Gemma-4 KV-sharing: the donor layer ran earlier THIS SAME forward call (donor index < layerIndex
                // always), so its cache slot for this position is already populated — just read it.
                int donor = _cfg.KvDonorLayer(layerIndex);
                kFull = cache.KeyPrefix(donor);
                vFull = cache.ValuePrefix(donor);
            }

            // FlashAttention: GQA-aware (no K/V replication) online-softmax, causal via the absolute query
            // offset (decode t=1 naturally attends the whole prefix; prefill t>1 is per-row causal). No score
            // matrix, no causal-mask tensor.
            // kFull/vFull may be a fixed-capacity buffer whose seq stride exceeds the valid length, so pass the
            // valid key count (posStart + t) explicitly; the kernel reads the stride from the tensor shape.
            int kvLen = posStart + t;
            float scale = _cfg.AttnScale;
            // Local (sliding-window) layers attend only the most recent SlidingWindow keys; global layers attend
            // the full causal prefix (window 0). No-op for non-SWA models.
            int window = _cfg.SlidingWindow > 0 && !_cfg.IsGlobalLayer(layerIndex) ? _cfg.SlidingWindow : 0;
            Tensor attn = new(new TensorShape(1, hq, t, d), DType.F32);
            backend.FlashAttention(attn, qMh, kFull, vFull, kvLen, group, causal: true, qOffset: posStart, scale, _cfg.AttnLogitSoftcap, _sink, window, _alibiSlopes);
            qMh.Dispose();

            // hq*d (not _cfg.QDim, which is the GLOBAL head dim) — Gemma-4 local/SWA layers project to a
            // narrower width than global layers, and _oW's own shape (loaded from the checkpoint) already
            // matches whichever width this specific layer actually uses.
            Tensor attnFlat = new(new TensorShape(1, t, hq * d), DType.F32);
            backend.Permute0213(attnFlat, attn, hq, t, d);
            attn.Dispose();
            Tensor attnOut = new(flat, DType.F32);
            Project(backend, attnOut, attnFlat, _oW!, _oB, _cfg.LowVramQuant);
            attnFlat.Dispose();

            if (_cfg.ParallelResidual)
            {
                // Cohere / GPT-NeoX: attention and the FFN both read the same normed `pre`; their outputs are
                // summed into the residual once. No post-attn norm, no intermediate residual, no pre-FFN norm.
                attnOut = PostSublayer(backend, attnOut, null, flat);   // applies the residual multiplier if any
                Tensor? ffnNormed = null;
                if (_parFfnNorm is not null)   // GPT-NeoX: FFN reads its own norm of the raw residual, not `pre`
                {
                    ffnNormed = new(flat, DType.F32);
                    Normalize(backend, ffnNormed, hidden, _parFfnNorm, _parFfnNormBias, _cfg.UseLayerNorm, _cfg.RmsNormEps);
                }
                Tensor mlpPar = Mlp(backend, ffnNormed ?? pre, t);      // consumes `pre` (Cohere) or its own norm (NeoX)
                ffnNormed?.Dispose();
                mlpPar = PostSublayer(backend, mlpPar, null, flat);
                Tensor sum1 = new(flat, DType.F32);
                backend.Add(sum1, hidden, attnOut);
                Tensor parResult = new(flat, DType.F32);
                backend.Add(parResult, sum1, mlpPar);
                attnOut.Dispose(); mlpPar.Dispose(); sum1.Dispose();
                return parResult;
            }

            attnOut = PostSublayer(backend, attnOut, _postAttnNorm, flat);   // Gemma post-attn norm (pre-residual)
            Tensor afterAttn = new(flat, DType.F32);
            backend.Add(afterAttn, hidden, attnOut);
            attnOut.Dispose();

            Tensor mlpOut;
            if (_ffnPreNorm2 is not null)   // Gemma-4 parallel dense+MoE branch (26B-A4B MoE layers only)
            {
                mlpOut = GemmaMoeFfn(backend, afterAttn, t);
            }
            else
            {
                Tensor preMlp = new(flat, DType.F32);
                PreSublayer(backend, preMlp, afterAttn, _postNorm, _postNormBias);
                mlpOut = Mlp(backend, preMlp, t);
            }

            mlpOut = PostSublayer(backend, mlpOut, _postFfnNorm, flat);   // Gemma post-FFN norm (pre-residual)
            Tensor result = new(flat, DType.F32);
            backend.Add(result, afterAttn, mlpOut);
            afterAttn.Dispose(); mlpOut.Dispose();

            if (perLayerInput is not null) result = ApplyPerLayerEmbedding(backend, result, perLayerInput, t);
            if (_outScale is float os) backend.Scale(result, result, os);
            return result;
        }

        /// <summary>Single-token decode step for CUDA-graph capture/replay: identical math to <see cref="Forward"/>
        /// (t=1, non-parallel-residual, pre-norm path) but RoPE/KV-append/attention read their position from a
        /// device buffer instead of host ints, so the whole layer is graph-capturable. Caller (<see cref="GenericTransformer"/>'s
        /// graph-decode eligibility check) guarantees this layer has none of the config this doesn't handle
        /// (MLA, MoE, cross-attention, sliding window, softcap, sink, ALiBi, parallel residual, sandwich norm) —
        /// this method does not re-check, it assumes the plain GQA/RoPE decoder shape.</summary>
        public Tensor ForwardGraphStep(IBackend backend, Tensor hidden, IKvCache cache, int layerIndex,
            Tensor cosTable, Tensor sinTable, ulong devicePos, Tensor? perLayerInput = null)
        {
            const int t = 1;
            int h = _cfg.HiddenSize;
            int hq = _cfg.NumHeads;
            int hkv = _cfg.NumKvHeads;
            int d = _cfg.HeadDimFor(layerIndex);        // Gemma-4: local/SWA layers are narrower than global
            int rotaryDim = _cfg.RotaryDimFor(layerIndex);
            int group = _cfg.KvGroup;
            TensorShape flat = new(1, t, h);

            Tensor pre = new(flat, DType.F32);
            // Quantize-at-producer: the plain-RMS pre-attn norm feeds the QKV GEMV directly, so emit its
            // Q8_1 sidecar in the same launch (identical output bytes; the GEMV skips its quantize).
            if (_cfg.NormPlacement == NormPlacement.PreNorm && !_cfg.UseLayerNorm && _normBias is null)
                backend.RmsNormEmitQ8(pre, hidden, _inNorm!, _cfg.RmsNormEps);
            else
                PreSublayer(backend, pre, hidden, _inNorm, _normBias);

            // t=1 makes [1,1,heads,d] and [1,heads,1,d] byte-identical contiguous layouts, so q/k/v are
            // allocated DIRECTLY in the head-major shape attention and KV-append expect — the four per-layer
            // Permute0213 copies the eager path needs (and their intermediates' alloc/free graph nodes) do
            // not exist on this path. Every producer/consumer between projection and attention is layout-
            // agnostic at t=1: Linear/SliceLastDim size by element count, per-head Q/K-norm reduces over the
            // last dim (d either way), the decode RoPE kernel derives heads from elementCount/headDim, and
            // KvCacheAppendDev reads tNew from Shape[2] (1 in both layouts).
            bool fullNorm = _cfg.QkNorm && _cfg.QkNormFullDim;
            bool interleaved = _cfg.Rope == RopeStyle.Interleaved;
            // Gemma-4 KV-sharing: shared layers (no k/v of their own) attend an earlier donor layer's
            // cache — the donor ran earlier in this same captured step, so its cache tensors are the same
            // device buffers the capture already references. Donor pairing preserves the head dim.
            bool hasOwnKv = _cfg.HasOwnKv(layerIndex);
            int kvLayer = hasOwnKv ? layerIndex : _cfg.KvDonorLayer(layerIndex);
            Tensor kFull = cache.KeyPrefix(kvLayer);
            Tensor vFull = cache.ValuePrefix(kvLayer);
            Tensor q;
            if (!hasOwnKv)
            {
                // Q-only path (mirrors eager): project q, per-head q-norm, rope in place — no scatter.
                q = new(new TensorShape(1, hq, t, d), DType.F32);
                Project(backend, q, pre, _qW!, _qB, _cfg.LowVramQuant);
                pre.Dispose();
                if (_cfg.QkNorm)
                {
                    Tensor qN = new(new TensorShape(1, hq, t, d), DType.F32);
                    if (_cfg.UseLayerNorm) backend.LayerNorm(qN, q, _qNorm!, _qNormBias!, _cfg.RmsNormEps);
                    else backend.RmsNorm(qN, q, _qNorm!, _cfg.RmsNormEps);
                    q.Dispose();
                    q = qN;
                }
                backend.RopeApplyDecodeStep(q, cosTable, sinTable, rotaryDim, interleaved, devicePos);
            }
            else if (!_cfg.QkNorm && !_cfg.VNorm && _qkvW is not null)
            {
                // Fused QKV epilogue (no QK-norm between projection and rope): the concatenated [q|k|v]
                // projection output feeds ONE kernel that ropes q, ropes k into the cache, and copies v into
                // the cache — replacing three slices, two ropes, and two appends (bit-identical math).
                int nq = (int)_qW!.Shape[0], nk = (int)_kW!.Shape[0], nv = (int)_vW!.Shape[0];
                Tensor qkvOut = new(new TensorShape(1, t, nq + nk + nv), DType.F32);
                Project(backend, qkvOut, pre, _qkvW, _qkvB, _cfg.LowVramQuant);
                pre.Dispose();
                q = new(new TensorShape(1, hq, t, d), DType.F32);
                backend.QkvRopeScatterDecodeStep(q, kFull, vFull, qkvOut, cosTable, sinTable,
                    hq, hkv, d, rotaryDim, interleaved, devicePos);
                qkvOut.Dispose();
            }
            else if (_cfg.QkNorm && !_cfg.QkNormFullDim && !_cfg.UseLayerNorm && !_cfg.VNorm
                && (_qkvW is not null || _qkW is not null)
                && Environment.GetEnvironmentVariable("HARTSY_QKNORM_SCATTER") != "0")   // kill-switch
            {
                // Per-head QK-norm epilogue (Qwen3/Gemma-3): ONE kernel consumes the fused projection
                // output, per-head RMS-norms q/k, ropes, and scatters k/v into the cache — replacing
                // three slices, two RMSNorm launches, and the rope-scatter (bit-identical reduction,
                // see lm_qknorm_rope_scatter_f32). Full-dim norm (OLMoE), LayerNorm QK-norm (StableLM),
                // and V-norm (Gemma-4) keep the composed path below.
                int nq = (int)_qW!.Shape[0], nk = (int)_kW!.Shape[0];
                q = new(new TensorShape(1, hq, t, d), DType.F32);
                if (_qkvW is not null)
                {
                    int nv = (int)_vW!.Shape[0];
                    Tensor qkvOut = new(new TensorShape(1, t, nq + nk + nv), DType.F32);
                    Project(backend, qkvOut, pre, _qkvW, _qkvB, _cfg.LowVramQuant);
                    pre.Dispose();
                    backend.QkvNormRopeScatterDecodeStep(q, kFull, vFull, qkvOut, _qNorm!, _kNorm!,
                        _cfg.RmsNormEps, cosTable, sinTable, hq, hkv, d, rotaryDim, interleaved, devicePos);
                    qkvOut.Dispose();
                }
                else
                {
                    Tensor qkOut = new(new TensorShape(1, t, nq + nk), DType.F32);
                    Project(backend, qkOut, pre, _qkW!, _qkB, _cfg.LowVramQuant);
                    Tensor v = new(new TensorShape(1, hkv, t, d), DType.F32);
                    Project(backend, v, pre, _vW!, _vB, _cfg.LowVramQuant);
                    pre.Dispose();
                    backend.QkNormRopeScatterVDecodeStep(q, kFull, vFull, qkOut, v, _qNorm!, _kNorm!,
                        _cfg.RmsNormEps, cosTable, sinTable, hq, hkv, d, rotaryDim, interleaved, devicePos);
                    qkOut.Dispose();
                    v.Dispose();
                }
            }
            else if (!_cfg.QkNorm && !_cfg.VNorm && _qkW is not null
                && Environment.GetEnvironmentVariable("HARTSY_QK_SCATTER") != "0")   // kill-switch, mirrors HARTSY_QK_FUSION
            {
                // Partial fusion (mixed-dtype v, no QK-norm): one [q|k] GEMV plus v's own projection feed the
                // same rope+scatter kernel — q/k come from the concatenated buffer, v from its own tensor.
                int nq = (int)_qW!.Shape[0], nk = (int)_kW!.Shape[0];
                Tensor qkOut = new(new TensorShape(1, t, nq + nk), DType.F32);
                Project(backend, qkOut, pre, _qkW, _qkB, _cfg.LowVramQuant);
                Tensor v = new(new TensorShape(1, hkv, t, d), DType.F32);
                Project(backend, v, pre, _vW!, _vB, _cfg.LowVramQuant);
                pre.Dispose();
                q = new(new TensorShape(1, hq, t, d), DType.F32);
                backend.QkRopeScatterVDecodeStep(q, kFull, vFull, qkOut, v, cosTable, sinTable,
                    hq, hkv, d, rotaryDim, interleaved, devicePos);
                qkOut.Dispose();
                v.Dispose();
            }
            else
            {
                q = new(fullNorm ? new TensorShape(1, t, _cfg.QDim) : new TensorShape(1, hq, t, d), DType.F32);
                Tensor k = new(fullNorm ? new TensorShape(1, t, _cfg.KvDim) : new TensorShape(1, hkv, t, d), DType.F32);
                Tensor v = new(new TensorShape(1, hkv, t, d), DType.F32);
                ProjectQkv(backend, q, k, v, pre, t, hasOwnKv: true, _cfg.LowVramQuant);
                pre.Dispose();

                // Gemma-4 weightless V RMSNorm (v / rms(v), constant-ones weight) — mirrors eager Forward.
                if (_cfg.VNorm)
                {
                    Tensor vN = new(v.Shape, DType.F32);
                    backend.RmsNorm(vN, v, _vNormOnes!, _cfg.RmsNormEps);
                    v.Dispose();
                    v = vN;
                }

                if (_cfg.QkNorm)
                {
                    Tensor qN = new(new TensorShape(1, hq, t, d), DType.F32);
                    Tensor kN = new(new TensorShape(1, hkv, t, d), DType.F32);
                    if (_cfg.UseLayerNorm)
                    {
                        backend.LayerNorm(qN, q, _qNorm!, _qNormBias!, _cfg.RmsNormEps);
                        backend.LayerNorm(kN, k, _kNorm!, _kNormBias!, _cfg.RmsNormEps);
                    }
                    else
                    {
                        backend.RmsNorm(qN, q, _qNorm!, _cfg.RmsNormEps);
                        backend.RmsNorm(kN, k, _kNorm!, _cfg.RmsNormEps);
                    }
                    q.Dispose(); k.Dispose();
                    q = qN; k = kN;
                }

                // One fused rope+scatter launch (rope q → qRoped, rope k → cache, copy v → cache) replaces
                // two ropes and two appends — the separate-source variant of the fused-QKV path above.
                Tensor qRoped = new(new TensorShape(1, hq, t, d), DType.F32);
                backend.RopeScatterKvDecodeStep(qRoped, kFull, vFull, q, k, v, cosTable, sinTable,
                    hq, hkv, d, rotaryDim, interleaved, devicePos);
                q.Dispose(); k.Dispose(); v.Dispose();
                q = qRoped;
            }

            float scale = _cfg.AttnScale;
            // Local (sliding-window) layers attend only the most recent SlidingWindow keys — same per-layer
            // selection as the eager path; the window and soft-cap are constants baked into the captured graph.
            int window = _cfg.SlidingWindow > 0 && !_cfg.IsGlobalLayer(layerIndex) ? _cfg.SlidingWindow : 0;
            Tensor attn = new(new TensorShape(1, hq, t, d), DType.F32);
            backend.FlashAttentionDev(attn, q, kFull, vFull, 0, group, causal: true, 0, scale, devicePos,
                _cfg.AttnLogitSoftcap, window);
            q.Dispose();

            Tensor attnOut = new(flat, DType.F32);
            Project(backend, attnOut, attn, _oW!, _oB, _cfg.LowVramQuant);
            attn.Dispose();

            bool plainPreNorm = _cfg.NormPlacement == NormPlacement.PreNorm && !_cfg.UseLayerNorm
                && _postNormBias is null && _postNorm is not null && _cfg.ResidualMultiplier == 1f;
            bool sandwichFusion = Environment.GetEnvironmentVariable("HARTSY_SANDWICH_FUSION") != "0";   // kill-switch
            Tensor afterAttn = new(flat, DType.F32);
            Tensor preMlp = new(flat, DType.F32);
            // Sandwich layers (Gemma-2/3, GLM-4): post-attn norm + residual add + pre-FFN norm collapse
            // into ONE launch (bit-identical reductions/expressions — see lm_norm_add_rmsnorm_q8_1_f32);
            // plain pre-norm layers keep the fused add+RMSNorm; LayerNorm/biased/post-norm keep the
            // composed sequence.
            if (plainPreNorm && _postAttnNorm is not null && sandwichFusion)
            {
                backend.NormAddRmsNormEmitQ8(afterAttn, preMlp, hidden, attnOut, _postAttnNorm, _postNorm!, _cfg.RmsNormEps);
            }
            else if (plainPreNorm && _postAttnNorm is null)
            {
                backend.AddRmsNormEmitQ8(afterAttn, preMlp, hidden, attnOut, _postNorm!, _cfg.RmsNormEps);
            }
            else
            {
                attnOut = PostSublayer(backend, attnOut, _postAttnNorm, flat);
                backend.Add(afterAttn, hidden, attnOut);
                PreSublayer(backend, preMlp, afterAttn, _postNorm, _postNormBias);
            }
            attnOut.Dispose();
            Tensor mlpOut = Mlp(backend, preMlp, t, emitQ: true);

            Tensor result = new(flat, DType.F32);
            // Same collapse for the post-FFN norm + final residual add (2 launches -> 1).
            if (_postFfnNorm is not null && _cfg.ResidualMultiplier == 1f && sandwichFusion)
            {
                backend.RmsNormAdd(result, afterAttn, mlpOut, _postFfnNorm, _cfg.RmsNormEps);
            }
            else
            {
                mlpOut = PostSublayer(backend, mlpOut, _postFfnNorm, flat);
                backend.Add(result, afterAttn, mlpOut);
            }
            afterAttn.Dispose(); mlpOut.Dispose();
            // Gemma-4: mix in this layer's per-layer embedding and apply the learned output scale —
            // identical ops and order to the eager Forward's tail.
            if (perLayerInput is not null) result = ApplyPerLayerEmbedding(backend, result, perLayerInput, t);
            if (_outScale is float os) backend.Scale(result, result, os);
            return result;
        }

        /// <summary>Two-stream (B=2, shared-position) variant of <see cref="ForwardGraphStep"/> for CSM/HeartMuLa's
        /// CFG decode: the two rows of <paramref name="hidden"/> <c>[1,2,H]</c> are one decode token per stream at
        /// the SAME absolute position (one shared <paramref name="devicePos"/>), each stream appending into its own
        /// KV cache. The heavy batched parts (norms, QKV/o-proj/FFN projections) run ONCE over both rows — each
        /// weight matrix streams from HBM once per frame instead of twice, which is the entire win of the CFG-batched
        /// path — while RoPE+KV-scatter and attention run per row against that row's cache. Copy-adapted from
        /// <see cref="ForwardGraphStep"/> rather than parameterizing it: the text-LLM fleet replays that method's
        /// captured graphs and it must stay bit-identical. Caller guarantees
        /// <see cref="GenericTransformer.SupportsDualGraphDecode"/> (plain pre-norm GQA/RoPE, every layer owns its
        /// KV, single RoPE table, no PLE / V-norm / LayerNorm- or full-dim QK-norm).</summary>
        public Tensor ForwardGraphStepDual(IBackend backend, Tensor hidden, IKvCache cacheA, IKvCache cacheB,
            int layerIndex, Tensor cosTable, Tensor sinTable, ulong devicePos)
        {
            const int b = 2;
            int h = _cfg.HiddenSize;
            int hq = _cfg.NumHeads;
            int hkv = _cfg.NumKvHeads;
            int d = _cfg.HeadDim;
            int rotaryDim = _cfg.RotaryDim;
            int group = _cfg.KvGroup;
            bool interleaved = _cfg.Rope == RopeStyle.Interleaved;
            TensorShape flat = new(1, b, h);

            // Rows=2 → the Q8-sidecar emit self-disables inside RmsNormEmitQ8/AddRmsNormEmitQ8/GluActivateEmitQ8
            // (falls back to the plain fused kernels); the M=2 GEMVs quantize their own activations instead.
            Tensor pre = new(flat, DType.F32);
            PreSublayer(backend, pre, hidden, _inNorm, _normBias);

            // Row s's [heads, d] block of a token-major [1, b, heads, d] projection is contiguous — byte-identical
            // to the head-major [1, heads, 1, d] single-token layout the scatter/attention kernels expect — so a
            // row slice feeds them directly. The row offsets are capture CONSTANTS (row 0 = stream A, row 1 =
            // stream B on every frame); only the shared devicePos varies per replay.
            IKvCache[] caches = [cacheA, cacheB];
            Tensor[] qRows = new Tensor[b];
            if (!_cfg.QkNorm && _qkvW is not null)
            {
                // Fused-QKV path: one M=2 GEMV over the concatenated [q|k|v] weight, then per row the same
                // rope-q + rope-k-into-cache + copy-v-into-cache kernel the B=1 graph step uses.
                int nq = (int)_qW!.Shape[0], nk = (int)_kW!.Shape[0], nv = (int)_vW!.Shape[0];
                Tensor qkvOut = new(new TensorShape(1, b, nq + nk + nv), DType.F32);
                Project(backend, qkvOut, pre, _qkvW, _qkvB, _cfg.LowVramQuant);
                pre.Dispose();
                for (int s = 0; s < b; s++)
                {
                    Tensor qkvRow = new(new TensorShape(1, 1, nq + nk + nv), DType.F32);
                    backend.SliceRows(qkvRow, qkvOut, s);
                    Tensor qRow = new(new TensorShape(1, hq, 1, d), DType.F32);
                    backend.QkvRopeScatterDecodeStep(qRow, caches[s].KeyPrefix(layerIndex), caches[s].ValuePrefix(layerIndex),
                        qkvRow, cosTable, sinTable, hq, hkv, d, rotaryDim, interleaved, devicePos);
                    qkvRow.Dispose();
                    qRows[s] = qRow;
                }
                qkvOut.Dispose();
            }
            else
            {
                // Composed path (separate/mixed-dtype projections, optional per-head RMS QK-norm): batched
                // projections + norms, then the separate-source rope+scatter kernel per row.
                Tensor q = new(new TensorShape(1, b, hq, d), DType.F32);
                Tensor k = new(new TensorShape(1, b, hkv, d), DType.F32);
                Tensor v = new(new TensorShape(1, b, hkv, d), DType.F32);
                ProjectQkv(backend, q, k, v, pre, b, hasOwnKv: true, _cfg.LowVramQuant);
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
                for (int s = 0; s < b; s++)
                {
                    Tensor qIn = new(new TensorShape(1, hq, 1, d), DType.F32);
                    Tensor kRow = new(new TensorShape(1, hkv, 1, d), DType.F32);
                    Tensor vRow = new(new TensorShape(1, hkv, 1, d), DType.F32);
                    backend.SliceRows(qIn, q, s * hq);
                    backend.SliceRows(kRow, k, s * hkv);
                    backend.SliceRows(vRow, v, s * hkv);
                    Tensor qRow = new(new TensorShape(1, hq, 1, d), DType.F32);
                    backend.RopeScatterKvDecodeStep(qRow, caches[s].KeyPrefix(layerIndex), caches[s].ValuePrefix(layerIndex),
                        qIn, kRow, vRow, cosTable, sinTable, hq, hkv, d, rotaryDim, interleaved, devicePos);
                    qIn.Dispose(); kRow.Dispose(); vRow.Dispose();
                    qRows[s] = qRow;
                }
                q.Dispose(); k.Dispose(); v.Dispose();
            }

            // Attention per row (each token attends only its own stream's prefix; ~1.4 ms/frame — the GEMV
            // batching above is where the win lives). [1,1,hq,d] and [1,hq,1,d] are byte-identical at t=1, so
            // the segments concat along dim 1 straight back into the token-major [1,2,hq,d] o_proj input.
            float scale = _cfg.AttnScale;
            int window = _cfg.SlidingWindow > 0 && !_cfg.IsGlobalLayer(layerIndex) ? _cfg.SlidingWindow : 0;
            Tensor[] segs = new Tensor[b];
            for (int s = 0; s < b; s++)
            {
                Tensor attnSeg = new(new TensorShape(1, 1, hq, d), DType.F32);
                backend.FlashAttentionDev(attnSeg, qRows[s], caches[s].KeyPrefix(layerIndex), caches[s].ValuePrefix(layerIndex),
                    0, group, causal: true, 0, scale, devicePos, _cfg.AttnLogitSoftcap, window);
                qRows[s].Dispose();
                segs[s] = attnSeg;
            }
            Tensor attnConcat = new(new TensorShape(1, b, hq, d), DType.F32);
            backend.Concat(attnConcat, segs, dim: 1);
            foreach (Tensor seg in segs) seg.Dispose();

            Tensor attnOut = new(flat, DType.F32);
            Project(backend, attnOut, attnConcat, _oW!, _oB, _cfg.LowVramQuant);
            attnConcat.Dispose();

            // Residual/norm tail: identical op sequence to ForwardGraphStep, just over 2 rows.
            bool plainPreNorm = _cfg.NormPlacement == NormPlacement.PreNorm && !_cfg.UseLayerNorm
                && _postNormBias is null && _postNorm is not null && _cfg.ResidualMultiplier == 1f;
            bool sandwichFusion = Environment.GetEnvironmentVariable("HARTSY_SANDWICH_FUSION") != "0";   // kill-switch
            Tensor afterAttn = new(flat, DType.F32);
            Tensor preMlp = new(flat, DType.F32);
            if (plainPreNorm && _postAttnNorm is not null && sandwichFusion)
            {
                backend.NormAddRmsNormEmitQ8(afterAttn, preMlp, hidden, attnOut, _postAttnNorm, _postNorm!, _cfg.RmsNormEps);
            }
            else if (plainPreNorm && _postAttnNorm is null)
            {
                backend.AddRmsNormEmitQ8(afterAttn, preMlp, hidden, attnOut, _postNorm!, _cfg.RmsNormEps);
            }
            else
            {
                attnOut = PostSublayer(backend, attnOut, _postAttnNorm, flat);
                backend.Add(afterAttn, hidden, attnOut);
                PreSublayer(backend, preMlp, afterAttn, _postNorm, _postNormBias);
            }
            attnOut.Dispose();
            Tensor mlpOut = Mlp(backend, preMlp, b, emitQ: true);

            Tensor result = new(flat, DType.F32);
            if (_postFfnNorm is not null && _cfg.ResidualMultiplier == 1f && sandwichFusion)
            {
                backend.RmsNormAdd(result, afterAttn, mlpOut, _postFfnNorm, _cfg.RmsNormEps);
            }
            else
            {
                mlpOut = PostSublayer(backend, mlpOut, _postFfnNorm, flat);
                backend.Add(result, afterAttn, mlpOut);
            }
            afterAttn.Dispose(); mlpOut.Dispose();
            if (_outScale is float os) backend.Scale(result, result, os);
            return result;
        }

        /// <summary>Multi-head Latent Attention forward (DeepSeek-V2/V3). Q is projected directly (V2-Lite);
        /// K/V come from a compressed latent (down-proj → RMSNorm → up-proj) plus a shared RoPE key. Each head's
        /// Q/K is [no-position | rope] (rope applied only to the rope part); scores use the full qk head dim while
        /// values use v_head_dim. To reuse the equal-dim FlashAttention kernel, V is zero-padded up to the qk head
        /// dim and the output sliced back. (Naive decode: caches the per-head decompressed K/V.)</summary>
        public Tensor MlaForward(IBackend backend, Tensor hidden, int t, int posStart,
            IKvCache cache, int layerIndex, Tensor cos, Tensor sin)
        {
            MlaConfig mla = _cfg.Mla!;
            int h = _cfg.HiddenSize;
            int hq = _cfg.NumHeads;
            int qkHead = mla.QkHeadDim;        // nope + rope (cache/flash head dim)
            int qkNope = mla.QkNopeHeadDim;
            int qkRope = mla.QkRopeHeadDim;
            int vDim = mla.VHeadDim;
            int kvLora = mla.KvLoraRank;
            TensorShape flat = new(1, t, h);

            Tensor pre = new(flat, DType.F32);
            PreSublayer(backend, pre, hidden, _inNorm, _normBias);

            // Query: [1,t,hq,qkHead] laid out [q_nope | q_rope] per head. Direct projection (V2-Lite) or q-LoRA
            // (DeepSeek-V3 / Kimi-K2): q_a_proj → q_a_norm (RMSNorm) → q_b_proj.
            Tensor q = new(new TensorShape(1, t, hq, qkHead), DType.F32);
            if (mla.QLoraRank > 0)
            {
                Tensor qA = new(new TensorShape(1, t, mla.QLoraRank), DType.F32);
                Project(backend, qA, pre, _qAProj!, null, _cfg.LowVramQuant);
                Tensor qAN = new(new TensorShape(1, t, mla.QLoraRank), DType.F32);
                backend.RmsNorm(qAN, qA, _qANorm!, _cfg.RmsNormEps);
                qA.Dispose();
                Project(backend, q, qAN, _qBProj!, null, _cfg.LowVramQuant);
                qAN.Dispose();
            }
            else
            {
                Project(backend, q, pre, _qW!, null, _cfg.LowVramQuant);
            }

            // KV down-projection → compressed latent (kvLora) + shared rope key (qkRope).
            Tensor kvA = new(new TensorShape(1, t, kvLora + qkRope), DType.F32);
            Project(backend, kvA, pre, _kvAProj!, null, _cfg.LowVramQuant);
            pre.Dispose();
            Tensor kvLatent = new(new TensorShape(1, t, kvLora), DType.F32);
            backend.SliceLastDim(kvLatent, kvA, 0);
            Tensor kRope = new(new TensorShape(1, t, 1, qkRope), DType.F32);   // one shared rope "head"
            backend.SliceLastDim(kRope, kvA, kvLora);
            kvA.Dispose();
            Tensor kvLatentN = new(new TensorShape(1, t, kvLora), DType.F32);
            backend.RmsNorm(kvLatentN, kvLatent, _kvANorm!, _cfg.RmsNormEps);
            kvLatent.Dispose();

            // KV up-projection → per-head [k_nope | v].
            Tensor kvB = new(new TensorShape(1, t, hq, qkNope + vDim), DType.F32);
            Project(backend, kvB, kvLatentN, _kvBProj!, null, _cfg.LowVramQuant);
            kvLatentN.Dispose();

            // Split q and kv_b into their parts (per-head, via the head-dim-strided slice).
            Tensor qNope = new(new TensorShape(1, t, hq, qkNope), DType.F32); backend.SliceLastDim(qNope, q, 0);
            Tensor qRope = new(new TensorShape(1, t, hq, qkRope), DType.F32); backend.SliceLastDim(qRope, q, qkNope);
            q.Dispose();
            Tensor kNope = new(new TensorShape(1, t, hq, qkNope), DType.F32); backend.SliceLastDim(kNope, kvB, 0);
            Tensor v = new(new TensorShape(1, t, hq, vDim), DType.F32); backend.SliceLastDim(v, kvB, qkNope);
            kvB.Dispose();

            // Decoupled RoPE on the rope parts only (cos/sin are sized to qkRope).
            backend.ApplyRopeSingle(qRope, cos, sin);
            backend.ApplyRopeSingle(kRope, cos, sin);

            // Move to head-major [1, heads, t, d] for the broadcast / concat / cache / attention.
            Tensor qNopeMh = new(new TensorShape(1, hq, t, qkNope), DType.F32); backend.Permute0213(qNopeMh, qNope, t, hq, qkNope);
            Tensor qRopeMh = new(new TensorShape(1, hq, t, qkRope), DType.F32); backend.Permute0213(qRopeMh, qRope, t, hq, qkRope);
            Tensor kNopeMh = new(new TensorShape(1, hq, t, qkNope), DType.F32); backend.Permute0213(kNopeMh, kNope, t, hq, qkNope);
            Tensor vMh = new(new TensorShape(1, hq, t, vDim), DType.F32); backend.Permute0213(vMh, v, t, hq, vDim);
            Tensor kRopeMh = new(new TensorShape(1, 1, t, qkRope), DType.F32); backend.Permute0213(kRopeMh, kRope, t, 1, qkRope);
            qNope.Dispose(); qRope.Dispose(); kNope.Dispose(); v.Dispose(); kRope.Dispose();

            // Broadcast the shared rope key across all heads.
            Tensor kRopeB = new(new TensorShape(1, hq, t, qkRope), DType.F32);
            backend.RepeatKvHeads(kRopeB, kRopeMh, 1, hq);
            kRopeMh.Dispose();

            // Assemble full per-head q/k [nope|rope] and v padded to qkHead.
            Tensor qFull = new(new TensorShape(1, hq, t, qkHead), DType.F32); backend.Concat(qFull, [qNopeMh, qRopeMh], dim: 3);
            Tensor kFull = new(new TensorShape(1, hq, t, qkHead), DType.F32); backend.Concat(kFull, [kNopeMh, kRopeB], dim: 3);
            Tensor vPad = ZeroTensor(new TensorShape(1, hq, t, qkHead - vDim));
            Tensor vFull = new(new TensorShape(1, hq, t, qkHead), DType.F32); backend.Concat(vFull, [vMh, vPad], dim: 3);
            qNopeMh.Dispose(); qRopeMh.Dispose(); kNopeMh.Dispose(); kRopeB.Dispose(); vMh.Dispose(); vPad.Dispose();

            cache.AppendStep(backend, layerIndex, kFull, vFull);
            kFull.Dispose(); vFull.Dispose();
            int kvLen = posStart + t;
            // MLA softmax scale: DeepSeek YaRN folds mscale²/√qkHead into the score scale (cos/sin left neutral);
            // 0 → the plain 1/√qkHead (V2-Lite / no long-context scaling).
            float scale = mla.AttnScale > 0f ? mla.AttnScale : 1f / MathF.Sqrt(qkHead);
            Tensor attn = new(new TensorShape(1, hq, t, qkHead), DType.F32);
            backend.FlashAttention(attn, qFull, cache.KeyPrefix(layerIndex), cache.ValuePrefix(layerIndex),
                kvLen, 1, causal: true, qOffset: posStart, scale);
            qFull.Dispose();

            // Keep only the real v_head_dim of each head's output (the zero-padded tail contributes 0).
            Tensor attnV = new(new TensorShape(1, hq, t, vDim), DType.F32); backend.SliceLastDim(attnV, attn, 0);
            attn.Dispose();
            Tensor attnFlat = new(new TensorShape(1, t, hq * vDim), DType.F32); backend.Permute0213(attnFlat, attnV, hq, t, vDim);
            attnV.Dispose();
            Tensor attnOut = new(flat, DType.F32); Project(backend, attnOut, attnFlat, _oW!, null, _cfg.LowVramQuant);
            attnFlat.Dispose();

            Tensor afterAttn = new(flat, DType.F32);
            backend.Add(afterAttn, hidden, attnOut);
            attnOut.Dispose();
            Tensor preMlp = new(flat, DType.F32);
            PreSublayer(backend, preMlp, afterAttn, _postNorm, _postNormBias);
            Tensor mlpOut = Mlp(backend, preMlp, t);
            Tensor result = new(flat, DType.F32);
            backend.Add(result, afterAttn, mlpOut);
            afterAttn.Dispose(); mlpOut.Dispose();
            return result;
        }

        /// <summary>Post-processes a sublayer output before it is added to the residual stream: Gemma's sandwich
        /// RMSNorm (when <paramref name="norm"/> is set) followed by Granite's residual multiplier (when not 1).
        /// Both are no-ops for plain Qwen/Llama. Consumes/returns the (possibly replaced) tensor.</summary>
        private Tensor PostSublayer(IBackend backend, Tensor x, Tensor? norm, TensorShape shape)
        {
            if (norm is not null)
            {
                Tensor normed = new(shape, DType.F32);
                backend.RmsNorm(normed, x, norm, _cfg.RmsNormEps);
                x.Dispose();
                x = normed;
            }
            if (_cfg.ResidualMultiplier != 1f) backend.Scale(x, x, _cfg.ResidualMultiplier);
            return x;
        }

        /// <summary>Batched decode (one token per sequence): projections/MLP run as a single GEMM over all B
        /// tokens; attention is looped per sequence (each token attends only its own KV prefix via the scalar
        /// FlashAttention). <paramref name="hidden"/> is <c>[1, B, hidden]</c>.</summary>
        public Tensor ForwardBatchDecode(IBackend backend, Tensor hidden, int b, ReadOnlySpan<int> positions,
            Tensor cos, Tensor sin, IKvCache[] caches, int layerIndex)
        {
            int h = _cfg.HiddenSize;
            int hq = _cfg.NumHeads;
            int hkv = _cfg.NumKvHeads;
            int d = _cfg.HeadDim;
            int group = _cfg.KvGroup;
            TensorShape flat = new(1, b, h);

            Tensor pre = new(flat, DType.F32);
            PreSublayer(backend, pre, hidden, _inNorm, _normBias);

            bool fullNorm = _cfg.QkNorm && _cfg.QkNormFullDim;
            Tensor q = new(fullNorm ? new TensorShape(1, b, _cfg.QDim) : new TensorShape(1, b, hq, d), DType.F32);
            Tensor k = new(fullNorm ? new TensorShape(1, b, _cfg.KvDim) : new TensorShape(1, b, hkv, d), DType.F32);
            Tensor v = new(new TensorShape(1, b, hkv, d), DType.F32);
            ProjectQkv(backend, q, k, v, pre, b, hasOwnKv: true, _cfg.LowVramQuant);
            if (!_cfg.ParallelResidual) pre.Dispose();

            if (_cfg.QkNorm)
            {
                Tensor qN = new(new TensorShape(1, b, hq, d), DType.F32);
                Tensor kN = new(new TensorShape(1, b, hkv, d), DType.F32);
                if (_cfg.UseLayerNorm)
                {
                    backend.LayerNorm(qN, q, _qNorm!, _qNormBias!, _cfg.RmsNormEps);
                    backend.LayerNorm(kN, k, _kNorm!, _kNormBias!, _cfg.RmsNormEps);
                }
                else
                {
                    backend.RmsNorm(qN, q, _qNorm!, _cfg.RmsNormEps);
                    backend.RmsNorm(kN, k, _kNorm!, _cfg.RmsNormEps);
                }
                q.Dispose(); k.Dispose();
                q = qN; k = kN;
            }

            // RoPE per row (each row b is one token at its own absolute position); skipped on ALiBi / abs-pos / Cohere2 NoPE layers.
            if (_cfg.AlibiMaxBias <= 0f && !_cfg.AbsolutePositionEmbeddings && !(_cfg.NoRopeOnGlobalLayers && _cfg.IsGlobalLayer(layerIndex)))
            {
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
            }

            float scale = _cfg.AttnScale;
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

                IKvCache cache = caches[s];
                cache.AppendStep(backend, layerIndex, kMh, vMh);
                kMh.Dispose(); vMh.Dispose();
                int kvLen = cache.CurrentLength + 1;   // append did not advance; +1 for the just-written token
                int window = _cfg.SlidingWindow > 0 && !_cfg.IsGlobalLayer(layerIndex) ? _cfg.SlidingWindow : 0;
                Tensor attnSeg = new(new TensorShape(1, 1, hq, d), DType.F32);
                backend.FlashAttention(attnSeg, qMh, cache.KeyPrefix(layerIndex), cache.ValuePrefix(layerIndex),
                    kvLen, group, causal: true, qOffset: cache.CurrentLength, scale, _cfg.AttnLogitSoftcap, _sink, window, _alibiSlopes);
                qMh.Dispose();
                segs[s] = attnSeg;
            }

            Tensor attnConcat = new(new TensorShape(1, b, hq, d), DType.F32);
            backend.Concat(attnConcat, segs, dim: 1);
            foreach (Tensor seg in segs) seg.Dispose();

            Tensor attnOut = new(flat, DType.F32);   // o_proj reads attnConcat as [B, QDim]
            Project(backend, attnOut, attnConcat, _oW!, _oB, _cfg.LowVramQuant);
            attnConcat.Dispose();

            if (_cfg.ParallelResidual)
            {
                attnOut = PostSublayer(backend, attnOut, null, flat);
                Tensor? ffnNormed = null;
                if (_parFfnNorm is not null)   // GPT-NeoX: separate FFN norm of the raw residual
                {
                    ffnNormed = new(flat, DType.F32);
                    Normalize(backend, ffnNormed, hidden, _parFfnNorm, _parFfnNormBias, _cfg.UseLayerNorm, _cfg.RmsNormEps);
                }
                Tensor mlpPar = Mlp(backend, ffnNormed ?? pre, b);
                ffnNormed?.Dispose();
                mlpPar = PostSublayer(backend, mlpPar, null, flat);
                Tensor sum1 = new(flat, DType.F32);
                backend.Add(sum1, hidden, attnOut);
                Tensor parResult = new(flat, DType.F32);
                backend.Add(parResult, sum1, mlpPar);
                attnOut.Dispose(); mlpPar.Dispose(); sum1.Dispose();
                return parResult;
            }

            attnOut = PostSublayer(backend, attnOut, _postAttnNorm, flat);
            Tensor afterAttn = new(flat, DType.F32);
            backend.Add(afterAttn, hidden, attnOut);
            attnOut.Dispose();

            Tensor preMlp = new(flat, DType.F32);
            PreSublayer(backend, preMlp, afterAttn, _postNorm, _postNormBias);
            Tensor mlpOut = Mlp(backend, preMlp, b);

            mlpOut = PostSublayer(backend, mlpOut, _postFfnNorm, flat);
            Tensor result = new(flat, DType.F32);
            backend.Add(result, afterAttn, mlpOut);
            afterAttn.Dispose(); mlpOut.Dispose();
            return result;
        }
    }
}

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
    private int _graphDecodeCapacity;
    // Gemma-4 per-layer embeddings (PLE): a top-level small embedding table + projection, shared across every
    // layer (each layer additionally has its own gate/proj/post-norm — see Layer). Null when PerLayerEmbeddingDim=0.
    private Tensor? _perLayerTokEmbd, _perLayerModelProj, _perLayerProjNorm;

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
        // Keep the quantized embed for the tied lm_head so the decode projection runs the fused quant GEMV
        // instead of a 622 MB-per-token F32 GEMM (Qwen 151k×1024 head was ~28% of decode time).
        if (_cfg.TieWordEmbeddings && embedRaw.DType.IsQuantized) _lmHeadQuant = embedRaw;
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
            _perLayerTokEmbd = EnsureF32(w[$"{prefix}.per_layer_token_embd.weight"]);
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
    public IEnumerable<Tensor> EnumerateWeights()
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
        Tensor output = ForwardEmbeds(backend, embeds, t, posStart, cache, tokenIds: tokenIds);
        embeds.Dispose();
        return output;
    }

    /// <summary>Embedding-in path. Runs decoder layers <c>[startLayer, endLayer)</c> (default the full stack)
    /// and returns the <c>[1, T, hidden]</c> hidden state, final-normed when <paramref name="applyFinalNorm"/>
    /// is true (else the raw last-layer hidden). <paramref name="posStart"/> = <see cref="IKvCache.CurrentLength"/>.
    /// The cache advances once per call (after the layers run), matching the reference decoder.
    /// <paramref name="tokenIds"/> is required (throws otherwise) when <see cref="TransformerConfig.PerLayerEmbeddingDim"/>
    /// is set (Gemma-4): its per-layer embedding mixing needs the actual token ids, not just their embedding —
    /// every other architecture ignores it.</summary>
    public Tensor ForwardEmbeds(IBackend backend, Tensor embeds, int t, int posStart, IKvCache cache,
        bool applyFinalNorm = true, int startLayer = 0, int? endLayer = null,
        Tensor? crossStates = null, int crossLen = 0, ReadOnlySpan<int> tokenIds = default)
    {
        ThrowIfDisposed();
        int last = endLayer ?? _layers.Length;
        if (startLayer < 0 || last > _layers.Length || startLayer > last)
            throw new ArgumentException($"Invalid layer range [{startLayer}, {last}) for a stack of {_layers.Length}.");

        // Absolute position embeddings (GPT-2/StarCoder): add posEmbed[posStart+s] to each token embedding,
        // host-side (the embeds buffer is host F32 on the token path). These models use no RoPE.
        if (_posEmbed is not null)
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

        // BLOOM applies a LayerNorm to the token embeddings before the first block.
        Tensor work = embeds;
        bool ownsWork = false;
        if (_embedNorm is not null)
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
            cache.AdvanceLength(t);

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

    /// <summary>True when this architecture is simple enough for CUDA-graph decode capture: the plain pre-norm
    /// GQA/RoPE decoder shape (Llama/Qwen/Mistral family — most dense models). Excludes anything
    /// <see cref="Layer.ForwardGraphStep"/> doesn't implement: MLA, MoE, cross-attention, sliding-window,
    /// attention softcap/sink, ALiBi, parallel residual, absolute-position embeddings, post-norm placement
    /// (OLMo-2), dual local/global RoPE (Gemma-3), BLOOM's embedding LayerNorm, and non-1.0 embedding scale
    /// (Granite/MiniCPM — <see cref="EmbedLookup"/>'s scale isn't applied by the graph-decode embed kernel).</summary>
    public bool SupportsGraphDecode(IBackend backend) =>
        backend.GraphDecodeSupported
        && _cfg.Mla is null
        && _cfg.Moe is null
        && _cfg.CrossAttnLayers.Count == 0
        && _cfg.SlidingWindow == 0
        && _cfg.AttnLogitSoftcap == 0f
        && !_cfg.AttnSink
        && _cfg.AlibiMaxBias <= 0f
        && !_cfg.ParallelResidual
        && !_cfg.AbsolutePositionEmbeddings
        && !_cfg.NoRopeOnGlobalLayers
        && _cfg.NormPlacement == NormPlacement.PreNorm
        && _cfg.RopeLocalTheta == 0f
        && !_cfg.EmbeddingLayerNorm
        && _cfg.EmbeddingScale == 1f
        && _cfg.PerLayerEmbeddingDim == 0;

    /// <summary>Ensures the embedding table is GPU-resident (permanent weight-cache, like any other weight) and
    /// returns it — <see cref="ForwardGraphDecodeStep"/>'s on-device embed gather needs this; untied models
    /// normally skip uploading <c>_embed</c> since <see cref="EmbedLookup"/>'s host gather doesn't need it there.
    /// Call once before entering a graph-decode session.</summary>
    public Tensor EnsureEmbedResidentForGraphDecode(IBackend backend)
    {
        ThrowIfDisposed();
        backend.PreloadWeights([_embed!]);
        return _embed!;
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
            (_graphDecodeCos, _graphDecodeSin) = backend.BuildRopeTableDevice(minCapacity, _cfg.HeadDim, _cfg.RotaryDim, _cfg.RopeTheta, _cfg.RopeScaling);
            _graphDecodeCapacity = minCapacity;
            oldCos?.Dispose();
            oldSin?.Dispose();
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

        Tensor hidden = embeds;
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].ForwardGraphStep(backend, hidden, cache, i, cosTable, sinTable, devicePos);
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
        // Fused decode GEMV supports Q4_K/Q6_K/Q8_0 directly from the F32 hidden — no F16 cast, no whole-weight
        // dequant, no F32 staging (so no large-vocab OOM). This is the fast path for the (tied or not) lm_head.
        bool fusedHead = t <= 8 && (headW.DType == DType.Q4_K || headW.DType == DType.Q6_K || headW.DType == DType.Q8_0)
            && _cfg.HiddenSize % 256 == 0;
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
    private static void BuildRope(Tensor cos, Tensor sin, int t, int posStart, int headDim, int rotaryDim, float theta, RopeScaling scaling)
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
        private Tensor? _qNorm, _kNorm, _qNormBias, _kNormBias;
        private Tensor? _sink;   // GPT-OSS: per-head attention-sink logits [Hq]
        private Tensor? _alibiSlopes;   // ALiBi: per-head slopes [Hq] (MPT/BLOOM/Falcon-classic); these models use no RoPE
        private Tensor? _kvAProj, _kvANorm, _kvBProj;   // MLA: KV down-proj (+latent norm) and up-proj
        private Tensor? _qAProj, _qANorm, _qBProj;      // MLA q-LoRA: Q down-proj (+norm) and up-proj (DeepSeek-V3/Kimi)
        private Tensor? _gateW, _upW, _downW;
        private Tensor? _gateB, _upB, _downB;   // FFN biases (GPT-2 / Falcon / BLOOM / MPT lineage)
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
        private Tensor Mlp(IBackend backend, Tensor preMlp, int n)
        {
            if (_moe is not null)
            {
                Tensor moeOut = _moe.Forward(backend, preMlp, n);
                preMlp.Dispose();
                return moeOut;
            }
            return DenseFfn(backend, preMlp, n);
        }

        /// <summary>The dense gated (SwiGLU/GeGLU) or non-gated FFN only — never the MoE block. Consumes
        /// (disposes) <paramref name="preMlp"/>. Used by <see cref="Mlp"/> for every non-MoE layer, and directly
        /// by <see cref="GemmaMoeFfn"/> for Gemma-4's parallel dense/"shared" branch (which has a live
        /// <see cref="_moe"/> on the SAME layer, so it cannot go through <see cref="Mlp"/>'s dispatch).</summary>
        private Tensor DenseFfn(IBackend backend, Tensor preMlp, int n)
        {
            // Derived from the loaded weight's own shape, not _cfg.IntermediateSize — Gemma-4's KV-sharing
            // layers use a WIDER FFN than the rest (E2B: 12288 vs 6144), so a single global config value would
            // under/over-allocate the intermediate buffer for those layers regardless of what governs the split.
            // Shape[0] is outDim (matches CudaBackend.LinearImpl's own N = weight.Shape[0] reading).
            TensorShape ff = new(1, n, (int)_upW!.Shape[0]);
            Tensor up = new(ff, DType.F32);
            Project(backend, up, preMlp, _upW!, _upB, _cfg.LowVramQuant);
            Tensor comb;
            if (_cfg.GatedFfn)
            {
                // Gated SwiGLU/GeGLU: down(act(gate(x)) · up(x)).
                Tensor gate = new(ff, DType.F32);
                Project(backend, gate, preMlp, _gateW!, _gateB, _cfg.LowVramQuant);
                Tensor gateAct = new(ff, DType.F32);
                Activate(backend, gateAct, gate);
                gate.Dispose();
                comb = new(ff, DType.F32);
                backend.Mul(comb, gateAct, up);
                gateAct.Dispose(); up.Dispose();
            }
            else
            {
                // Non-gated MLP (GPT-2 / Falcon / BLOOM / MPT / Nemotron): down(act(up(x))).
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

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] all = [_inNorm, _postNorm, _postAttnNorm, _postFfnNorm, _normBias, _postNormBias, _parFfnNorm, _parFfnNormBias, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _qNorm, _kNorm, _qNormBias, _kNormBias, _sink, _alibiSlopes, _vNormOnes, _kvAProj, _kvANorm, _kvBProj, _qAProj, _qANorm, _qBProj, _gateW, _gateB, _upW, _upB, _downW, _downB, _pleInpGate, _pleProj, _plePostNorm, _ffnPreNorm2, _ffnPostNorm1, _ffnPostNorm2, _moeGateInpScale];
            foreach (Tensor? t in all) if (t is not null) yield return t;
            if (_moe is not null) foreach (Tensor t in _moe.EnumerateWeights()) yield return t;
            if (_crossAttn is not null) foreach (Tensor t in _crossAttn.EnumerateWeights()) yield return t;
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
            Project(backend, q, pre, _qW!, _qB, _cfg.LowVramQuant);
            if (hasOwnKv)
            {
                k = new(fullNorm ? new TensorShape(1, t, _cfg.KvDim) : new TensorShape(1, t, hkv, d), DType.F32);
                v = new(new TensorShape(1, t, hkv, d), DType.F32);
                Project(backend, k, pre, _kW!, _kB, _cfg.LowVramQuant);
                Project(backend, v, pre, _vW!, _vB, _cfg.LowVramQuant);
            }
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
                    backend.ApplyRopeInterleaved(q, cos, sin);
                    if (k is not null) backend.ApplyRopeInterleaved(k, cos, sin);
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
            Tensor cosTable, Tensor sinTable, ulong devicePos)
        {
            const int t = 1;
            int h = _cfg.HiddenSize;
            int hq = _cfg.NumHeads;
            int hkv = _cfg.NumKvHeads;
            int d = _cfg.HeadDim;
            int group = _cfg.KvGroup;
            TensorShape flat = new(1, t, h);

            Tensor pre = new(flat, DType.F32);
            PreSublayer(backend, pre, hidden, _inNorm, _normBias);

            bool fullNorm = _cfg.QkNorm && _cfg.QkNormFullDim;
            Tensor q = new(fullNorm ? new TensorShape(1, t, _cfg.QDim) : new TensorShape(1, t, hq, d), DType.F32);
            Tensor k = new(fullNorm ? new TensorShape(1, t, _cfg.KvDim) : new TensorShape(1, t, hkv, d), DType.F32);
            Tensor v = new(new TensorShape(1, t, hkv, d), DType.F32);
            Project(backend, q, pre, _qW!, _qB, _cfg.LowVramQuant);
            Project(backend, k, pre, _kW!, _kB, _cfg.LowVramQuant);
            Project(backend, v, pre, _vW!, _vB, _cfg.LowVramQuant);
            pre.Dispose();

            if (_cfg.QkNorm)
            {
                Tensor qN = new(new TensorShape(1, t, hq, d), DType.F32);
                Tensor kN = new(new TensorShape(1, t, hkv, d), DType.F32);
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

            // q/k are [1,1,heads,d] here (pre-Permute0213) — identical contiguous layout to the [1,heads,1,d]
            // the graph-decode RoPE kernel assumes (both collapse to heads*d contiguous when t=1), so no reshape.
            bool interleaved = _cfg.Rope == RopeStyle.Interleaved;
            backend.RopeApplyDecodeStep(q, cosTable, sinTable, _cfg.RotaryDim, interleaved, devicePos);
            backend.RopeApplyDecodeStep(k, cosTable, sinTable, _cfg.RotaryDim, interleaved, devicePos);

            Tensor qMh = new(new TensorShape(1, hq, t, d), DType.F32);
            Tensor kMh = new(new TensorShape(1, hkv, t, d), DType.F32);
            Tensor vMh = new(new TensorShape(1, hkv, t, d), DType.F32);
            backend.Permute0213(qMh, q, t, hq, d);
            backend.Permute0213(kMh, k, t, hkv, d);
            backend.Permute0213(vMh, v, t, hkv, d);
            q.Dispose(); k.Dispose(); v.Dispose();

            Tensor kFull = cache.KeyPrefix(layerIndex);
            Tensor vFull = cache.ValuePrefix(layerIndex);
            backend.KvCacheAppendDev(kFull, kMh, 0, devicePos);
            backend.KvCacheAppendDev(vFull, vMh, 0, devicePos);
            kMh.Dispose(); vMh.Dispose();

            float scale = _cfg.AttnScale;
            Tensor attn = new(new TensorShape(1, hq, t, d), DType.F32);
            backend.FlashAttentionDev(attn, qMh, kFull, vFull, 0, group, causal: true, 0, scale, devicePos);
            qMh.Dispose();

            Tensor attnFlat = new(new TensorShape(1, t, _cfg.QDim), DType.F32);
            backend.Permute0213(attnFlat, attn, hq, t, d);
            attn.Dispose();
            Tensor attnOut = new(flat, DType.F32);
            Project(backend, attnOut, attnFlat, _oW!, _oB, _cfg.LowVramQuant);
            attnFlat.Dispose();

            attnOut = PostSublayer(backend, attnOut, _postAttnNorm, flat);
            Tensor afterAttn = new(flat, DType.F32);
            backend.Add(afterAttn, hidden, attnOut);
            attnOut.Dispose();

            Tensor preMlp = new(flat, DType.F32);
            PreSublayer(backend, preMlp, afterAttn, _postNorm, _postNormBias);
            Tensor mlpOut = Mlp(backend, preMlp, t);

            mlpOut = PostSublayer(backend, mlpOut, _postFfnNorm, flat);
            Tensor result = new(flat, DType.F32);
            backend.Add(result, afterAttn, mlpOut);
            afterAttn.Dispose(); mlpOut.Dispose();
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
            Project(backend, q, pre, _qW!, _qB, _cfg.LowVramQuant);
            Project(backend, k, pre, _kW!, _kB, _cfg.LowVramQuant);
            Project(backend, v, pre, _vW!, _vB, _cfg.LowVramQuant);
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
                    backend.ApplyRopeInterleaved(q, cos, sin);
                    backend.ApplyRopeInterleaved(k, cos, sin);
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

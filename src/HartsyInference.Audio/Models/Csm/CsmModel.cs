using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Sampling;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.Csm;

/// <summary>Sesame CSM dual-transformer. The Llama-3.2-1B <see cref="Qwen2Model"/> backbone (bias-off)
/// predicts codebook 0 of the next frame from the running text+audio context; the Llama-100M decoder
/// fills codebooks 1..7 of that frame, conditioned on the backbone hidden + the sampled codebook-0 token.
/// Embedding tables, codebook heads, and the backbone→decoder projection live here; both transformer
/// bodies + the shared <see cref="NucleusSampler"/> are reused.
/// <para><b>Incremental decode:</b> the fast path is <see cref="CreateSession"/> + <see cref="StepFrame"/>, which
/// persists the backbone KV cache across frames (a <see cref="FixedKvCache"/>) so each frame feeds only the newly
/// appended rows through the backbone — O(n) over a song instead of the O(n²) full re-scan. It is numerically
/// identical to the full-context <see cref="GenerateFrame"/> (same kernels, same RoPE positions, same causal key
/// set); <see cref="GenerateFrame"/> is retained as a stateless one-shot / parity path.</para></summary>
public sealed unsafe class CsmModel : IDisposable
{
    private readonly CsmConfig _cfg;
    private readonly Qwen2Model _backbone;
    private readonly Qwen2Model _decoder;
    private int _disposed;

    private Tensor? _textEmbed;            // [textVocab, backboneHidden]
    private Tensor?[] _audioEmbed;         // numCodebooks × [audioVocab, backboneHidden]
    private Tensor? _c0Head;               // [audioVocab, backboneHidden]
    private Tensor? _projW;                // backboneHidden → decoderHidden
    private Tensor?[] _audioHead;          // (numCodebooks-1) × [audioVocab, decoderHidden]
    private Tensor? _muqW, _muqB;          // muq_linear [backboneHidden, muqDim] + bias (HeartMuLa only)
    private Tensor? _uncondText;           // unconditional_text_embedding [1, backboneHidden] (HeartMuLa only)

    public CsmConfig Config => _cfg;

    public CsmModel(CsmConfig cfg)
    {
        _cfg = cfg;
        _backbone = new Qwen2Model(cfg.Backbone);
        _decoder = new Qwen2Model(cfg.Decoder);
        _audioEmbed = new Tensor?[cfg.NumCodebooks];
        _audioHead = new Tensor?[cfg.NumCodebooks - 1];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        // Weights may arrive pre-quantized (the AR decode is memory-bandwidth-bound, so quantized projection/head
        // matrices let the fused GEMV read 2–4× fewer bytes/token). Quantization is done at LOAD time from a disk
        // cache (see CsmWeightCache) — NOT here — to avoid holding the BF16 model + a quantized copy at once (OOM).
        // This method just keeps whatever dtype it receives: quantized bodies/heads stay quantized (fused GEMV),
        // embeds/norms stay F16/F32 (host gather / vector ops).

        // Headless Llama bodies: load layers/norm only (no embed_tokens / lm_head inside the transformer). Quantized
        // layer weights are kept as-is by the transformer (the matmul backend runs the fused quant GEMV).
        _backbone.LoadWeightsHeadless(w, "backbone");
        _decoder.LoadWeightsHeadless(w, "decoder");
        // Embedding tables stay in their source dtype (BF16 stays BF16 — ~2.4 GB saved on HeartMuLa);
        // row reads convert per-element via CopyRowF32/AddRowF32.
        _textEmbed = KeepF32OrBf16(w["text_embeddings.weight"]);
        for (int i = 0; i < _cfg.NumCodebooks; i++) _audioEmbed[i] = KeepF32OrBf16(w[$"audio_embeddings.{i}.weight"]);
        // Heads + projection: keep quantized (fused GEMV) if quantized, else F32. The audio heads (128k-vocab) are the
        // single biggest per-frame weight-stream, so quantizing them is the largest bandwidth win.
        _c0Head = KeepQuantOrF32(w["codebook0_head.weight"]);
        _projW = KeepQuantOrF32(w["projection.weight"]);
        for (int i = 0; i < _cfg.NumCodebooks - 1; i++) _audioHead[i] = KeepQuantOrF32(w[$"audio_head.{i}.weight"]);
        // HeartMuLa-only conditioning weights (absent from CSM TTS checkpoints).
        if (w.TryGetValue("muq_linear.weight", out Tensor? mw)) _muqW = WhisperOps.EnsureF32(mw);
        if (w.TryGetValue("muq_linear.bias", out Tensor? mb)) _muqB = WhisperOps.EnsureF32(mb);
        if (w.TryGetValue("unconditional_text_embedding.weight", out Tensor? ue)) _uncondText = WhisperOps.EnsureF32(ue);
    }

    /// <summary>Keeps a quantized head/projection as-is (fused GEMV path); otherwise forces F32.</summary>
    private static Tensor KeepQuantOrF32(Tensor t) => t.DType.IsQuantized ? t : WhisperOps.EnsureF32(t);

    /// <summary>True when the HeartMuLa conditioning weights (muq_linear + unconditional_text_embedding) loaded.</summary>
    public bool HasMulaConditioning => _muqW is not null && _uncondText is not null;

    /// <summary>MuQ style row: <c>muq_linear(muqEmbed ?? zeros)</c> → <c>[1, 1, backboneHidden]</c>. Upstream always
    /// injects this row between tags and lyrics; with no reference audio the input is zeros (= the bias).</summary>
    public Tensor EmbedMuq(Tensor? muqEmbed512)
    {
        if (_muqW is null) throw new InvalidOperationException("muq_linear weights not loaded.");
        int h = _cfg.Backbone.HiddenSize;
        int muqDim = (int)_muqW.Shape[1];
        Tensor outT = new(new TensorShape(1, 1, h), DType.F32);
        float* op = (float*)outT.DataPointer;
        float* wp = (float*)_muqW.DataPointer;
        float* bp = _muqB is not null ? (float*)_muqB.DataPointer : null;
        float* ep = muqEmbed512 is not null ? (float*)muqEmbed512.DataPointer : null;
        for (int r = 0; r < h; r++)
        {
            float acc = bp is not null ? bp[r] : 0f;
            if (ep is not null)
            {
                float* row = wp + (long)r * muqDim;
                for (int c = 0; c < muqDim; c++) acc += row[c] * ep[c];
            }
            op[r] = acc;
        }
        return outT;
    }

    /// <summary>The learned unconditional text embedding repeated <paramref name="rows"/> times →
    /// <c>[1, rows, backboneHidden]</c>: upstream's CFG negative stream replaces every prompt position
    /// (tags, MuQ row, lyrics) with this single vector; audio frames stay shared.</summary>
    public Tensor UncondTextRows(int rows)
    {
        if (_uncondText is null) throw new InvalidOperationException("unconditional_text_embedding not loaded.");
        int h = _cfg.Backbone.HiddenSize;
        Tensor outT = new(new TensorShape(1, Math.Max(1, rows), h), DType.F32);
        float* op = (float*)outT.DataPointer;
        float* sp = (float*)_uncondText.DataPointer;
        for (int r = 0; r < rows; r++) Buffer.MemoryCopy(sp, op + (long)r * h, h * 4, h * 4);
        return outT;
    }

    /// <summary>Embeds a text token via the text table → <c>[1, 1, backboneHidden]</c>.</summary>
    public Tensor EmbedText(int tokenId) => EmbedRow(_textEmbed!, tokenId);

    /// <summary>Embeds one frame's 8 codebook values (summed across codebooks) → <c>[1, 1, backboneHidden]</c>.</summary>
    public Tensor EmbedAudioFrame(ReadOnlySpan<int> frameCodes)
    {
        int h = _cfg.Backbone.HiddenSize;
        Tensor outT = new(new TensorShape(1, 1, h), DType.F32);
        float* op = (float*)outT.DataPointer;
        for (int cb = 0; cb < frameCodes.Length && cb < _cfg.NumCodebooks; cb++)
            AddRowF32(_audioEmbed[cb]!, frameCodes[cb], op, h);
        return outT;
    }

    // Keep F32/BF16 tables as loaded (no F32 duplication); anything else casts up once.
    private static Tensor KeepF32OrBf16(Tensor t) =>
        t.DType == DType.F32 || t.DType == DType.BF16 ? t : t.CastTo(DType.F32);

    // dst = table[id] as F32 (row-clamped; table F32 or BF16).
    private static void CopyRowF32(Tensor table, int id, float* dst, int h)
    {
        long off = (long)Math.Clamp(id, 0, (int)table.Shape[0] - 1) * h;
        if (table.DType == DType.BF16)
        {
            ushort* row = (ushort*)table.DataPointer + off;
            for (int c = 0; c < h; c++) { uint b = (uint)row[c] << 16; dst[c] = *(float*)&b; }
        }
        else
        {
            Buffer.MemoryCopy((float*)table.DataPointer + off, dst, h * 4L, h * 4L);
        }
    }

    // dst += table[id] as F32 (row-clamped; table F32 or BF16).
    private static void AddRowF32(Tensor table, int id, float* dst, int h)
    {
        long off = (long)Math.Clamp(id, 0, (int)table.Shape[0] - 1) * h;
        if (table.DType == DType.BF16)
        {
            ushort* row = (ushort*)table.DataPointer + off;
            for (int c = 0; c < h; c++) { uint b = (uint)row[c] << 16; dst[c] += *(float*)&b; }
        }
        else
        {
            float* row = (float*)table.DataPointer + off;
            for (int c = 0; c < h; c++) dst[c] += row[c];
        }
    }

    /// <summary>Generates one full multi-codebook frame from the running <paramref name="contextEmbeds"/>
    /// <c>[1, T, backboneHidden]</c>. Returns the codebook values; callers check for end-of-utterance
    /// themselves (CSM: all codebooks equal <see cref="CsmConfig.CodebookEosToken"/>; HeartMuLa: codebook-0
    /// <c>&gt;= HeartMulaConfig.AudioEosToken</c>) since the stop convention differs per model.
    /// <para>Sampling params default to the config but are exposed per-call (<paramref name="temperature"/>,
    /// <paramref name="topK"/>, <paramref name="topP"/>). When <paramref name="uncondContext"/> is supplied and
    /// <paramref name="cfgScale"/> ≠ 1, classifier-free guidance is applied to every codebook's logits
    /// (<c>logit = uncond + g·(cond − uncond)</c>) via a parallel backbone+depth pass over the unconditional
    /// context (HeartMuLa's <c>cfg_scale</c>). CSM callers omit both and get the plain single-pass behavior.</para></summary>
    public int[] GenerateFrame(IBackend backend, Tensor contextEmbeds, ref uint rng,
        float? temperature = null, int? topK = null, float? topP = null,
        float cfgScale = 1f, Tensor? uncondContext = null)
    {
        bool useCfg = cfgScale != 1f && uncondContext is not null;
        int bt = (int)contextEmbeds.Shape[1];
        int bh = _cfg.Backbone.HiddenSize;
        using StreamingKvCache bCache = new(_cfg.Backbone.NumHiddenLayers, 1, _cfg.Backbone.NumKeyValueHeads, bt, _cfg.Backbone.HeadDim);
        Tensor hidden = _backbone.ForwardEmbeds(backend, contextEmbeds, 1, bt, 0, bCache);
        Tensor last = SliceLast(hidden, bh);
        hidden.Dispose();

        // Parallel unconditional backbone pass (CFG). Its last hidden anchors the uncond depth decoder.
        Tensor? uLast = null;
        if (useCfg)
        {
            int ubt = (int)uncondContext!.Shape[1];
            using StreamingKvCache ubCache = new(_cfg.Backbone.NumHiddenLayers, 1, _cfg.Backbone.NumKeyValueHeads, ubt, _cfg.Backbone.HeadDim);
            Tensor uHidden = _backbone.ForwardEmbeds(backend, uncondContext, 1, ubt, 0, ubCache);
            uLast = SliceLast(uHidden, bh);
            uHidden.Dispose();
        }

        // Stateless path: no persistent session → depth decoder stays eager (per-frame cache).
        return DecodeFrameTail(backend, last, uLast, ref rng, temperature, topK, topP, cfgScale, session: null, graphEnabled: false);
    }

    /// <summary>Persistent per-utterance decode state: the backbone KV cache (and, for CFG, a parallel
    /// unconditional cache) that survive across frames so <see cref="StepFrame"/> only feeds new rows. Dispose it
    /// when the utterance/song completes. Not thread-safe; one session per generation.</summary>
    public sealed class DecodeSession : IDisposable
    {
        internal FixedKvCache Backbone { get; }
        internal FixedKvCache? Uncond { get; }
        internal GraphStream? CondGraph;     // lazily created on the first graph-eligible (single-row) frame
        internal GraphStream? UncondGraph;
        // CFG dual-stream backbone graph (see BackboneLastDual): ONE [1,2,bh] fixed in/out buffer pair + one
        // shared devicePos serves both position-aligned streams — the whole batched B=2 step captures as one graph.
        internal GraphStream? DualGraph;
        // Persistent depth-decoder caches + graphs (graph mode): the depth KV cache must survive across frames so a
        // captured depth step keeps valid baked addresses; it is RESET (length→0) at the start of each frame's depth
        // decode. Lazily created on the first graph-eligible frame; null in the eager path (per-frame caches instead).
        internal FixedKvCache? DepthCond;
        internal FixedKvCache? DepthUncond;
        internal GraphStream? DepthCondGraph;
        internal GraphStream? DepthUncondGraph;
        private int _disposed;

        internal DecodeSession(FixedKvCache backbone, FixedKvCache? uncond)
        {
            Backbone = backbone;
            Uncond = uncond;
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            // Best-effort like GraphStream.Dispose (see its comment): a poisoned CUDA context makes cache
            // frees throw during unwind, masking the generation's original exception.
            CondGraph?.Dispose();
            UncondGraph?.Dispose();
            DualGraph?.Dispose();
            DepthCondGraph?.Dispose();
            DepthUncondGraph?.Dispose();
            try { DepthCond?.Dispose(); }
            catch (Exception ex) { Logs.Warning($"[CSM] Depth KV dispose failed during cleanup (continuing): {ex.Message}"); }
            try { DepthUncond?.Dispose(); }
            catch (Exception ex) { Logs.Warning($"[CSM] Depth uncond KV dispose failed during cleanup (continuing): {ex.Message}"); }
            try { Backbone.Dispose(); }
            catch (Exception ex) { Logs.Warning($"[CSM] Backbone KV dispose failed during cleanup (continuing): {ex.Message}"); }
            try { Uncond?.Dispose(); }
            catch (Exception ex) { Logs.Warning($"[CSM] Uncond KV dispose failed during cleanup (continuing): {ex.Message}"); }
        }
    }

    /// <summary>Allocates an incremental decode session. <paramref name="maxCondLen"/> is the maximum conditional
    /// context length (prefix rows + frames); <paramref name="maxUncondLen"/> is the unconditional stream length
    /// (frames only) and is ignored unless <paramref name="useCfg"/>.</summary>
    public DecodeSession CreateSession(int maxCondLen, int maxUncondLen, bool useCfg)
    {
        ThrowIfDisposed();
        FixedKvCache bb = new(_cfg.Backbone.NumHiddenLayers, 1, _cfg.Backbone.NumKeyValueHeads, _cfg.Backbone.HeadDim, Math.Max(1, maxCondLen));
        FixedKvCache? un = useCfg
            ? new FixedKvCache(_cfg.Backbone.NumHiddenLayers, 1, _cfg.Backbone.NumKeyValueHeads, _cfg.Backbone.HeadDim, Math.Max(1, maxUncondLen))
            : null;
        return new DecodeSession(bb, un);
    }

    /// <summary>Incremental frame decode: appends <paramref name="condNewEmbeds"/> <c>[1, tNew, bh]</c> to the
    /// session's persistent backbone cache at the current position and decodes one full 8-codebook frame. The first
    /// call feeds the whole prefix (text/lyrics/style); later calls feed just the previous frame's summed audio
    /// embedding (one row) — so the backbone cost is O(1) per frame, not O(n).
    /// <para>For CFG, pass <paramref name="uncondNewEmbeds"/> and <paramref name="cfgScale"/> ≠ 1: the unconditional
    /// stream is decoded in parallel through <see cref="DecodeSession.Uncond"/>. Set <paramref name="uncondStandalone"/>
    /// to run the uncond rows through a throwaway cache (used for the frame-0 dummy row, which must not persist so the
    /// uncond audio-frame positions match the stateless path exactly).</para>
    /// This is numerically identical to <see cref="GenerateFrame"/> over the equivalent full context.</summary>
    public int[] StepFrame(IBackend backend, DecodeSession session, Tensor condNewEmbeds, ref uint rng,
        float? temperature = null, int? topK = null, float? topP = null,
        float cfgScale = 1f, Tensor? uncondNewEmbeds = null, bool uncondStandalone = false)
    {
        ThrowIfDisposed();
        bool useCfg = cfgScale != 1f && uncondNewEmbeds is not null;
        int bh = _cfg.Backbone.HiddenSize;

        // CUDA-graph decode collapses each single-row backbone step (~layers×kernels launches, the per-frame hot
        // path over a whole song) into one graph replay. Only the steady-state one-row appends are graphed; the
        // frame-0 prefill (many rows) and the standalone uncond dummy stay eager. Env-gated with an eager fallback.
        bool graphEnabled = EnvSwitch.IsEnabled("HARTSY_CSM_GRAPH", defaultOn: true)
            && backend.GraphDecodeSupported && _backbone.SupportsGraphDecode(backend);

        // CFG steady state runs cond+uncond as ONE B=2 batched backbone step: the two streams are
        // position-aligned by construction (the pipeline mirrors upstream CFG — same prompt length, frames
        // appended to both), and the batched layers read each 3B weight matrix ONCE for both rows instead of
        // streaming the whole model twice per frame. The LM's per-frame GEMV traffic is at the bandwidth
        // roofline (nsys 2026-07-25: ~11.4 ms per stream-frame on the 4090), so this halves the dominant
        // per-frame cost in CFG runs. The batched step itself is graph-captured when eligible (see below);
        // otherwise it runs eager. Kill-switch HARTSY_CSM_CFG_BATCH=0 restores two-stream.
        if (useCfg && !uncondStandalone && (int)condNewEmbeds.Shape[1] == 1 && (int)uncondNewEmbeds!.Shape[1] == 1
            && EnvSwitch.IsEnabled("HARTSY_CSM_CFG_BATCH", defaultOn: true))
        {
            FixedKvCache ucB = session.Uncond ?? throw new InvalidOperationException("CFG StepFrame needs a CFG session (useCfg:true).");
            // Graph-captured batched step: the whole B=2 backbone step replays as ONE cuGraphLaunch, killing the
            // ~17 ms/frame of eager per-op host overhead the batched path reintroduced. Requires the two streams
            // to be position-aligned (they are by construction — the pipeline mirrors upstream CFG, same prefix
            // length, frames appended to both — so one devicePos serves both rows' rope AND kvLen; the equality
            // check is a safety net that falls back to the eager batched path, never wrong math). Kill-switch
            // HARTSY_CSM_CFG_GRAPH=0 restores the eager batched step below.
            if (graphEnabled && EnvSwitch.IsEnabled("HARTSY_CSM_CFG_GRAPH", defaultOn: true)
                && _backbone.SupportsDualGraphDecode(backend) && session.Backbone.CurrentLength == ucB.CurrentLength)
            {
                (Tensor lastG, Tensor uLastG) = BackboneLastDual(backend, session, ucB, condNewEmbeds, uncondNewEmbeds!, bh);
                return DecodeFrameTail(backend, lastG, uLastG, ref rng, temperature, topK, topP, cfgScale, session, graphEnabled);
            }
            Tensor stacked = new(new TensorShape(1, 2, bh), DType.F32);
            Buffer.MemoryCopy((void*)condNewEmbeds.DataPointer, (void*)stacked.DataPointer, (long)bh * 4, (long)bh * 4);
            Buffer.MemoryCopy((void*)uncondNewEmbeds.DataPointer, (float*)stacked.DataPointer + bh, (long)bh * 4, (long)bh * 4);
            Span<int> positions = [session.Backbone.CurrentLength, ucB.CurrentLength];
            Tensor both = _backbone.ForwardBatchDecode(backend, stacked, positions, [session.Backbone, ucB]);
            stacked.Dispose();
            Tensor lastB = new(new TensorShape(1, 1, bh), DType.F32);
            Tensor uLastB = new(new TensorShape(1, 1, bh), DType.F32);
            Buffer.MemoryCopy((void*)both.DataPointer, (void*)lastB.DataPointer, (long)bh * 4, (long)bh * 4);
            Buffer.MemoryCopy((float*)both.DataPointer + bh, (void*)uLastB.DataPointer, (long)bh * 4, (long)bh * 4);
            both.Dispose();
            return DecodeFrameTail(backend, lastB, uLastB, ref rng, temperature, topK, topP, cfgScale, session, graphEnabled);
        }

        Tensor last = BackboneLast(backend, session.Backbone, ref session.CondGraph, condNewEmbeds, graphEnabled, bh);

        Tensor? uLast = null;
        if (useCfg)
        {
            int utNew = (int)uncondNewEmbeds!.Shape[1];
            if (uncondStandalone)
            {
                using StreamingKvCache tmp = new(_cfg.Backbone.NumHiddenLayers, 1, _cfg.Backbone.NumKeyValueHeads, utNew, _cfg.Backbone.HeadDim);
                Tensor uHidden = _backbone.ForwardEmbeds(backend, uncondNewEmbeds, 1, utNew, 0, tmp);
                uLast = SliceLast(uHidden, bh);
                uHidden.Dispose();
            }
            else
            {
                FixedKvCache uc = session.Uncond ?? throw new InvalidOperationException("CFG StepFrame needs a CFG session (useCfg:true).");
                uLast = BackboneLast(backend, uc, ref session.UncondGraph, uncondNewEmbeds!, graphEnabled, bh);
            }
        }

        return DecodeFrameTail(backend, last, uLast, ref rng, temperature, topK, topP, cfgScale, session, graphEnabled);
    }

    /// <summary>Runs one backbone step for a stream and returns its last-position hidden <c>[1,1,bh]</c> (a fresh,
    /// caller-owned tensor). The eager path (prefill / graph disabled) is a plain <see cref="Qwen2Model.ForwardEmbeds"/>
    /// + slice. The graph path (steady-state single appended row) refreshes the stream's fixed input buffer and device
    /// position OUTSIDE any capture, warms a couple of frames eagerly-through-the-fixed-buffers so weights/buffers are
    /// resident, then captures the single-frame step once and replays it every subsequent frame — reading the hidden
    /// back with a non-freeing DtoH and returning a device copy the frame tail can consume and dispose.</summary>
    private Tensor BackboneLast(IBackend backend, FixedKvCache cache, ref GraphStream? gs, Tensor newEmbeds, bool graphEnabled, int bh)
    {
        int tNew = (int)newEmbeds.Shape[1];
        // Eager for prefill (many rows) or when graphing is off. The graph path also advances the cache length, so
        // cache.CurrentLength stays the single source of truth for the append position in both branches.
        if (!graphEnabled || tNew != 1)
        {
            Tensor hidden = _backbone.ForwardEmbeds(backend, newEmbeds, 1, tNew, cache.CurrentLength, cache);
            Tensor lastEager = SliceLast(hidden, bh);
            hidden.Dispose();
            return lastEager;
        }

        gs ??= new GraphStream(backend, bh);
        int pos = cache.CurrentLength;                          // absolute slot this frame's row appends at
        backend.CopyInto(gs.InEmbed, newEmbeds);                // refresh fixed input OUTSIDE capture
        backend.WriteDevicePos(gs.DevicePos, pos + 1, pos);     // {kvLen, qOffset} for this frame's append
        (Tensor cos, Tensor sin) = _backbone.EnsureRopeTableForGraphDecode(backend, cache.MaxSequenceLength);

        const int warmup = 2;   // run a few frames eagerly first so nothing allocates during capture
        if (gs.Graph is not null)
        {
            backend.LaunchGraph(gs.Graph);
        }
        else if (gs.Warmed >= warmup)
        {
            // Capture records the step without executing it; launch runs this frame. One capture is valid for every
            // later frame because RoPE/KV-append/attention read the position from gs.DevicePos (refreshed per frame).
            GraphStream s = gs;
            gs.Graph = backend.CaptureGraph(() =>
                _backbone.ForwardGraphDecodeStepEmbeds(backend, s.InEmbed, cache, cos, sin, s.DevicePos, s.OutHidden));
            if (gs.Graph is not null)
            {
                backend.LaunchGraph(gs.Graph);   // captured — run this frame via replay
            }
            else
            {
                // Backend declined to capture: run eagerly and keep warming (never retry-capture in a tight loop).
                _backbone.ForwardGraphDecodeStepEmbeds(backend, gs.InEmbed, cache, cos, sin, gs.DevicePos, gs.OutHidden);
            }
        }
        else
        {
            _backbone.ForwardGraphDecodeStepEmbeds(backend, gs.InEmbed, cache, cos, sin, gs.DevicePos, gs.OutHidden);
            gs.Warmed++;
        }
        cache.AdvanceLength(1);   // graph step appended one row; keep CurrentLength authoritative

        Tensor last = new(new TensorShape(1, 1, bh), DType.F32);
        backend.CopyInto(last, gs.OutHidden);   // fresh owned copy; the fixed buffer is reused next frame
        return last;
    }

    /// <summary>CFG dual-stream flavor of <see cref="BackboneLast"/>: runs cond+uncond as ONE position-aligned
    /// B=2 backbone step (stacked <c>[1,2,bh]</c> through the session's <see cref="DecodeSession.DualGraph"/> fixed
    /// buffers), captured once and replayed per frame. Combines both wins: the batched layers read each weight
    /// matrix ONCE for both rows (the CFG-batch win) AND the ~layers×kernels eager launches collapse into one
    /// graph replay (the graph win). Warmup frames run the same dual step eagerly through the SAME fixed buffers
    /// so every weight/cache is device-resident and the buffers are pool-bound before capture. Advances BOTH
    /// caches by one and returns fresh caller-owned cond/uncond last hiddens (each <c>[1,1,bh]</c>).</summary>
    private (Tensor last, Tensor uLast) BackboneLastDual(IBackend backend, DecodeSession session, FixedKvCache uncond,
        Tensor condNew, Tensor uncondNew, int bh)
    {
        FixedKvCache cond = session.Backbone;
        GraphStream gs = session.DualGraph ??= new GraphStream(backend, bh, rows: 2);
        int pos = cond.CurrentLength;
        Tensor stacked = new(new TensorShape(1, 2, bh), DType.F32);
        Buffer.MemoryCopy((void*)condNew.DataPointer, (void*)stacked.DataPointer, (long)bh * 4, (long)bh * 4);
        Buffer.MemoryCopy((void*)uncondNew.DataPointer, (float*)stacked.DataPointer + bh, (long)bh * 4, (long)bh * 4);
        backend.CopyInto(gs.InEmbed, stacked);                  // refresh fixed input OUTSIDE capture
        stacked.Dispose();
        backend.WriteDevicePos(gs.DevicePos, pos + 1, pos);     // shared {kvLen, qOffset} — both streams are aligned
        (Tensor cos, Tensor sin) = _backbone.EnsureRopeTableForGraphDecode(backend,
            Math.Max(cond.MaxSequenceLength, uncond.MaxSequenceLength));

        const int warmup = 2;   // run a few frames eagerly first so nothing allocates during capture
        if (gs.Graph is not null)
        {
            backend.LaunchGraph(gs.Graph);
        }
        else if (gs.Warmed >= warmup)
        {
            GraphStream s = gs;
            gs.Graph = backend.CaptureGraph(() =>
                _backbone.ForwardGraphDecodeStepDualEmbeds(backend, s.InEmbed, cond, uncond, cos, sin, s.DevicePos, s.OutHidden));
            if (gs.Graph is not null)
            {
                backend.LaunchGraph(gs.Graph);   // captured — run this frame via replay
            }
            else
            {
                // Backend declined to capture: run eagerly and keep warming (never retry-capture in a tight loop).
                _backbone.ForwardGraphDecodeStepDualEmbeds(backend, gs.InEmbed, cond, uncond, cos, sin, gs.DevicePos, gs.OutHidden);
            }
        }
        else
        {
            _backbone.ForwardGraphDecodeStepDualEmbeds(backend, gs.InEmbed, cond, uncond, cos, sin, gs.DevicePos, gs.OutHidden);
            gs.Warmed++;
        }
        cond.AdvanceLength(1);      // the dual step appended one row to each stream
        uncond.AdvanceLength(1);

        // Row split stays on-device (SliceRows, no stream sync); the frame tail's head projections consume the
        // resident copies and the one host sync per frame stays at the logits read, exactly like BackboneLast.
        Tensor last = new(new TensorShape(1, 1, bh), DType.F32);
        Tensor uLast = new(new TensorShape(1, 1, bh), DType.F32);
        backend.SliceRows(last, gs.OutHidden, 0);
        backend.SliceRows(uLast, gs.OutHidden, 1);
        return (last, uLast);
    }

    /// <summary>Shared frame tail: codebook-0 from the backbone <paramref name="last"/> hidden, then the depth
    /// decoder for codebooks 1..7, with optional CFG against <paramref name="uLast"/>. Disposes both hiddens.</summary>
    private int[] DecodeFrameTail(IBackend backend, Tensor last, Tensor? uLast, ref uint rng,
        float? temperature, int? topK, float? topP, float cfgScale, DecodeSession? session, bool graphEnabled)
    {
        float temp = temperature ?? _cfg.Temperature;
        int tk = topK ?? _cfg.TopK;
        float tp = topP ?? _cfg.TopP;
        bool useCfg = cfgScale != 1f && uLast is not null;
        int bh = _cfg.Backbone.HiddenSize;
        int dh = _cfg.Decoder.HiddenSize;

        // codebook 0 from the backbone head.
        Tensor c0Logits = WhisperOps.ProjectLinear(backend, last, _c0Head!, bias: null, 1, 1, bh, _cfg.AudioVocab);
        if (useCfg) CombineCfg(backend, c0Logits, uLast!, _c0Head!, bh, cfgScale);
        int c0 = NucleusSampler.Draw(new Span<float>((void*)c0Logits.DataPointer, _cfg.AudioVocab), _cfg.AudioVocab, temp, tk, tp, ref rng);
        c0Logits.Dispose();

        int[] frame = new int[_cfg.NumCodebooks];
        frame[0] = c0;
        // No early-exit here: the real stop condition (see CsmPipeline.Synthesize) is ALL codebooks of the frame
        // being CsmConfig.CodebookEosToken (0), not codebook-0 alone — upstream's generator.py appends an
        // all-zero "eos_frame" to the training context and checks `torch.all(sample == 0)` after decoding every
        // codebook, so codebook 0 being 0 doesn't by itself mean the rest will be; the depth decoder must always
        // run to know.

        // Decoder fills codebooks 1..7 INCREMENTALLY: one persistent KV cache per frame (cond + optional uncond),
        // prefilled with [proj(anchor), proj(embed_0(c0))] then fed one projected embedding per subsequent
        // codebook. The previous path created a fresh KV cache AND rebuilt/re-projected the whole growing sequence
        // on the host for every codebook — O(codebooks²) and the dominant cost of the HeartMuLa ~1hr/30s run.
        // Codebook value c_k is embedded via audio_embed[k]; the last position's hidden predicts the next codebook.
        // GPU-resident cache (matches the backbone): device in-place appends (KvCacheAppend) and device K/V prefixes
        // to FlashAttention — no per-layer/per-codebook D2H stream-drain + host prefix realloc + H2D re-upload that a
        // host StreamingKvCache incurred (~8 codebooks × decoder layers of cuStreamSynchronize stalls per frame).
        int maxDec = _cfg.NumCodebooks + 1;
        // Graph mode: the 6 single-token depth steps per frame are captured once and replayed (cond + uncond = two
        // more graphs). That needs a PERSISTENT depth KV cache (stable baked addresses) reset each frame. Eager mode
        // (stateless path / graph off / decoder not graph-eligible) keeps the per-frame cache.
        bool depthGraph = graphEnabled && session is not null && _decoder.SupportsGraphDecode(backend);
        FixedKvCache dCache;
        FixedKvCache? uCache;
        if (depthGraph)
        {
            session!.DepthCond ??= new FixedKvCache(_cfg.Decoder.NumHiddenLayers, 1, _cfg.Decoder.NumKeyValueHeads, _cfg.Decoder.HeadDim, maxDec);
            dCache = session.DepthCond;
            dCache.Reset();
            if (useCfg)
            {
                session.DepthUncond ??= new FixedKvCache(_cfg.Decoder.NumHiddenLayers, 1, _cfg.Decoder.NumKeyValueHeads, _cfg.Decoder.HeadDim, maxDec);
                uCache = session.DepthUncond;
                uCache.Reset();
            }
            else { uCache = null; }
        }
        else
        {
            dCache = new FixedKvCache(_cfg.Decoder.NumHiddenLayers, 1, _cfg.Decoder.NumKeyValueHeads, _cfg.Decoder.HeadDim, maxDec);
            uCache = useCfg ? new FixedKvCache(_cfg.Decoder.NumHiddenLayers, 1, _cfg.Decoder.NumKeyValueHeads, _cfg.Decoder.HeadDim, maxDec) : null;
        }
        try
        {
            Tensor dLast = DepthPrefill(backend, last, c0, bh, dh, dCache);
            Tensor? uLastDec = useCfg ? DepthPrefill(backend, uLast!, c0, bh, dh, uCache!) : null;

            for (int cb = 1; cb < _cfg.NumCodebooks; cb++)
            {
                Tensor cbLogits = WhisperOps.ProjectLinear(backend, dLast, _audioHead[cb - 1]!, bias: null, 1, 1, dh, _cfg.AudioVocab);
                dLast.Dispose();
                if (useCfg)
                {
                    Tensor uCbLogits = WhisperOps.ProjectLinear(backend, uLastDec!, _audioHead[cb - 1]!, bias: null, 1, 1, dh, _cfg.AudioVocab);
                    uLastDec!.Dispose();
                    BlendCfg(cbLogits, uCbLogits, cfgScale);
                    uCbLogits.Dispose();
                }
                int cbVal = NucleusSampler.Draw(new Span<float>((void*)cbLogits.DataPointer, _cfg.AudioVocab), _cfg.AudioVocab, temp, tk, tp, ref rng);
                cbLogits.Dispose();
                frame[cb] = cbVal;

                // Feed this codebook's embedding as the next decoder token (skip after the final codebook).
                if (cb < _cfg.NumCodebooks - 1)
                {
                    if (depthGraph)
                    {
                        dLast = DepthStepGraph(backend, _audioEmbed[cb]!, cbVal, bh, dh, dCache, ref session!.DepthCondGraph);
                        if (useCfg) uLastDec = DepthStepGraph(backend, _audioEmbed[cb]!, cbVal, bh, dh, uCache!, ref session!.DepthUncondGraph);
                    }
                    else
                    {
                        dLast = DepthStep(backend, _audioEmbed[cb]!, cbVal, bh, dh, dCache);
                        if (useCfg) uLastDec = DepthStep(backend, _audioEmbed[cb]!, cbVal, bh, dh, uCache!);
                    }
                }
            }
        }
        finally
        {
            if (!depthGraph) { dCache.Dispose(); uCache?.Dispose(); }   // persistent depth caches live on the session
        }
        last.Dispose();
        uLast?.Dispose();
        return frame;
    }

    /// <summary>Prefills a depth-decoder cache with [proj(anchor), proj(embed_0(c0))] (positions 0,1) and returns
    /// the last-position hidden <c>[1,1,dh]</c> (which predicts codebook 1).</summary>
    private Tensor DepthPrefill(IBackend backend, Tensor anchor, int c0, int bh, int dh, FixedKvCache cache)
    {
        Tensor curr = BuildDecoderSequence(anchor, new List<int> { c0 }, bh);   // [1, 2, bh]
        Tensor decInput = WhisperOps.ProjectLinear(backend, curr, _projW!, bias: null, 1, 2, bh, dh);
        curr.Dispose();
        Tensor dHidden = _decoder.ForwardEmbeds(backend, decInput, 1, 2, 0, cache);
        decInput.Dispose();
        Tensor dLast = SliceLast(dHidden, dh);
        dHidden.Dispose();
        return dLast;
    }

    /// <summary>Feeds one projected codebook embedding (audio_embed[k] row <paramref name="id"/>) as the next
    /// decoder token at the cache's current position, returning the new last hidden <c>[1,1,dh]</c>.</summary>
    private Tensor DepthStep(IBackend backend, Tensor table, int id, int bh, int dh, FixedKvCache cache)
    {
        Tensor embed = EmbedRow(table, id);                                     // [1, 1, bh]
        Tensor decInput = WhisperOps.ProjectLinear(backend, embed, _projW!, bias: null, 1, 1, bh, dh);
        embed.Dispose();
        Tensor dHidden = _decoder.ForwardEmbeds(backend, decInput, 1, 1, cache.CurrentLength, cache);
        decInput.Dispose();
        Tensor dLast = SliceLast(dHidden, dh);
        dHidden.Dispose();
        return dLast;
    }

    /// <summary>Graph-decode version of <see cref="DepthStep"/>: the codebook embedding is projected eagerly (outside
    /// capture — it depends on the just-sampled value), copied into the stream's fixed input buffer, then the depth
    /// decoder body runs via the captured single-frame graph (or eagerly during warmup). The projected-input →
    /// last-hidden step is identical math to <see cref="DepthStep"/>; only the launch mechanics differ. Advances the
    /// persistent depth cache by one and returns a fresh owned last hidden <c>[1,1,dh]</c>.</summary>
    private Tensor DepthStepGraph(IBackend backend, Tensor table, int id, int bh, int dh, FixedKvCache cache, ref GraphStream? gs)
    {
        Tensor embed = EmbedRow(table, id);                                              // [1,1,bh] host-built
        Tensor decInput = WhisperOps.ProjectLinear(backend, embed, _projW!, bias: null, 1, 1, bh, dh);
        embed.Dispose();

        gs ??= new GraphStream(backend, dh);
        int pos = cache.CurrentLength;
        backend.CopyInto(gs.InEmbed, decInput);                 // refresh fixed input OUTSIDE capture
        decInput.Dispose();
        backend.WriteDevicePos(gs.DevicePos, pos + 1, pos);
        (Tensor cos, Tensor sin) = _decoder.EnsureRopeTableForGraphDecode(backend, cache.MaxSequenceLength);

        const int warmup = 2;
        if (gs.Graph is not null)
        {
            backend.LaunchGraph(gs.Graph);
        }
        else if (gs.Warmed >= warmup)
        {
            GraphStream s = gs;
            gs.Graph = backend.CaptureGraph(() =>
                _decoder.ForwardGraphDecodeStepEmbeds(backend, s.InEmbed, cache, cos, sin, s.DevicePos, s.OutHidden));
            if (gs.Graph is not null) backend.LaunchGraph(gs.Graph);
            else _decoder.ForwardGraphDecodeStepEmbeds(backend, gs.InEmbed, cache, cos, sin, gs.DevicePos, gs.OutHidden);
        }
        else
        {
            _decoder.ForwardGraphDecodeStepEmbeds(backend, gs.InEmbed, cache, cos, sin, gs.DevicePos, gs.OutHidden);
            gs.Warmed++;
        }
        cache.AdvanceLength(1);

        Tensor dLast = new(new TensorShape(1, 1, dh), DType.F32);
        backend.CopyInto(dLast, gs.OutHidden);
        return dLast;
    }

    /// <summary>CFG-combines <paramref name="condLogits"/> in place with the uncond head projection of
    /// <paramref name="uLast"/> through <paramref name="head"/>: <c>logit = uncond + g·(cond − uncond)</c>.</summary>
    private void CombineCfg(IBackend backend, Tensor condLogits, Tensor uLast, Tensor head, int inDim, float g)
    {
        Tensor uLogits = WhisperOps.ProjectLinear(backend, uLast, head, bias: null, 1, 1, inDim, _cfg.AudioVocab);
        BlendCfg(condLogits, uLogits, g);
        uLogits.Dispose();
    }

    /// <summary>In-place CFG blend of already-computed cond/uncond logits: <c>cond = uncond + g·(cond − uncond)</c>.</summary>
    private void BlendCfg(Tensor condLogits, Tensor uncondLogits, float g)
    {
        float* c = (float*)condLogits.DataPointer;
        float* u = (float*)uncondLogits.DataPointer;
        for (int v = 0; v < _cfg.AudioVocab; v++) c[v] = u[v] + g * (c[v] - u[v]);
    }

    /// <summary>Teacher-forced parity probe (not used in inference). Runs the backbone over
    /// <paramref name="contextEmbeds"/>, returns the codebook-0 logits, then runs the depth decoder with the
    /// codebooks teacher-forced to <paramref name="forcedCodes"/> (length = <see cref="CsmConfig.NumCodebooks"/>;
    /// <c>forcedCodes[0]</c> is c0). Returns <c>(c0Logits[vocab], decoderLogits[NumCodebooks-1][vocab])</c>.</summary>
    public (float[] C0, float[][] Dec) DebugFrameLogits(IBackend backend, Tensor contextEmbeds, ReadOnlySpan<int> forcedCodes)
    {
        int bt = (int)contextEmbeds.Shape[1];
        int bh = _cfg.Backbone.HiddenSize;
        int vocab = _cfg.AudioVocab;
        using StreamingKvCache bCache = new(_cfg.Backbone.NumHiddenLayers, 1, _cfg.Backbone.NumKeyValueHeads, bt, _cfg.Backbone.HeadDim);
        Tensor hidden = _backbone.ForwardEmbeds(backend, contextEmbeds, 1, bt, 0, bCache);
        Tensor last = SliceLast(hidden, bh);
        hidden.Dispose();

        Tensor c0Logits = WhisperOps.ProjectLinear(backend, last, _c0Head!, bias: null, 1, 1, bh, vocab);
        float[] c0 = new float[vocab];
        new Span<float>((void*)c0Logits.DataPointer, vocab).CopyTo(c0);
        c0Logits.Dispose();

        int dh = _cfg.Decoder.HiddenSize;
        float[][] dec = new float[_cfg.NumCodebooks - 1][];
        List<int> decoderSeq = new() { forcedCodes[0] };
        for (int cb = 1; cb < _cfg.NumCodebooks; cb++)
        {
            Tensor curr = BuildDecoderSequence(last, decoderSeq, bh);
            Tensor decInput = WhisperOps.ProjectLinear(backend, curr, _projW!, bias: null, 1, (int)curr.Shape[1], bh, dh);
            curr.Dispose();
            int dt = (int)decInput.Shape[1];
            using StreamingKvCache dCache = new(_cfg.Decoder.NumHiddenLayers, 1, _cfg.Decoder.NumKeyValueHeads, dt, _cfg.Decoder.HeadDim);
            Tensor dHidden = _decoder.ForwardEmbeds(backend, decInput, 1, dt, 0, dCache);
            decInput.Dispose();
            Tensor dLast = SliceLast(dHidden, dh);
            dHidden.Dispose();
            Tensor cbLogits = WhisperOps.ProjectLinear(backend, dLast, _audioHead[cb - 1]!, bias: null, 1, 1, dh, vocab);
            dLast.Dispose();
            float[] row = new float[vocab];
            new Span<float>((void*)cbLogits.DataPointer, vocab).CopyTo(row);
            cbLogits.Dispose();
            dec[cb - 1] = row;
            decoderSeq.Add(forcedCodes[cb]);
        }
        last.Dispose();
        return (c0, dec);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
        if (_textEmbed is not null) yield return _textEmbed;
        foreach (Tensor? e in _audioEmbed) if (e is not null) yield return e;
        if (_c0Head is not null) yield return _c0Head;
        if (_projW is not null) yield return _projW;
        foreach (Tensor? hd in _audioHead) if (hd is not null) yield return hd;
        if (_muqW is not null) yield return _muqW;
        if (_muqB is not null) yield return _muqB;
        if (_uncondText is not null) yield return _uncondText;
    }

    /// <summary>True when the LM matmul weights are quantized (GGUF). The quantized fused GEMV re-reads the weight
    /// each call, so it MUST be GPU-resident (see <see cref="EnumerateDeviceWeights"/>); the bf16 path caches its
    /// F16 cast on first use and stays fast without pinning, so callers should preload only when this is true (pinning
    /// the 5.4 GB bf16 bodies non-evictably would OOM smaller cards).</summary>
    public bool WeightsQuantized => _c0Head?.DType.IsQuantized ?? false;

    /// <summary>Weights consumed by device matmuls (the per-token decode hot path) — for <see cref="IBackend.PreloadWeights"/>.
    /// Excludes the embed tables and MuQ/uncond conditioning vectors, which are gathered/read <b>host-side</b>
    /// (<see cref="CopyRowF32"/>/<see cref="AddRowF32"/>/<see cref="EmbedMuq"/>): pinning those to device wastes VRAM
    /// (the embed tables are ~2.4 GB) for no speed gain.</summary>
    public IEnumerable<Tensor> EnumerateDeviceWeights()
    {
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
        if (_c0Head is not null) yield return _c0Head;
        if (_projW is not null) yield return _projW;
        foreach (Tensor? hd in _audioHead) if (hd is not null) yield return hd;
    }

    /// <summary>Decoder sequence (in backbone-hidden space, pre-projection) = the backbone last hidden, then the
    /// per-codebook audio embedding of each already-sampled codebook value of this frame. Codebook <c>j</c> uses
    /// audio table <c>j</c> (upstream <c>_embed_audio(j, code)</c> = <c>audio_embeddings(code + j·vocab)</c>).</summary>
    private Tensor BuildDecoderSequence(Tensor lastHidden, List<int> seq, int bh)
    {
        int n = 1 + seq.Count;
        Tensor outT = new(new TensorShape(1, n, bh), DType.F32);
        float* op = (float*)outT.DataPointer;
        Buffer.MemoryCopy((void*)lastHidden.DataPointer, op, bh * 4, bh * 4);
        for (int i = 0; i < seq.Count; i++)
        {
            CopyRowF32(_audioEmbed[i]!, seq[i], op + (long)(i + 1) * bh, bh);
        }
        return outT;
    }

    private Tensor EmbedRow(Tensor table, int id)
    {
        int h = (int)table.Shape[1];
        Tensor outT = new(new TensorShape(1, 1, h), DType.F32);
        CopyRowF32(table, id, (float*)outT.DataPointer, h);
        return outT;
    }

    private static Tensor SliceLast(Tensor hidden, int h)
    {
        int t = (int)hidden.Shape[1];
        Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        Buffer.MemoryCopy((float*)hidden.DataPointer + (long)(t - 1) * h, (void*)last.DataPointer, h * 4, h * 4);
        return last;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _backbone.Dispose();
        _decoder.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(CsmModel));
    }
}

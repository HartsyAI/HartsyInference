using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Sampling;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
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
        // Headless Llama bodies: load layers/norm only (no embed_tokens / lm_head inside the transformer).
        _backbone.LoadWeightsHeadless(w, "backbone");
        _decoder.LoadWeightsHeadless(w, "decoder");
        _textEmbed = WhisperOps.EnsureF32(w["text_embeddings.weight"]);
        for (int i = 0; i < _cfg.NumCodebooks; i++) _audioEmbed[i] = WhisperOps.EnsureF32(w[$"audio_embeddings.{i}.weight"]);
        _c0Head = WhisperOps.EnsureF32(w["codebook0_head.weight"]);
        _projW = WhisperOps.EnsureF32(w["projection.weight"]);
        for (int i = 0; i < _cfg.NumCodebooks - 1; i++) _audioHead[i] = WhisperOps.EnsureF32(w[$"audio_head.{i}.weight"]);
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
        {
            float* tab = (float*)_audioEmbed[cb]!.DataPointer;
            int id = Math.Clamp(frameCodes[cb], 0, _cfg.AudioVocab - 1);
            float* row = tab + (long)id * h;
            for (int c = 0; c < h; c++) op[c] += row[c];
        }
        return outT;
    }

    /// <summary>Generates one full 8-codebook frame from the running <paramref name="contextEmbeds"/>
    /// <c>[1, T, backboneHidden]</c>. Returns the 8 codebook values; the last codebook-0 value equals
    /// <see cref="CsmConfig.AudioEosToken"/> at end-of-utterance.
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

        return DecodeFrameTail(backend, last, uLast, ref rng, temperature, topK, topP, cfgScale);
    }

    /// <summary>Persistent per-utterance decode state: the backbone KV cache (and, for CFG, a parallel
    /// unconditional cache) that survive across frames so <see cref="StepFrame"/> only feeds new rows. Dispose it
    /// when the utterance/song completes. Not thread-safe; one session per generation.</summary>
    public sealed class DecodeSession : IDisposable
    {
        internal FixedKvCache Backbone { get; }
        internal FixedKvCache? Uncond { get; }
        private int _disposed;

        internal DecodeSession(FixedKvCache backbone, FixedKvCache? uncond)
        {
            Backbone = backbone;
            Uncond = uncond;
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            Backbone.Dispose();
            Uncond?.Dispose();
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

        int tNew = (int)condNewEmbeds.Shape[1];
        Tensor hidden = _backbone.ForwardEmbeds(backend, condNewEmbeds, 1, tNew, session.Backbone.CurrentLength, session.Backbone);
        Tensor last = SliceLast(hidden, bh);
        hidden.Dispose();

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
                Tensor uHidden = _backbone.ForwardEmbeds(backend, uncondNewEmbeds, 1, utNew, uc.CurrentLength, uc);
                uLast = SliceLast(uHidden, bh);
                uHidden.Dispose();
            }
        }

        return DecodeFrameTail(backend, last, uLast, ref rng, temperature, topK, topP, cfgScale);
    }

    /// <summary>Shared frame tail: codebook-0 from the backbone <paramref name="last"/> hidden, then the depth
    /// decoder for codebooks 1..7, with optional CFG against <paramref name="uLast"/>. Disposes both hiddens.</summary>
    private int[] DecodeFrameTail(IBackend backend, Tensor last, Tensor? uLast, ref uint rng,
        float? temperature, int? topK, float? topP, float cfgScale)
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
        if (c0 == _cfg.AudioEosToken) { last.Dispose(); uLast?.Dispose(); return frame; }

        // Decoder fills codebooks 1..7. Upstream `generate_frame`: build curr_h = [last_h, embed(c0), …] in the
        // backbone hidden space, apply `projection` to the WHOLE sequence, run the depth decoder over it (fresh
        // KV cache, max len = num_codebooks), and matmul the last hidden by `audio_head[i-1]`. The decoder input
        // is the full backbone-hidden (3072) — no truncation.
        List<int> decoderSeq = new() { c0 };
        for (int cb = 1; cb < _cfg.NumCodebooks; cb++)
        {
            Tensor cbLogits = DepthCodebookLogits(backend, last, decoderSeq, bh, dh, cb);
            if (useCfg)
            {
                Tensor uCbLogits = DepthCodebookLogits(backend, uLast!, decoderSeq, bh, dh, cb);
                BlendCfg(cbLogits, uCbLogits, cfgScale);
                uCbLogits.Dispose();
            }
            int cbVal = NucleusSampler.Draw(new Span<float>((void*)cbLogits.DataPointer, _cfg.AudioVocab), _cfg.AudioVocab, temp, tk, tp, ref rng);
            cbLogits.Dispose();
            frame[cb] = cbVal;
            decoderSeq.Add(cbVal);
        }
        last.Dispose();
        uLast?.Dispose();
        return frame;
    }

    /// <summary>Runs one depth-decoder step for codebook <paramref name="cb"/> from the backbone last-hidden
    /// <paramref name="anchor"/> and the sampled <paramref name="decoderSeq"/>, returning that codebook's logits.</summary>
    private Tensor DepthCodebookLogits(IBackend backend, Tensor anchor, List<int> decoderSeq, int bh, int dh, int cb)
    {
        Tensor curr = BuildDecoderSequence(anchor, decoderSeq, bh);     // [1, 1+seq, bh]
        Tensor decInput = WhisperOps.ProjectLinear(backend, curr, _projW!, bias: null, 1, (int)curr.Shape[1], bh, dh);
        curr.Dispose();
        int dt = (int)decInput.Shape[1];
        using StreamingKvCache dCache = new(_cfg.Decoder.NumHiddenLayers, 1, _cfg.Decoder.NumKeyValueHeads, dt, _cfg.Decoder.HeadDim);
        Tensor dHidden = _decoder.ForwardEmbeds(backend, decInput, 1, dt, 0, dCache);
        decInput.Dispose();
        Tensor dLast = SliceLast(dHidden, dh);
        dHidden.Dispose();
        Tensor cbLogits = WhisperOps.ProjectLinear(backend, dLast, _audioHead[cb - 1]!, bias: null, 1, 1, dh, _cfg.AudioVocab);
        dLast.Dispose();
        return cbLogits;
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
            float* tab = (float*)_audioEmbed[i]!.DataPointer;
            int id = Math.Clamp(seq[i], 0, _cfg.AudioVocab - 1);
            float* row = tab + (long)id * bh;
            float* dst = op + (long)(i + 1) * bh;
            for (int c = 0; c < bh; c++) dst[c] = row[c];
        }
        return outT;
    }

    private Tensor EmbedRow(Tensor table, int id)
    {
        int h = (int)table.Shape[1];
        Tensor outT = new(new TensorShape(1, 1, h), DType.F32);
        int clamped = Math.Clamp(id, 0, (int)table.Shape[0] - 1);
        Buffer.MemoryCopy((float*)table.DataPointer + (long)clamped * h, (void*)outT.DataPointer, h * 4, h * 4);
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

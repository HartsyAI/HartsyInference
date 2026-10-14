using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.Chatterbox;

/// <summary>Chatterbox T3 — the text→speech-token AR LM. A headless Llama backbone (<see cref="Qwen2Model"/>)
/// runs over the concatenated <c>[cond ++ text ++ speech]</c> embedding sequence (each with learned position
/// embeddings) and the speech head autoregressively emits S3 tokens until the stop token. Conditioning is a
/// projected 256-d voice-encoder embedding plus an exaggeration scalar. Reuses <see cref="NucleusSampler"/>
/// (temp / top-p / min-p / repetition penalty).</summary>
public sealed unsafe class ChatterboxT3 : IDisposable
{
    // Upstream T3.inference embeds the start-of-speech token twice at speech position 0: prepare_input_embeds already
    // ends with initial_speech_tokens (t3.py:258-267) and t3.py:305-313 appends another bos_embed.
    private const int UpstreamPrefillBosCount = 2;

    private readonly ChatterboxConfig _cfg;
    private readonly Qwen2Model _backbone;
    private readonly ChatterboxPerceiver _perceiver = new();
    private Tensor? _textEmb, _speechEmb, _textPos, _speechPos, _speechHead, _spkrEncW, _spkrEncB, _emotionFcW;
    private int _disposed;

    public ChatterboxT3(ChatterboxConfig cfg)
    {
        _cfg = cfg;
        _backbone = new Qwen2Model(cfg.T3);
    }

    public int HiddenSize => _cfg.T3.HiddenSize;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _backbone.LoadWeightsHeadless(w, "tfmr");
        _textEmb = WhisperOps.EnsureF32(w["text_emb.weight"]);
        _speechEmb = WhisperOps.EnsureF32(w["speech_emb.weight"]);
        _textPos = WhisperOps.EnsureF32(w["text_pos_emb.emb.weight"]);
        _speechPos = WhisperOps.EnsureF32(w["speech_pos_emb.emb.weight"]);
        _speechHead = WhisperOps.EnsureF32(w["speech_head.weight"]);
        _spkrEncW = WhisperOps.EnsureF32(w["cond_enc.spkr_enc.weight"]);
        _spkrEncB = w.TryGetValue("cond_enc.spkr_enc.bias", out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
        _emotionFcW = WhisperOps.EnsureF32(w["cond_enc.emotion_adv_fc.weight"]);
        if (w.ContainsKey("cond_enc.perceiver.pre_attention_query")) _perceiver.LoadWeights(w);
    }

    /// <summary>Builds the conditioning prefix embeddings <c>[1, 2, hidden]</c> = [speaker_proj, emotion_proj]
    /// (the prompt-speech perceiver branch is deferred).</summary>
    public Tensor BuildCond(IBackend backend, Tensor speakerEmbed, float exaggeration, ReadOnlySpan<int> promptSpeechTokens)
    {
        int h = _cfg.T3.HiddenSize;
        Tensor spkRow = speakerEmbed.Reshape(new TensorShape(1, 1, _cfg.SpeakerEmbedDim));
        Tensor spk = WhisperOps.ProjectLinear(backend, spkRow, _spkrEncW!, _spkrEncB, 1, 1, _cfg.SpeakerEmbedDim, h);
        Tensor emoIn = new(new TensorShape(1, 1, 1), DType.F32);
        *(float*)emoIn.DataPointer = exaggeration;
        Tensor emo = WhisperOps.ProjectLinear(backend, emoIn, _emotionFcW!, null, 1, 1, 1, h);
        emoIn.Dispose();

        // Cond-prompt: speech_emb(prompt) + speech_pos_emb(0..) → Perceiver resampler → 32 tokens. The reference
        // assembles cond as cat([speaker(1), perceiver_prompt(32), emotion(1)]).
        Tensor? promptOut = null;
        int promptLen = 0;
        if (promptSpeechTokens.Length > 0)
        {
            int pn = promptSpeechTokens.Length;
            Tensor promptEmb = new(new TensorShape(1, pn, h), DType.F32);
            float* pe = (float*)promptEmb.DataPointer;
            for (int i = 0; i < pn; i++) AddEmbPlusPos(pe + (long)i * h, _speechEmb!, promptSpeechTokens[i], _speechPos!, i, h);
            promptOut = _perceiver.Forward(backend, promptEmb);   // [1, 32, h]
            promptEmb.Dispose();
            promptLen = (int)promptOut.Shape[1];
        }

        int condLen = 1 + promptLen + 1;
        Tensor cond = new(new TensorShape(1, condLen, h), DType.F32);
        byte* dst = (byte*)cond.DataPointer;
        Buffer.MemoryCopy((void*)spk.DataPointer, dst, h * 4, h * 4);
        long off = (long)h * 4;
        if (promptOut is not null)
        {
            long pbytes = (long)promptLen * h * 4;
            Buffer.MemoryCopy((void*)promptOut.DataPointer, dst + off, pbytes, pbytes);
            off += pbytes;
            promptOut.Dispose();
        }
        Buffer.MemoryCopy((void*)emo.DataPointer, dst + off, h * 4, h * 4);
        spk.Dispose(); emo.Dispose();
        return cond;
    }

    /// <summary>Autoregressively generates S3 speech tokens from text token ids + a speaker embedding. When
    /// <paramref name="cfgWeight"/> is positive, runs Chatterbox's real dual-stream CFG — a second forward
    /// pass over a parallel KV cache with the text embedding zeroed (speaker/emotion/prompt/speech-start
    /// conditioning stays shared with the conditional stream, matching upstream <c>t3.py</c>'s
    /// <c>text_emb[1].zero_()</c>), combined per step as <c>cond + cfgWeight * (cond - uncond)</c> — note
    /// this is Chatterbox's own formula, not CSM's <c>uncond + g*(cond-uncond)</c> convention.
    /// <see cref="LLM.Transformer.GenericTransformer.ForwardEmbeds"/> is batch=1 only, so the uncond stream
    /// runs as an independent forward over its own cache rather than a batched call — mathematically
    /// equivalent to a real batch=2 forward since causal self-attention has no cross-batch term.</summary>
    public List<int> GenerateSpeechTokens(IBackend backend, ReadOnlySpan<int> textTokens, Tensor speakerEmbed,
        float exaggeration, int maxNew, int seed = 0, ReadOnlySpan<int> promptSpeechTokens = default, float cfgWeight = 0f)
    {
        if (_textEmb is null) throw new InvalidOperationException("ChatterboxT3 not loaded.");
        bool useCfg = cfgWeight > 0f;
        int vocab = _cfg.SpeechVocab;
        using DecodeState state = Prefill(backend, textTokens, speakerEmbed, exaggeration, promptSpeechTokens, useCfg,
            UpstreamPrefillBosCount, maxNew);
        uint rng = DeterministicRng.Seed(seed);
        List<int> tokens = new(maxNew);
        // t3.py:316 starts generated_ids at the BOS token, so the repetition penalty covers it from step 0.
        HashSet<int> seen = [_cfg.StartSpeechToken];
        float[] logits = new float[vocab];
        float[]? logitsUncond = useCfg ? new float[vocab] : null;
        for (int step = 0; step < maxNew; step++)
        {
            state.Logits(backend, logits, logitsUncond);
            if (useCfg)
                for (int v = 0; v < vocab; v++)
                    logits[v] += cfgWeight * (logits[v] - logitsUncond![v]);
            if (_cfg.RepetitionPenalty != 1f)
                foreach (int tok in seen)
                    logits[tok] = logits[tok] > 0 ? logits[tok] / _cfg.RepetitionPenalty : logits[tok] * _cfg.RepetitionPenalty;
            int next = NucleusSampler.Draw(logits, vocab, _cfg.Temperature, 0, _cfg.TopP, ref rng, -1, _cfg.MinP);

            if (next == _cfg.StopSpeechToken) break;
            tokens.Add(next); seen.Add(next);
            if (!state.Step(backend, next, tokens.Count)) break;
        }
        return tokens;
    }

    /// <summary>Raw speech-head logits <c>[steps, 2, SpeechVocab]</c> (row 0 conditional, row 1 the CFG
    /// unconditional stream) before every step when <paramref name="forcedTokens"/> are fed in order: the
    /// teacher-forced view of <see cref="GenerateSpeechTokens"/> that parity tests compare against upstream.</summary>
    internal float[] TeacherForcedLogits(IBackend backend, ReadOnlySpan<int> textTokens, Tensor speakerEmbed,
        float exaggeration, ReadOnlySpan<int> promptSpeechTokens, ReadOnlySpan<int> forcedTokens,
        int bosCount = UpstreamPrefillBosCount)
    {
        if (_textEmb is null) throw new InvalidOperationException("ChatterboxT3 not loaded.");
        int vocab = _cfg.SpeechVocab;
        int steps = forcedTokens.Length + 1;
        float[] result = new float[steps * 2 * vocab];
        using DecodeState state = Prefill(backend, textTokens, speakerEmbed, exaggeration, promptSpeechTokens, true,
            bosCount, forcedTokens.Length);
        for (int step = 0; step < steps; step++)
        {
            state.Logits(backend, result.AsSpan(step * 2 * vocab, vocab), result.AsSpan((step * 2 + 1) * vocab, vocab));
            // The cache is sized for every forced token, so Step never reports it full here.
            if (step < forcedTokens.Length) state.Step(backend, forcedTokens[step], step + 1);
        }
        return result;
    }

    /// <summary>Builds <c>[cond] ++ [text+pos] ++ bosCount x [speechStart+pos0]</c> (the uncond stream zeroes only the
    /// text token embedding) and runs the prefill forward of each stream.</summary>
    private DecodeState Prefill(IBackend backend, ReadOnlySpan<int> textTokens, Tensor speakerEmbed, float exaggeration,
        ReadOnlySpan<int> promptSpeechTokens, bool useCfg, int bosCount, int maxNew)
    {
        int h = _cfg.T3.HiddenSize;
        Tensor cond = BuildCond(backend, speakerEmbed, exaggeration, promptSpeechTokens);
        int condLen = (int)cond.Shape[1];
        int tt = textTokens.Length;
        int prefill = condLen + tt + bosCount;
        Tensor embeds = new(new TensorShape(1, prefill, h), DType.F32);
        Tensor? embedsUncond = useCfg ? new Tensor(new TensorShape(1, prefill, h), DType.F32) : null;
        float* ep = (float*)embeds.DataPointer;
        float* eup = useCfg ? (float*)embedsUncond!.DataPointer : null;
        Buffer.MemoryCopy((void*)cond.DataPointer, ep, (long)condLen * h * 4, (long)condLen * h * 4);
        if (useCfg) Buffer.MemoryCopy((void*)cond.DataPointer, eup, (long)condLen * h * 4, (long)condLen * h * 4);
        cond.Dispose();
        for (int i = 0; i < tt; i++)
        {
            AddEmbPlusPos(ep + (long)(condLen + i) * h, _textEmb!, textTokens[i], _textPos!, i, h);
            if (useCfg) AddPosOnly(eup + (long)(condLen + i) * h, _textPos!, i, h);
        }
        for (int b = 0; b < bosCount; b++)
        {
            long row = (long)(condLen + tt + b) * h;
            AddEmbPlusPos(ep + row, _speechEmb!, _cfg.StartSpeechToken, _speechPos!, 0, h);
            if (useCfg) AddEmbPlusPos(eup + row, _speechEmb!, _cfg.StartSpeechToken, _speechPos!, 0, h);
        }

        int cacheCap = Math.Min(_cfg.T3.MaxPositionEmbeddings, prefill + maxNew + 4);
        IKvCache? cache = null, cacheUncond = null;
        Tensor? hidden = null;
        try
        {
            cache = _backbone.CreateDecodeCache(cacheCap);
            cacheUncond = useCfg ? _backbone.CreateDecodeCache(cacheCap) : null;
            hidden = _backbone.ForwardEmbeds(backend, embeds, 1, prefill, 0, cache);
            Tensor? hiddenUncond = useCfg ? _backbone.ForwardEmbeds(backend, embedsUncond!, 1, prefill, 0, cacheUncond!) : null;
            return new DecodeState(this, cache, cacheUncond, hidden, hiddenUncond, cacheCap);
        }
        catch
        {
            hidden?.Dispose();
            cache?.Dispose();
            cacheUncond?.Dispose();
            throw;
        }
        finally
        {
            embeds.Dispose();
            embedsUncond?.Dispose();
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_textEmb, _speechEmb, _textPos, _speechPos, _speechHead, _spkrEncW, _spkrEncB, _emotionFcW];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
    }

    private static void AddEmbPlusPos(float* dst, Tensor emb, int token, Tensor pos, int posIdx, int h)
    {
        float* er = (float*)emb.DataPointer + (long)token * h;
        float* pr = (float*)pos.DataPointer + (long)posIdx * h;
        for (int i = 0; i < h; i++) dst[i] = er[i] + pr[i];
    }

    /// <summary>CFG's uncond text region: position embedding only, zero token embedding (upstream's
    /// <c>text_emb[1].zero_()</c>).</summary>
    private static void AddPosOnly(float* dst, Tensor pos, int posIdx, int h)
    {
        float* pr = (float*)pos.DataPointer + (long)posIdx * h;
        for (int i = 0; i < h; i++) dst[i] = pr[i];
    }

    private void ProjectLast(IBackend backend, Tensor hidden, Span<float> dst)
    {
        int h = _cfg.T3.HiddenSize;
        int t = (int)hidden.Shape[1];
        Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        Buffer.MemoryCopy((float*)hidden.DataPointer + (long)(t - 1) * h, (void*)last.DataPointer, h * 4, h * 4);
        Tensor logits = WhisperOps.ProjectLinear(backend, last, _speechHead!, null, 1, 1, h, _cfg.SpeechVocab);
        last.Dispose();
        new ReadOnlySpan<float>((void*)logits.DataPointer, _cfg.SpeechVocab).CopyTo(dst);
        logits.Dispose();
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _backbone.Dispose();
        _perceiver.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <summary>The cond (and optional CFG uncond) KV caches plus the last hidden state of each stream.</summary>
    private sealed class DecodeState(ChatterboxT3 owner, IKvCache cache, IKvCache? cacheUncond, Tensor hidden,
        Tensor? hiddenUncond, int cacheCap) : IDisposable
    {
        private Tensor _hidden = hidden;
        private Tensor? _hiddenUncond = hiddenUncond;

        /// <summary>Projects the last position of each stream through the speech head.</summary>
        public void Logits(IBackend backend, Span<float> cond, Span<float> uncond)
        {
            owner.ProjectLast(backend, _hidden, cond);
            if (_hiddenUncond is not null && !uncond.IsEmpty) owner.ProjectLast(backend, _hiddenUncond, uncond);
        }

        /// <summary>Feeds <paramref name="token"/> at speech position <paramref name="speechPos"/> to every stream;
        /// false once the cache is full.</summary>
        public bool Step(IBackend backend, int token, int speechPos)
        {
            int h = owner._cfg.T3.HiddenSize;
            using Tensor step1 = new(new TensorShape(1, 1, h), DType.F32);
            AddEmbPlusPos((float*)step1.DataPointer, owner._speechEmb!, token, owner._speechPos!, speechPos, h);
            Tensor next = owner._backbone.ForwardEmbeds(backend, step1, 1, 1, cache.CurrentLength, cache);
            _hidden.Dispose();
            _hidden = next;
            if (cacheUncond is not null)
            {
                Tensor nextUncond = owner._backbone.ForwardEmbeds(backend, step1, 1, 1, cacheUncond.CurrentLength, cacheUncond);
                _hiddenUncond!.Dispose();
                _hiddenUncond = nextUncond;
            }
            return cache.CurrentLength < cacheCap - 2;
        }

        public void Dispose()
        {
            _hidden.Dispose();
            _hiddenUncond?.Dispose();
            cache.Dispose();
            cacheUncond?.Dispose();
        }
    }
}

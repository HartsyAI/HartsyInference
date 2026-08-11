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
        int h = _cfg.T3.HiddenSize;
        bool useCfg = cfgWeight > 0f;
        Tensor cond = BuildCond(backend, speakerEmbed, exaggeration, promptSpeechTokens);
        int condLen = (int)cond.Shape[1];
        int tt = textTokens.Length;

        // Prefill embeds = [cond] ++ [text+pos] ++ [speechStart+pos0]. The uncond stream is byte-identical
        // except the text region keeps its position embedding but zeroes the token embedding.
        int prefill = condLen + tt + 1;
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
        AddEmbPlusPos(ep + (long)(condLen + tt) * h, _speechEmb!, _cfg.StartSpeechToken, _speechPos!, 0, h);
        if (useCfg) AddEmbPlusPos(eup + (long)(condLen + tt) * h, _speechEmb!, _cfg.StartSpeechToken, _speechPos!, 0, h);

        int cacheCap = Math.Min(_cfg.T3.MaxPositionEmbeddings, prefill + maxNew + 4);
        using IKvCache cache = _backbone.CreateDecodeCache(cacheCap);
        using IKvCache? cacheUncond = useCfg ? _backbone.CreateDecodeCache(cacheCap) : null;
        uint rng = DeterministicRng.Seed(seed);
        List<int> tokens = new(maxNew);
        HashSet<int> seen = new();

        Tensor hidden = _backbone.ForwardEmbeds(backend, embeds, 1, prefill, 0, cache);
        embeds.Dispose();
        Tensor? hiddenUncond = useCfg ? _backbone.ForwardEmbeds(backend, embedsUncond!, 1, prefill, 0, cacheUncond!) : null;
        embedsUncond?.Dispose();
        for (int step = 0; step < maxNew; step++)
        {
            Tensor last = SliceLast(hidden, h); hidden.Dispose();
            Tensor logitsT = WhisperOps.ProjectLinear(backend, last, _speechHead!, null, 1, 1, h, _cfg.SpeechVocab);
            last.Dispose();
            Span<float> logits = new((void*)logitsT.DataPointer, _cfg.SpeechVocab);
            if (useCfg)
            {
                Tensor lastU = SliceLast(hiddenUncond!, h); hiddenUncond!.Dispose();
                Tensor logitsUncondT = WhisperOps.ProjectLinear(backend, lastU, _speechHead!, null, 1, 1, h, _cfg.SpeechVocab);
                lastU.Dispose();
                Span<float> logitsUncond = new((void*)logitsUncondT.DataPointer, _cfg.SpeechVocab);
                for (int v = 0; v < _cfg.SpeechVocab; v++)
                    logits[v] += cfgWeight * (logits[v] - logitsUncond[v]);
                logitsUncondT.Dispose();
            }
            if (_cfg.RepetitionPenalty != 1f)
                foreach (int tok in seen)
                    logits[tok] = logits[tok] > 0 ? logits[tok] / _cfg.RepetitionPenalty : logits[tok] * _cfg.RepetitionPenalty;
            int next = NucleusSampler.Draw(logits, _cfg.SpeechVocab, _cfg.Temperature, 0, _cfg.TopP, ref rng, -1, _cfg.MinP);
            logitsT.Dispose();

            if (next == _cfg.StopSpeechToken) break;
            tokens.Add(next); seen.Add(next);

            Tensor step1 = new(new TensorShape(1, 1, h), DType.F32);
            AddEmbPlusPos((float*)step1.DataPointer, _speechEmb!, next, _speechPos!, tokens.Count, h);
            hidden = _backbone.ForwardEmbeds(backend, step1, 1, 1, cache.CurrentLength, cache);
            if (useCfg)
                hiddenUncond = _backbone.ForwardEmbeds(backend, step1, 1, 1, cacheUncond!.CurrentLength, cacheUncond);
            step1.Dispose();
            if (cache.CurrentLength >= cacheCap - 2) break;
        }
        hidden.Dispose();
        hiddenUncond?.Dispose();
        return tokens;
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
        _perceiver.Dispose();
        GC.SuppressFinalize(this);
    }
}

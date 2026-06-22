using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.PocketTts;

/// <summary>The Pocket-TTS <c>FlowLMModel</c>: an autoregressive streaming transformer over <b>continuous</b>
/// audio latents. Each time step embeds its input (a text token early in the sequence, or the previous frame's
/// latent projected into model width), runs the headless transformer (<see cref="Qwen2Model.ForwardEmbeds"/> with
/// a <see cref="StreamingKvCache"/>), and feeds the resulting hidden into the per-frame
/// <see cref="PocketTtsFlowHead"/> — a <b>regression</b> head (not token logits) that samples the next continuous
/// latent via a short flow ODE. Generation runs text prefix → optional voice-prompt priming → AR latent loop, then
/// decodes the collected latents through <see cref="MimiContinuousLatent.DecodeFromLatent"/> to 24 kHz PCM.
///
/// <para>Backbone, latent, and head dims all come from <see cref="PocketTtsConfig"/>; the real <c>b6369a24</c>
/// values drop in unchanged at load.</para></summary>
public sealed unsafe class PocketTtsModel : IDisposable
{
    private readonly PocketTtsConfig _cfg;
    private readonly Qwen2Config _lmCfg;
    private readonly Qwen2Model _transformer;
    private readonly PocketTtsFlowHead _flowHead;
    private readonly MimiContinuousLatent _codec;
    private readonly int _dModel;
    private readonly int _latentDim;

    private Tensor? _textEmbed;       // [VocabSize, DModel] — text-token embedding table.
    private Tensor? _latentInW;       // [DModel, LatentDim] — projects a previous latent into model width.
    private Tensor? _latentInB;       // [DModel]
    private Tensor? _bosLatentEmbed;  // [DModel] — learned start-of-audio embedding (kicks off the AR loop).
    private int _disposed;

    public PocketTtsConfig Config => _cfg;
    public MimiContinuousLatent Codec => _codec;
    public int SampleRate => _cfg.SampleRate;

    public PocketTtsModel(PocketTtsConfig cfg)
    {
        _cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
        if (cfg.DModel <= 0) throw new ArgumentException("PocketTtsConfig.DModel must be > 0.", nameof(cfg));
        if (cfg.LatentDim <= 0) throw new ArgumentException("PocketTtsConfig.LatentDim must be > 0.", nameof(cfg));
        _dModel = cfg.DModel;
        _latentDim = cfg.LatentDim;
        _lmCfg = cfg.ToTransformerConfig();
        _transformer = new Qwen2Model(_lmCfg);
        _flowHead = new PocketTtsFlowHead(cfg);
        _codec = new MimiContinuousLatent(cfg.Mimi);
    }

    /// <summary>Loads all sub-module weights. Prefixes follow the documented <c>flow_lm.*</c> / <c>mimi.*</c>
    /// config sub-trees; reconcile the exact key strings against the real <c>b6369a24</c> safetensors at first
    /// load. The transformer body is loaded <b>headless</b> (no token <c>lm_head</c> — Pocket-TTS regresses
    /// latents, it does not emit token logits).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w,
        string lmPrefix = "flow_lm.transformer",
        string flowHeadPrefix = "flow_lm.flow_head")
    {
        ThrowIfDisposed();
        if (w is null) throw new ArgumentNullException(nameof(w));

        _transformer.LoadWeightsHeadless(w, lmPrefix);
        _textEmbed = WhisperOps.EnsureF32(w[$"{lmPrefix}.text_embed.weight"]);
        _latentInW = WhisperOps.EnsureF32(w[$"{lmPrefix}.latent_in.weight"]);
        _latentInB = WhisperOps.EnsureF32(w[$"{lmPrefix}.latent_in.bias"]);
        _bosLatentEmbed = WhisperOps.EnsureF32(w[$"{lmPrefix}.audio_bos"]);

        _flowHead.LoadWeights(w, flowHeadPrefix);
        _codec.LoadWeights(w);
    }

    /// <summary>Generates 24 kHz mono PCM for <paramref name="textTokens"/>. <paramref name="voiceState"/> primes
    /// the AR state with a voice prompt's continuous latents (<c>[1, LatentDim, T_prime]</c>, e.g. from
    /// <see cref="MimiContinuousLatent.EncodeToLatent"/> or a stored built-in voice embedding); pass <c>null</c>
    /// for the default voice. <paramref name="numFrames"/> bounds the AR loop; when ≤ 0 it derives a length from
    /// the text and is clamped to <see cref="PocketTtsConfig.MaxFrames"/>.</summary>
    public float[] Generate(IBackend backend, ReadOnlySpan<int> textTokens, Tensor? voiceState, int seed, int numFrames = 0)
    {
        ThrowIfDisposed();
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        EnsureLoaded();

        int primeFrames = voiceState is not null ? (int)voiceState.Shape[2] : 0;
        int genFrames = numFrames > 0 ? numFrames : DeriveFrameCount(textTokens.Length);
        genFrames = Math.Min(genFrames, _cfg.MaxFrames);

        int maxSeq = textTokens.Length + primeFrames + genFrames + 1;
        using StreamingKvCache cache = new(_lmCfg.NumHiddenLayers, 1, _lmCfg.NumKeyValueHeads, maxSeq, _lmCfg.HeadDim);

        int pos = 0;

        // 1) Text prefix — embed tokens then prime the transformer (hidden discarded; only KV matters).
        if (textTokens.Length > 0)
        {
            Tensor textEmb = EmbedText(textTokens);
            Tensor primed = _transformer.ForwardEmbeds(backend, textEmb, 1, textTokens.Length, pos, cache);
            primed.Dispose();     // text prefix only primes the KV cache; its hidden is unused.
            textEmb.Dispose();
            pos += textTokens.Length;
        }

        // 2) Voice-prompt priming — project each prompt latent into model width and prime, one frame at a time so
        //    the AR state matches what teacher-forced latents would produce.
        if (voiceState is not null)
        {
            for (int f = 0; f < primeFrames; f++)
            {
                Tensor latent = ExtractFrame(voiceState, f);
                Tensor emb = ProjectLatentIn(backend, latent);
                latent.Dispose();
                Tensor primed = _transformer.ForwardEmbeds(backend, emb, 1, 1, pos, cache);
                primed.Dispose();
                emb.Dispose();
                pos++;
            }
        }

        // 3) Autoregressive latent loop. Step 0 is kicked off by the learned audio-BOS embedding; every later step
        //    feeds the previous sampled latent back in.
        Tensor[] frames = new Tensor[genFrames];
        Tensor stepEmb = BosEmbed();
        for (int f = 0; f < genFrames; f++)
        {
            Tensor hidden = _transformer.ForwardEmbeds(backend, stepEmb, 1, 1, pos, cache);
            stepEmb.Dispose();
            pos++;

            Tensor cond = LastHidden(hidden);
            hidden.Dispose();
            Tensor latent = _flowHead.SampleLatent(backend, cond, unchecked(seed + f));
            cond.Dispose();
            frames[f] = latent;

            if (f + 1 < genFrames) stepEmb = ProjectLatentIn(backend, latent);
        }
        if (genFrames == 0) stepEmb.Dispose();

        // 4) Assemble [1, LatentDim, genFrames] and decode to PCM.
        Tensor latents = AssembleLatents(frames, genFrames);
        foreach (Tensor t in frames) t.Dispose();

        Tensor pcm = _codec.DecodeFromLatent(backend, latents, 1, genFrames);
        latents.Dispose();

        float[] audio = new float[pcm.ElementCount];
        fixed (float* dst = audio)
            Buffer.MemoryCopy((void*)pcm.DataPointer, dst, audio.Length * 4L, pcm.ElementCount * 4L);
        pcm.Dispose();
        return audio;
    }

    private Tensor EmbedText(ReadOnlySpan<int> tokens)
    {
        Tensor emb = new(new TensorShape(1, tokens.Length, _dModel), DType.F32);
        float* op = (float*)emb.DataPointer;
        float* ep = (float*)_textEmbed!.DataPointer;
        int vocab = Math.Max(1, _cfg.VocabSize);
        for (int s = 0; s < tokens.Length; s++)
        {
            int id = tokens[s];
            if ((uint)id >= (uint)vocab)
                throw new ArgumentException($"text token id {id} out of range [0, {vocab}).");
            Buffer.MemoryCopy(ep + (long)id * _dModel, op + (long)s * _dModel, _dModel * 4L, _dModel * 4L);
        }
        return emb;
    }

    private Tensor BosEmbed()
    {
        Tensor emb = new(new TensorShape(1, 1, _dModel), DType.F32);
        Buffer.MemoryCopy((void*)_bosLatentEmbed!.DataPointer, (void*)emb.DataPointer, _dModel * 4L, _dModel * 4L);
        return emb;
    }

    /// <summary>Projects a latent <c>[1, LatentDim]</c> into a <c>[1, 1, DModel]</c> input embedding.</summary>
    private Tensor ProjectLatentIn(IBackend backend, Tensor latent)
    {
        Tensor latent3 = latent.Reshape(new TensorShape(1, 1, _latentDim));
        Tensor proj = WhisperOps.ProjectLinear(backend, latent3, _latentInW!, _latentInB, 1, 1, _latentDim, _dModel);
        latent3.Dispose();
        return proj;     // already [1, 1, DModel]
    }

    /// <summary>Slices the last time position of a <c>[1, T, DModel]</c> hidden into <c>[1, DModel]</c>.</summary>
    private Tensor LastHidden(Tensor hidden)
    {
        int t = (int)hidden.Shape[1];
        Tensor last = new(new TensorShape(1, _dModel), DType.F32);
        float* src = (float*)hidden.DataPointer + (long)(t - 1) * _dModel;
        Buffer.MemoryCopy(src, (void*)last.DataPointer, _dModel * 4L, _dModel * 4L);
        return last;
    }

    private Tensor ExtractFrame(Tensor latentsCf, int frame)
    {
        int tFrames = (int)latentsCf.Shape[2];
        Tensor f = new(new TensorShape(1, _latentDim), DType.F32);
        float* src = (float*)latentsCf.DataPointer;
        float* dst = (float*)f.DataPointer;
        for (int c = 0; c < _latentDim; c++) dst[c] = src[(long)c * tFrames + frame];
        return f;
    }

    private Tensor AssembleLatents(Tensor[] frames, int n)
    {
        Tensor latents = new(new TensorShape(1, _latentDim, Math.Max(1, n)), DType.F32);
        float* dst = (float*)latents.DataPointer;
        if (n == 0) return latents;
        int tFrames = n;
        for (int f = 0; f < n; f++)
        {
            float* src = (float*)frames[f].DataPointer;
            for (int c = 0; c < _latentDim; c++) dst[(long)c * tFrames + f] = src[c];
        }
        return latents;
    }

    private int DeriveFrameCount(int textLen)
    {
        // No public EOS predictor for the gated checkpoint — bound generation by a text-proportional estimate.
        // ~3 frames per token at 12.5 Hz is a conservative upper bound for natural English pacing.
        int est = Math.Max(_cfg.FrameRateHz, textLen * 3);
        return Math.Min(est, _cfg.MaxFrames);
    }

    private void EnsureLoaded()
    {
        if (_textEmbed is null || _latentInW is null || _bosLatentEmbed is null)
            throw new InvalidOperationException("PocketTtsModel weights not loaded.");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(PocketTtsModel));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _transformer.Dispose();
        _textEmbed = null;
        _latentInW = null;
        _latentInB = null;
        _bosLatentEmbed = null;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _transformer.EnumerateWeights()) yield return t;
        if (_textEmbed is not null) yield return _textEmbed;
        if (_latentInW is not null) yield return _latentInW;
        if (_latentInB is not null) yield return _latentInB;
        if (_bosLatentEmbed is not null) yield return _bosLatentEmbed;
        foreach (Tensor t in _flowHead.EnumerateWeights()) yield return t;
        foreach (Tensor t in _codec.EnumerateWeights()) yield return t;
    }
}

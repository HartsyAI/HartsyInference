using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.MeloTts;
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>MeloTTS pipeline — the MeloTTS extended text encoder (phoneme + tone + language + BERT) feeding the
/// shared VITS duration / flow / HiFi-GAN stages, with a speaker embedding injected throughout. Duration is the
/// Bert-VITS2 blend <c>logw = sdp·sdp_ratio + dp·(1-sdp_ratio)</c>. The caller runs the per-language BERT +
/// phonemizer and passes phoneme/tone/language ids (already blank-interleaved) + the two BERT feature streams.</summary>
public sealed unsafe class MeloTtsPipeline : IDisposable
{
    private readonly MeloTtsConfig _cfg;
    private readonly MeloTtsTextEncoder _enc;
    private readonly VitsDurationPredictor _dp;
    private readonly VitsStochasticDurationPredictor _sdp;
    private readonly VitsTransformerFlow _flow;
    private readonly VitsHiFiGan _dec;
    private Tensor? _embG;     // speaker embedding [num_speakers, gin]
    private int _disposed;

    public MeloTtsPipeline(MeloTtsConfig cfg)
    {
        _cfg = cfg;
        _enc = new MeloTtsTextEncoder(cfg);
        _dp = new VitsDurationPredictor(cfg.Core);
        _sdp = new VitsStochasticDurationPredictor(cfg.Core);
        _flow = new VitsTransformerFlow(cfg.Core);
        _dec = new VitsHiFiGan(cfg.Core);
    }

    public int SampleRate => _cfg.Core.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _enc.LoadWeights(w);
        _dp.LoadWeights(w);
        _sdp.LoadWeights(w);
        _flow.LoadWeights(w);
        _dec.LoadWeights(w);
        _embG = w.TryGetValue("emb_g.weight", out Tensor? g) ? g : null;
    }

    /// <summary>Synthesizes audio. <paramref name="bert"/>/<paramref name="jaBert"/> are channels-first
    /// <c>[BertDim, T]</c>/<c>[JaBertDim, T]</c> feature tensors aligned to the phoneme sequence;
    /// <paramref name="speakerId"/> indexes the speaker embedding table.</summary>
    public float[] Synthesize(IBackend backend, ReadOnlySpan<int> phonemes, ReadOnlySpan<int> tones,
        ReadOnlySpan<int> languages, Tensor bert, Tensor jaBert, int speakerId = 0, float? lengthScale = null,
        float? noiseScale = null, int seed = 0)
    {
        ThrowIfDisposed();
        int inter = _cfg.Core.InterChannels, tx = phonemes.Length;
        float ls = lengthScale ?? _cfg.LengthScale, ns = noiseScale ?? _cfg.NoiseScale;

        using Tensor? g = SpeakerEmbedding(speakerId);

        (Tensor hidden, Tensor mP, Tensor logsP) = _enc.Forward(backend, phonemes, tones, languages, bert, jaBert, g);

        // Duration: blend the stochastic and deterministic predictors (Bert-VITS2 sdp_ratio).
        float[] logwSdp = _sdp.Forward(backend, hidden, tx, _cfg.NoiseScaleW, seed, g);
        float[] logwDp = _dp.Forward(backend, hidden, tx, g);
        hidden.Dispose();
        float r = _cfg.SdpRatio;
        float[] logw = new float[tx];
        for (int j = 0; j < tx; j++) logw[j] = logwSdp[j] * r + logwDp[j] * (1f - r);

        int[] durations = VitsLengthRegulator.Durations(logw, ls);
        int ty = VitsLengthRegulator.TotalFrames(durations);
        Tensor mPe = new(new TensorShape(1, inter, ty), DType.F32);
        Tensor logsPe = new(new TensorShape(1, inter, ty), DType.F32);
        VitsLengthRegulator.Expand((float*)mP.DataPointer, (float*)mPe.DataPointer, inter, tx, durations, ty);
        VitsLengthRegulator.Expand((float*)logsP.DataPointer, (float*)logsPe.DataPointer, inter, tx, durations, ty);
        mP.Dispose(); logsP.Dispose();

        uint rng = DeterministicRng.Seed(seed + 1);     // prior noise; SDP consumed `seed`
        Tensor zP = new(new TensorShape(1, inter, ty), DType.F32);
        float* zp = (float*)zP.DataPointer;
        float* mp = (float*)mPe.DataPointer;
        float* lp = (float*)logsPe.DataPointer;
        for (long n = 0; n < (long)inter * ty; n++)
            zp[n] = mp[n] + DeterministicRng.NextGaussian(ref rng) * MathF.Exp(lp[n]) * ns;
        mPe.Dispose(); logsPe.Dispose();

        Tensor z = _flow.Reverse(backend, zP, ty, g);     // transforms zP in place and returns it
        float[] audio = _dec.Forward(backend, z, ty, g); z.Dispose();
        return audio;
    }

    // g = emb_g(sid).unsqueeze(-1) → [1, gin, 1]. Null when the model has no speaker table (single anonymous voice).
    private Tensor? SpeakerEmbedding(int sid)
    {
        if (_embG is null) return null;
        int gin = _cfg.Core.GinChannels;
        Tensor g = new(new TensorShape(1, gin, 1), DType.F32);
        float* src = (float*)_embG.DataPointer, dst = (float*)g.DataPointer;
        for (int c = 0; c < gin; c++) dst[c] = src[(long)sid * gin + c];
        return g;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _enc.EnumerateWeights()) yield return t;
        foreach (Tensor t in _dp.EnumerateWeights()) yield return t;
        foreach (Tensor t in _sdp.EnumerateWeights()) yield return t;
        foreach (Tensor t in _flow.EnumerateWeights()) yield return t;
        foreach (Tensor t in _dec.EnumerateWeights()) yield return t;
        if (_embG is not null) yield return _embG;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(MeloTtsPipeline));
    }
}

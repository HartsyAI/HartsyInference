using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.MeloTts;
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>MeloTTS pipeline — the MeloTTS extended text encoder (phoneme + tone + language + BERT) feeding
/// the shared VITS duration / flow / HiFi-GAN stages. Takes phoneme/tone/language ids + the two BERT feature
/// streams (caller runs the per-language BERT + phonemizer). Multispeaker <c>g</c> conditioning and the
/// SDP/DP blend are staged.</summary>
public sealed unsafe class MeloTtsPipeline : IDisposable
{
    private readonly MeloTtsConfig _cfg;
    private readonly MeloTtsTextEncoder _enc;
    private readonly VitsDurationPredictor _dp;
    private readonly VitsFlow _flow;
    private readonly VitsHiFiGan _dec;
    private int _disposed;

    public MeloTtsPipeline(MeloTtsConfig cfg)
    {
        _cfg = cfg;
        _enc = new MeloTtsTextEncoder(cfg);
        _dp = new VitsDurationPredictor(cfg.Core);
        _flow = new VitsFlow(cfg.Core);
        _dec = new VitsHiFiGan(cfg.Core);
    }

    public int SampleRate => _cfg.Core.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _enc.LoadWeights(w);
        _dp.LoadWeights(w);
        _flow.LoadWeights(w);
        _dec.LoadWeights(w);
    }

    /// <summary>Synthesizes audio. <paramref name="bert"/>/<paramref name="jaBert"/> are channels-first
    /// <c>[BertDim, T]</c>/<c>[JaBertDim, T]</c> feature tensors aligned to the phoneme sequence.</summary>
    public float[] Synthesize(IBackend backend, ReadOnlySpan<int> phonemes, ReadOnlySpan<int> tones,
        ReadOnlySpan<int> languages, Tensor bert, Tensor jaBert, float? lengthScale = null,
        float? noiseScale = null, int seed = 0)
    {
        ThrowIfDisposed();
        int inter = _cfg.Core.InterChannels, tx = phonemes.Length;
        float ls = lengthScale ?? _cfg.LengthScale, ns = noiseScale ?? _cfg.NoiseScale;

        (Tensor hidden, Tensor mP, Tensor logsP) = _enc.Forward(backend, phonemes, tones, languages, bert, jaBert);
        float[] logw = _dp.Forward(backend, hidden, tx);
        hidden.Dispose();

        int[] durations = VitsLengthRegulator.Durations(logw, ls);
        int ty = VitsLengthRegulator.TotalFrames(durations);
        Tensor mPe = new(new TensorShape(1, inter, ty), DType.F32);
        Tensor logsPe = new(new TensorShape(1, inter, ty), DType.F32);
        VitsLengthRegulator.Expand((float*)mP.DataPointer, (float*)mPe.DataPointer, inter, tx, durations, ty);
        VitsLengthRegulator.Expand((float*)logsP.DataPointer, (float*)logsPe.DataPointer, inter, tx, durations, ty);
        mP.Dispose(); logsP.Dispose();

        uint rng = DeterministicRng.Seed(seed);
        Tensor zP = new(new TensorShape(1, inter, ty), DType.F32);
        float* zp = (float*)zP.DataPointer;
        float* mp = (float*)mPe.DataPointer;
        float* lp = (float*)logsPe.DataPointer;
        for (long n = 0; n < (long)inter * ty; n++)
            zp[n] = mp[n] + DeterministicRng.NextGaussian(ref rng) * MathF.Exp(lp[n]) * ns;
        mPe.Dispose(); logsPe.Dispose();

        Tensor z = _flow.Reverse(backend, zP, ty); zP.Dispose();
        float[] audio = _dec.Forward(backend, z, ty); z.Dispose();
        return audio;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _enc.EnumerateWeights()) yield return t;
        foreach (Tensor t in _dp.EnumerateWeights()) yield return t;
        foreach (Tensor t in _flow.EnumerateWeights()) yield return t;
        foreach (Tensor t in _dec.EnumerateWeights()) yield return t;
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

using HartsyInference.Audio.Dsp;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Vits;

/// <summary>VITS <c>SynthesizerTrn</c> inference assembly: text encoder → duration predictor → monotonic
/// length regulation → prior sample → reverse flow → HiFi-GAN decoder. The shared TTS core behind Piper /
/// MeloTTS / GPT-SoVITS / OpenVoice. Single-speaker (deterministic duration) path; the stochastic duration
/// predictor and speaker conditioning are staged. Reuses every sub-module's <c>IBackend</c> ops.</summary>
public sealed unsafe class VitsSynthesizer : IDisposable
{
    private readonly VitsConfig _cfg;
    private readonly VitsTextEncoder _enc;
    private readonly VitsDurationPredictor _dp;
    private readonly VitsFlow _flow;
    private readonly VitsHiFiGan _dec;
    private int _disposed;

    public VitsSynthesizer(VitsConfig cfg)
    {
        _cfg = cfg;
        _enc = new VitsTextEncoder(cfg);
        _dp = new VitsDurationPredictor(cfg);
        _flow = new VitsFlow(cfg);
        _dec = new VitsHiFiGan(cfg);
    }

    public int SampleRate => _cfg.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _enc.LoadWeights(w);
        _dp.LoadWeights(w);
        _flow.LoadWeights(w);
        _dec.LoadWeights(w);
    }

    /// <summary>Synthesizes audio from interspersed phoneme ids (use <see cref="VitsLengthRegulator.Intersperse"/>).</summary>
    public float[] Infer(IBackend backend, ReadOnlySpan<int> tokens, float? lengthScale = null,
        float? noiseScale = null, int seed = 0)
    {
        ThrowIfDisposed();
        int inter = _cfg.InterChannels, tx = tokens.Length;
        float ls = lengthScale ?? _cfg.LengthScale, ns = noiseScale ?? _cfg.NoiseScale;

        (Tensor hidden, Tensor mP, Tensor logsP) = _enc.Forward(backend, tokens);
        float[] logw = _dp.Forward(backend, hidden, tx);
        hidden.Dispose();

        int[] durations = VitsLengthRegulator.Durations(logw, ls);
        int ty = VitsLengthRegulator.TotalFrames(durations);

        Tensor mPe = new(new TensorShape(1, inter, ty), DType.F32);
        Tensor logsPe = new(new TensorShape(1, inter, ty), DType.F32);
        VitsLengthRegulator.Expand((float*)mP.DataPointer, (float*)mPe.DataPointer, inter, tx, durations, ty);
        VitsLengthRegulator.Expand((float*)logsP.DataPointer, (float*)logsPe.DataPointer, inter, tx, durations, ty);
        mP.Dispose(); logsP.Dispose();

        // z_p = m_p + randn * exp(logs_p) * noise_scale.
        uint rng = DeterministicRng.Seed(seed);
        Tensor zP = new(new TensorShape(1, inter, ty), DType.F32);
        float* zp = (float*)zP.DataPointer;
        float* mp = (float*)mPe.DataPointer;
        float* lp = (float*)logsPe.DataPointer;
        for (long n = 0; n < (long)inter * ty; n++)
            zp[n] = mp[n] + DeterministicRng.NextGaussian(ref rng) * MathF.Exp(lp[n]) * ns;
        mPe.Dispose(); logsPe.Dispose();

        Tensor z = _flow.Reverse(backend, zP, ty);
        zP.Dispose();
        float[] audio = _dec.Forward(backend, z, ty);
        z.Dispose();
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
        if (_disposed != 0) throw new ObjectDisposedException(nameof(VitsSynthesizer));
    }
}

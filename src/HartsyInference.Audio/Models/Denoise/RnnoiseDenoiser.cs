using HartsyInference.Audio.Preprocessing;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;

namespace HartsyInference.Audio.Models.Denoise;

/// <summary>Streaming RNNoise: 480-sample frames of 48 kHz mono in, denoised frames out.
///
/// <para>Per frame: high-pass the input, take a 50%-overlap windowed spectrum, estimate pitch and build the
/// pitch-shifted spectrum, reduce both to 32 triangular bands, hand the network 65 features, and apply the 32
/// gains it returns. <see cref="RnnoiseBands"/>, <see cref="RnnoisePitchAnalyzer"/> and
/// <see cref="RnnoiseModel"/> hold the three halves; this ties them together and owns the frame-to-frame state
/// they share.</para>
///
/// <para><b>The gains lag the spectrum by one frame, on purpose.</b> Features from frame N are applied to frame
/// N-1, so the network has seen 10 ms past the audio it is cleaning. That lookahead is what lets it open the
/// gate on a consonant's onset rather than clipping it, at the cost of 10 ms of latency on top of the window.</para>
///
/// <para><b>Silence is passed through untouched.</b> Below a total-band-energy floor the network is not run at
/// all and its recurrent state is left frozen, matching upstream: feeding near-zero features to three stacked
/// GRUs walks their state somewhere unhelpful, and the first real speech afterwards is then scored from a
/// corrupted context.</para>
///
/// <para><b>Audio is int16-scaled (±32768), not ±1.</b> Upstream's demo feeds raw <c>short</c> values straight
/// through as floats, and the thresholds inherited from it are absolute — the silence floor, the <c>1e-2</c>
/// inside the log, the <c>0.001</c> in the correlation normalizer. At ±1 every frame reads as silence, the
/// network never runs, and this degrades into a passthrough that looks like it is working. This matches the
/// scale the wake pipeline already carries, so no conversion is needed between them.</para>
///
/// <para>48 kHz is not negotiable — <see cref="RnnoiseBands"/>' band edges are bin indices that only map to the
/// trained frequencies at this rate. Resample around this class rather than retuning it.</para>
///
/// <para>Holds per-stream state throughout; one instance per stream, not thread-safe. The
/// <see cref="IBackend"/> is supplied per call so the caller (and the engine's device selection) decides where
/// the network runs.</para></summary>
public sealed class RnnoiseDenoiser : IDisposable
{
    /// <summary>Samples consumed and produced per call.</summary>
    public const int FrameSize = 480;

    /// <summary>Analysis window; 50% overlap at <see cref="FrameSize"/>.</summary>
    public const int WindowSize = 2 * FrameSize;

    /// <summary>The only rate the band edges are valid at.</summary>
    public const int SampleRate = 48_000;

    private const int Bands = RnnoiseBands.BandCount;
    private const int Bins = RnnoiseBands.FreqSize;

    /// <summary>Total band energy below which the frame is treated as silence.</summary>
    private const float SilenceEnergy = 0.04f;

    /// <summary>Upstream's FFT (CELT's kiss_fft) scales the <b>forward</b> transform by 1/N — which is why its
    /// inverse multiplies by N again. <see cref="Fft.RealTransform"/> returns an unscaled DFT, so spectra are
    /// brought onto upstream's scale here. This is not cosmetic: band energies go as |X|², so leaving it out
    /// inflates them by N² (921,600 at this window) and every absolute threshold downstream — the silence floor,
    /// the 1e-2 inside the log, the 0.001 in the correlation normalizer — lands in the wrong place.</summary>
    private const float ForwardFftScale = 1f / WindowSize;

    /// <summary>Per-frame floor on gain decay — an RT60 of about 135 ms. Without it the gate slams shut between
    /// syllables and the result sounds chopped rather than clean.</summary>
    private const float GainDecay = 0.6f;

    private static readonly float[] HighPassB = [-2f, 1f];
    private static readonly float[] HighPassA = [-1.99599f, 0.99600f];

    private readonly float[] _window;
    private readonly StreamingStft _stft;
    private readonly StreamingIstft _istft;
    private readonly RnnoisePitchAnalyzer _pitch = new();
    private readonly RnnoiseModel _model;

    private readonly float[] _highPassed = new float[FrameSize];
    private readonly float[] _hpMem = new float[2];
    private readonly float[] _xRe = new float[Bins];
    private readonly float[] _xIm = new float[Bins];
    private readonly float[] _pRe = new float[Bins];
    private readonly float[] _pIm = new float[Bins];
    private readonly float[] _delayedRe = new float[Bins];
    private readonly float[] _delayedIm = new float[Bins];
    private readonly float[] _delayedPRe = new float[Bins];
    private readonly float[] _delayedPIm = new float[Bins];
    private readonly float[] _pitchWindow = new float[WindowSize];
    private readonly float[] _ex = new float[Bands];
    private readonly float[] _ep = new float[Bands];
    private readonly float[] _exp = new float[Bands];
    private readonly float[] _ly = new float[Bands];
    private readonly float[] _delayedEx = new float[Bands];
    private readonly float[] _delayedEp = new float[Bands];
    private readonly float[] _delayedExp = new float[Bands];
    private readonly float[] _features = new float[RnnoiseBands.FeatureCount];
    private readonly float[] _gains = new float[Bands];
    private readonly float[] _lastGains = new float[Bands];
    private readonly float[] _scratchBands = new float[Bands];
    private readonly float[] _binGain = new float[Bins];
    private int _disposed;

    /// <summary>The VAD head's output for the most recent non-silent frame. A by-product of denoising, and a
    /// far better speech/noise signal than the RMS gate it could replace upstream of wake scoring.</summary>
    public float SpeechProbability { get; private set; }

    /// <summary>Builds a stream over shared <paramref name="weights"/>, which are borrowed, not owned.</summary>
    public RnnoiseDenoiser(RnnoiseWeights weights)
    {
        _model = new RnnoiseModel(weights);
        _window = RnnoiseBands.BuildWindow(FrameSize);
        _stft = new StreamingStft(WindowSize, FrameSize, WindowSize * 2, _window);
        _istft = new StreamingIstft(WindowSize, FrameSize, _window);
        PrimeAnalysis();
    }

    /// <summary>Denoises one frame. <paramref name="input"/> and <paramref name="output"/> are both
    /// <see cref="FrameSize"/> samples of 48 kHz mono; they may not overlap.</summary>
    public void Process(IBackend backend, ReadOnlySpan<float> input, Span<float> output)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (input.Length != FrameSize)
            throw new ArgumentException($"input must be {FrameSize} samples, got {input.Length}.", nameof(input));
        if (output.Length < FrameSize)
            throw new ArgumentException($"output must hold {FrameSize} samples.", nameof(output));

        HighPass(input, _highPassed);
        _stft.AddSamples(_highPassed);
        if (!_stft.TryExtractFrame(_xRe, _xIm))
            throw new InvalidOperationException("StreamingStft did not yield a frame; analysis priming is wrong.");
        Scale(_xRe, _xIm, ForwardFftScale);

        RnnoiseBands.ComputeBandEnergy(_xRe, _xIm, _ex);
        bool silence = ComputeFeatures();

        if (!silence)
        {
            _model.Process(backend, _features, _gains, out float vad);
            SpeechProbability = vad;
            PitchFilter();
            for (int i = 0; i < Bands; i++)
            {
                _gains[i] = MathF.Max(_gains[i], GainDecay * _lastGains[i]);
                // Rescale by the energy change across the frame, so a rising transient does not carry the
                // previous frame's permissive gain and leak noise with it.
                _lastGains[i] = MathF.Min(1f, _gains[i] * (_delayedEx[i] + 1e-3f) / (_ex[i] + 1e-3f));
            }
            RnnoiseBands.InterpolateBandGain(_gains, _binGain);
            for (int k = 0; k < Bins; k++)
            {
                _delayedRe[k] *= _binGain[k];
                _delayedIm[k] *= _binGain[k];
            }
        }

        // Back to an unscaled DFT, which is what StreamingIstft's inverse expects. Safe in place: the delayed
        // spectrum is overwritten from the current frame immediately below.
        Scale(_delayedRe, _delayedIm, WindowSize);
        _istft.PushFrame(_delayedRe, _delayedIm, output);

        _xRe.CopyTo(_delayedRe, 0);
        _xIm.CopyTo(_delayedIm, 0);
        _pRe.CopyTo(_delayedPRe, 0);
        _pIm.CopyTo(_delayedPIm, 0);
        _ex.CopyTo(_delayedEx, 0);
        _ep.CopyTo(_delayedEp, 0);
        _exp.CopyTo(_delayedExp, 0);
    }

    /// <summary>Builds the 65-value feature vector for the current frame. Returns true when the frame is silent,
    /// in which case the features are zeroed and the caller must skip the network.</summary>
    private bool ComputeFeatures()
    {
        _pitch.Push(_highPassed);
        int period = _pitch.Analyze(out _);
        period = Math.Clamp(period, RnnoisePitchAnalyzer.MinPeriod, RnnoisePitchAnalyzer.MaxPeriod);

        ReadOnlySpan<float> history = _pitch.History;
        int start = RnnoisePitchAnalyzer.BufferSize - WindowSize - period;
        for (int i = 0; i < WindowSize; i++) _pitchWindow[i] = history[start + i] * _window[i];
        Fft.RealTransform(_pitchWindow, _pRe, _pIm, WindowSize);
        Scale(_pRe, _pIm, ForwardFftScale);

        RnnoiseBands.ComputeBandEnergy(_pRe, _pIm, _ep);
        RnnoiseBands.ComputeBandCorrelation(_xRe, _xIm, _pRe, _pIm, _exp);
        for (int i = 0; i < Bands; i++)
            _exp[i] /= MathF.Sqrt(0.001f + _ex[i] * _ep[i]);
        RnnoiseBands.Dct(_exp, _features.AsSpan(Bands));
        _features[2 * Bands] = 0.01f * (period - 300);

        // Log band energies, floored twice: against the loudest band (-70 dB) and against a per-band decay, so a
        // single quiet band cannot dominate the cepstrum.
        float logMax = -2f;
        float follow = -2f;
        float energy = 0f;
        for (int i = 0; i < Bands; i++)
        {
            float ly = MathF.Log10(1e-2f + _ex[i]);
            ly = MathF.Max(logMax - 7f, MathF.Max(follow - 1.5f, ly));
            logMax = MathF.Max(logMax, ly);
            follow = MathF.Max(follow - 1.5f, ly);
            _ly[i] = ly;
            energy += _ex[i];
        }

        if (energy < SilenceEnergy)
        {
            Array.Clear(_features);
            return true;
        }

        RnnoiseBands.Dct(_ly, _features);
        _features[0] -= 12f;
        _features[1] -= 4f;
        return false;
    }

    /// <summary>Comb-filters the delayed spectrum toward its pitch harmonics: where the pitch correlation beats
    /// the network's gain, some of the pitch-shifted spectrum is mixed back in, then each band is renormalized
    /// to the energy it had. This restores harmonic structure the band gains alone would have flattened.</summary>
    private void PitchFilter()
    {
        for (int i = 0; i < Bands; i++)
        {
            float corr = _delayedExp[i];
            float g = _gains[i];
            float r;
            if (corr > g) r = 1f;
            else
            {
                float c2 = corr * corr;
                float g2 = g * g;
                r = c2 * (1f - g2) / (0.001f + g2 * (1f - c2));
            }
            r = MathF.Sqrt(Math.Clamp(r, 0f, 1f));
            _scratchBands[i] = r * MathF.Sqrt(_delayedEx[i] / (1e-8f + _delayedEp[i]));
        }
        RnnoiseBands.InterpolateBandGain(_scratchBands, _binGain);
        // The delayed pitch spectrum, not the current one: everything this filter touches belongs to frame N-1,
        // and mixing in frame N's harmonics would comb the wrong spectrum.
        for (int k = 0; k < Bins; k++)
        {
            _delayedRe[k] += _binGain[k] * _delayedPRe[k];
            _delayedIm[k] += _binGain[k] * _delayedPIm[k];
        }

        RnnoiseBands.ComputeBandEnergy(_delayedRe, _delayedIm, _scratchBands);
        for (int i = 0; i < Bands; i++)
            _scratchBands[i] = MathF.Sqrt(_delayedEx[i] / (1e-8f + _scratchBands[i]));
        RnnoiseBands.InterpolateBandGain(_scratchBands, _binGain);
        for (int k = 0; k < Bins; k++)
        {
            _delayedRe[k] *= _binGain[k];
            _delayedIm[k] *= _binGain[k];
        }
    }

    private static void Scale(Span<float> re, Span<float> im, float scale)
    {
        for (int k = 0; k < re.Length; k++)
        {
            re[k] *= scale;
            im[k] *= scale;
        }
    }

    /// <summary>Direct-form biquad high-pass, removing DC and rumble the band energies would otherwise carry.</summary>
    private void HighPass(ReadOnlySpan<float> input, Span<float> output)
    {
        float m0 = _hpMem[0];
        float m1 = _hpMem[1];
        for (int i = 0; i < input.Length; i++)
        {
            float x = input[i];
            float y = x + m0;
            m0 = m1 + (HighPassB[0] * x - HighPassA[0] * y);
            m1 = HighPassB[1] * x - HighPassA[1] * y;
            output[i] = y;
        }
        _hpMem[0] = m0;
        _hpMem[1] = m1;
    }

    /// <summary>Feeds one silent frame so the very first real frame yields a spectrum immediately. Upstream gets
    /// this from a zeroed <c>analysis_mem</c>; the streaming analyzer needs a full window before it emits, so the
    /// same zeros are pushed explicitly to keep the two in lockstep.</summary>
    private void PrimeAnalysis()
    {
        Span<float> silence = stackalloc float[FrameSize];
        silence.Clear();
        _stft.AddSamples(silence);
    }

    /// <summary>Clears every piece of stream state. Call on a discontinuity — the GRUs, the overlap-add tail, the
    /// pitch history and the one-frame delay all assume the audio was contiguous.</summary>
    public void Reset()
    {
        _stft.Reset();
        _istft.Reset();
        _pitch.Reset();
        _model.Reset();
        Array.Clear(_hpMem);
        Array.Clear(_delayedRe);
        Array.Clear(_delayedIm);
        Array.Clear(_delayedPRe);
        Array.Clear(_delayedPIm);
        Array.Clear(_delayedEx);
        Array.Clear(_delayedEp);
        Array.Clear(_delayedExp);
        Array.Clear(_lastGains);
        SpeechProbability = 0f;
        PrimeAnalysis();
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _model.Dispose();
    }
}

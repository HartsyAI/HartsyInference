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
    /// inverse multiplies by N again. <see cref="FftPlan"/> returns an unscaled DFT, so spectra are
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
    private readonly FftPlan _fft = new(WindowSize);
    private readonly RnnoiseModel _model;

    private readonly float[] _highPassed = new float[FrameSize];
    private readonly float[] _hpMem = new float[2];
    private readonly AnalyzedFrame _first = new();
    private readonly AnalyzedFrame _second = new();
    private readonly float[] _delayedRe = new float[Bins];
    private readonly float[] _delayedIm = new float[Bins];
    private readonly float[] _delayedPRe = new float[Bins];
    private readonly float[] _delayedPIm = new float[Bins];
    private readonly float[] _pitchWindow = new float[WindowSize];
    private readonly float[] _ly = new float[Bands];
    private readonly float[] _delayedEx = new float[Bands];
    private readonly float[] _delayedEp = new float[Bands];
    private readonly float[] _delayedExp = new float[Bands];
    private readonly float[] _lastGains = new float[Bands];
    private readonly float[] _scratchBands = new float[Bands];
    private readonly float[] _binGain = new float[Bins];
    private int _disposed;

    /// <summary>The VAD head's output for the most recent non-silent frame. A by-product of denoising, and a
    /// far better speech/noise signal than the RMS gate it could replace upstream of wake scoring.</summary>
    public float SpeechProbability { get; private set; }

    /// <summary>Pairs in which only the first frame was silent, so the second ran alone; counted so a test can show
    /// it covered that branch.</summary>
    internal int PairsWithFirstSilent { get; private set; }

    /// <summary>Pairs in which only the second frame was silent, so the first ran alone.</summary>
    internal int PairsWithSecondSilent { get; private set; }

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
        CheckFrame(input, output);

        Analyze(input, _first);
        if (!_first.Silent)
        {
            _model.Process(backend, _first.Features, _first.Gains, out float vad);
            SpeechProbability = vad;
        }
        Synthesize(_first, output);
    }

    /// <summary>Denoises two consecutive frames, <paramref name="first"/> then <paramref name="second"/>, with the
    /// same output and the same state afterwards as two <see cref="Process"/> calls, bit for bit.</summary>
    /// <remarks>Both frames are analyzed before either is synthesized. Analysis depends only on the input and
    /// synthesis only on the gains and the delayed spectrum, so the reordering changes nothing. It lets the network
    /// run the pair layer by layer through <see cref="RnnoiseModel.ProcessPair"/>, which reads the weights shared by
    /// the two frames once instead of twice. When either frame is silent, each non-silent one runs alone, as it would
    /// have.</remarks>
    public void ProcessPair(IBackend backend, ReadOnlySpan<float> first, ReadOnlySpan<float> second,
        Span<float> firstOutput, Span<float> secondOutput)
    {
        ArgumentNullException.ThrowIfNull(backend);
        CheckFrame(first, firstOutput);
        CheckFrame(second, secondOutput);

        Analyze(first, _first);
        Analyze(second, _second);
        if (!_first.Silent && !_second.Silent)
        {
            _model.ProcessPair(backend, _first.Features, _second.Features, _first.Gains, _second.Gains,
                out _, out float vad);
            SpeechProbability = vad;
        }
        else
        {
            if (!_first.Silent)
            {
                _model.Process(backend, _first.Features, _first.Gains, out float vad);
                SpeechProbability = vad;
                PairsWithSecondSilent++;
            }
            if (!_second.Silent)
            {
                _model.Process(backend, _second.Features, _second.Gains, out float vad);
                SpeechProbability = vad;
                PairsWithFirstSilent++;
            }
        }
        Synthesize(_first, firstOutput);
        Synthesize(_second, secondOutput);
    }

    private static void CheckFrame(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != FrameSize)
            throw new ArgumentException($"input must be {FrameSize} samples, got {input.Length}.", nameof(input));
        if (output.Length < FrameSize)
            throw new ArgumentException($"output must hold {FrameSize} samples.", nameof(output));
    }

    /// <summary>The input half of a frame: high-pass, spectrum, pitch spectrum, band energies and features. Reads and
    /// advances only input-side state, so the next frame can be analyzed before this one is synthesized.</summary>
    private void Analyze(ReadOnlySpan<float> input, AnalyzedFrame frame)
    {
        HighPass(input, _highPassed);
        _stft.AddSamples(_highPassed);
        if (!_stft.TryExtractFrame(frame.XRe, frame.XIm))
            throw new InvalidOperationException("StreamingStft did not yield a frame; analysis priming is wrong.");
        Scale(frame.XRe, frame.XIm, ForwardFftScale);

        RnnoiseBands.ComputeBandEnergy(frame.XRe, frame.XIm, frame.Ex);
        frame.Silent = ComputeFeatures(frame);
    }

    /// <summary>The output half: apply the frame's gains, which the network has already written, to the delayed
    /// spectrum, synthesize it, and keep this frame's spectra as the next frame's delayed ones.</summary>
    private void Synthesize(AnalyzedFrame frame, Span<float> output)
    {
        if (!frame.Silent)
        {
            float[] gains = frame.Gains;
            PitchFilter(gains);
            for (int i = 0; i < Bands; i++)
            {
                gains[i] = MathF.Max(gains[i], GainDecay * _lastGains[i]);
                // Rescale by the energy change across the frame, so a rising transient does not carry the
                // previous frame's permissive gain and leak noise with it.
                _lastGains[i] = MathF.Min(1f, gains[i] * (_delayedEx[i] + 1e-3f) / (frame.Ex[i] + 1e-3f));
            }
            RnnoiseBands.InterpolateBandGain(gains, _binGain);
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

        frame.XRe.CopyTo(_delayedRe, 0);
        frame.XIm.CopyTo(_delayedIm, 0);
        frame.PRe.CopyTo(_delayedPRe, 0);
        frame.PIm.CopyTo(_delayedPIm, 0);
        frame.Ex.CopyTo(_delayedEx, 0);
        frame.Ep.CopyTo(_delayedEp, 0);
        frame.Exp.CopyTo(_delayedExp, 0);
    }

    /// <summary>Builds the 65-value feature vector for the frame being analyzed. Returns true when the frame is
    /// silent, in which case the features are zeroed and the caller must skip the network.</summary>
    private bool ComputeFeatures(AnalyzedFrame frame)
    {
        _pitch.Push(_highPassed);
        int period = _pitch.Analyze(out _);
        period = Math.Clamp(period, RnnoisePitchAnalyzer.MinPeriod, RnnoisePitchAnalyzer.MaxPeriod);

        float[] features = frame.Features;
        float[] ex = frame.Ex;
        float[] ep = frame.Ep;
        float[] exp = frame.Exp;
        ReadOnlySpan<float> history = _pitch.History;
        int start = RnnoisePitchAnalyzer.BufferSize - WindowSize - period;
        for (int i = 0; i < WindowSize; i++) _pitchWindow[i] = history[start + i] * _window[i];
        _fft.ForwardReal(_pitchWindow, frame.PRe, frame.PIm);
        Scale(frame.PRe, frame.PIm, ForwardFftScale);

        RnnoiseBands.ComputeBandEnergy(frame.PRe, frame.PIm, ep);
        RnnoiseBands.ComputeBandCorrelation(frame.XRe, frame.XIm, frame.PRe, frame.PIm, exp);
        for (int i = 0; i < Bands; i++)
            exp[i] /= MathF.Sqrt(0.001f + ex[i] * ep[i]);
        RnnoiseBands.Dct(exp, features.AsSpan(Bands));
        features[2 * Bands] = 0.01f * (period - 300);

        // Log band energies, floored twice: against the loudest band (-70 dB) and against a per-band decay, so a
        // single quiet band cannot dominate the cepstrum.
        float logMax = -2f;
        float follow = -2f;
        float energy = 0f;
        for (int i = 0; i < Bands; i++)
        {
            float ly = MathF.Log10(1e-2f + ex[i]);
            ly = MathF.Max(logMax - 7f, MathF.Max(follow - 1.5f, ly));
            logMax = MathF.Max(logMax, ly);
            follow = MathF.Max(follow - 1.5f, ly);
            _ly[i] = ly;
            energy += ex[i];
        }

        if (energy < SilenceEnergy)
        {
            Array.Clear(features);
            return true;
        }

        RnnoiseBands.Dct(_ly, features);
        features[0] -= 12f;
        features[1] -= 4f;
        return false;
    }

    /// <summary>Comb-filters the delayed spectrum toward its pitch harmonics: where the pitch correlation beats
    /// the network's gain, some of the pitch-shifted spectrum is mixed back in, then each band is renormalized
    /// to the energy it had. This restores harmonic structure the band gains alone would have flattened.</summary>
    private void PitchFilter(float[] gains)
    {
        for (int i = 0; i < Bands; i++)
        {
            float corr = _delayedExp[i];
            float g = gains[i];
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

    /// <summary>One frame's analysis and the gains the network writes for it. Two of these let a pair of frames
    /// both be analyzed before either is synthesized.</summary>
    private sealed class AnalyzedFrame
    {
        public readonly float[] XRe = new float[Bins];
        public readonly float[] XIm = new float[Bins];
        public readonly float[] PRe = new float[Bins];
        public readonly float[] PIm = new float[Bins];
        public readonly float[] Ex = new float[Bands];
        public readonly float[] Ep = new float[Bands];
        public readonly float[] Exp = new float[Bands];
        public readonly float[] Features = new float[RnnoiseBands.FeatureCount];
        public readonly float[] Gains = new float[Bands];
        public bool Silent;
    }
}

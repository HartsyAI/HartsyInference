namespace HartsyInference.Audio.Preprocessing;

/// <summary>Kaldi-compatible log-Mel filter-bank features, matching
/// <c>torchaudio.compliance.kaldi.fbank(..., dither=0)</c> with its defaults: 25 ms Povey window, 10 ms shift,
/// DC removal, 0.97 pre-emphasis, power spectrum, FFT size rounded up to a power of two, HTK-style Kaldi mel
/// scale (<c>1127·ln(1+f/700)</c>) over <c>[low_freq, nyquist]</c>, natural-log compression, and
/// <c>snip_edges=true</c> framing (no padding). This is the feature CAM++ / 3D-Speaker speaker encoders consume
/// (CosyVoice calls it with <c>num_mel_bins=80, dither=0, sample_frequency=16000</c>); it is NOT a Slaney mel,
/// so it cannot be produced by <see cref="MelSpectrogramExtractor"/>.
///
/// <para>Output is <c>[n_frames, n_mels]</c> (frame-major, matching torchaudio). Cepstral mean normalization
/// (CosyVoice subtracts the per-bin mean over time before the speaker encoder) is left to the caller.</para></summary>
public sealed class KaldiFbankExtractor
{
    // torchaudio.compliance.kaldi floors mel energies at torch.finfo(float).eps (machine epsilon, 2^-23),
    // NOT the denormal min — log(eps) = -15.9424 is the characteristic empty-bin floor of Kaldi fbank.
    private const float Epsilon = 1.1920929e-7f;
    private const float PreemphasisCoefficient = 0.97f;

    private readonly int _frameLength;   // samples per analysis window (e.g. 400 @ 16 kHz / 25 ms)
    private readonly int _frameShift;    // hop in samples (e.g. 160 @ 16 kHz / 10 ms)
    private readonly int _fftSize;       // next power of two >= frameLength (e.g. 512)
    private readonly int _numFftBins;    // fftSize/2 + 1 (rfft output length)
    private readonly int _numMel;
    private readonly float[] _window;    // Povey window over _frameLength
    private readonly float[,] _melBanks; // [numMel, numFftBins]; the Nyquist column is zero (Kaldi excludes it)

    // Per-call scratch (one extractor is single-threaded; allocate once).
    private readonly float[] _frame;
    private readonly float[] _re;
    private readonly float[] _im;

    public int NumMelBins => _numMel;

    public KaldiFbankExtractor(int sampleRate = 16_000, int numMelBins = 80,
        double frameLengthMs = 25.0, double frameShiftMs = 10.0, double lowFreq = 20.0, double highFreq = 0.0)
    {
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (numMelBins <= 0) throw new ArgumentOutOfRangeException(nameof(numMelBins));

        _frameLength = (int)(sampleRate * frameLengthMs / 1000.0);
        _frameShift = (int)(sampleRate * frameShiftMs / 1000.0);
        _fftSize = Fft.NextPow2(_frameLength);
        _numFftBins = _fftSize / 2 + 1;
        _numMel = numMelBins;
        _window = PoveyWindow(_frameLength);

        // high_freq <= 0 is interpreted as (nyquist + high_freq), matching Kaldi (0 → nyquist).
        double nyquist = sampleRate / 2.0;
        double high = highFreq <= 0.0 ? nyquist + highFreq : highFreq;
        _melBanks = KaldiMelBanks(_numMel, _fftSize, sampleRate, lowFreq, high);

        _frame = new float[_fftSize];
        _re = new float[_numFftBins];
        _im = new float[_numFftBins];
    }

    /// <summary>Number of fbank frames for an audio buffer (snip_edges: trailing partial frame is dropped).</summary>
    public int OutputFrames(int audioLength)
    {
        if (audioLength < _frameLength) return 0;
        return 1 + (audioLength - _frameLength) / _frameShift;
    }

    /// <summary>Computes the <c>[n_frames, n_mels]</c> log-fbank of <paramref name="audio"/>.</summary>
    public float[,] Compute(ReadOnlySpan<float> audio)
    {
        int frames = OutputFrames(audio.Length);
        float[,] output = new float[Math.Max(frames, 0), _numMel];
        for (int t = 0; t < frames; t++)
        {
            int start = t * _frameShift;

            // 1) Copy the raw frame.
            for (int i = 0; i < _frameLength; i++) _frame[i] = audio[start + i];

            // 2) Remove DC offset (subtract the frame mean).
            float mean = 0f;
            for (int i = 0; i < _frameLength; i++) mean += _frame[i];
            mean /= _frameLength;
            for (int i = 0; i < _frameLength; i++) _frame[i] -= mean;

            // 3) Pre-emphasis with replicate padding (x[-1] := x[0]), in place from the right.
            for (int i = _frameLength - 1; i >= 1; i--) _frame[i] -= PreemphasisCoefficient * _frame[i - 1];
            _frame[0] -= PreemphasisCoefficient * _frame[0];

            // 4) Apply the Povey window, then zero-pad to the FFT size.
            for (int i = 0; i < _frameLength; i++) _frame[i] *= _window[i];
            for (int i = _frameLength; i < _fftSize; i++) _frame[i] = 0f;

            // 5) Power spectrum.
            Fft.RealTransform(_frame, _re, _im, _fftSize);

            // 6) Mel integration + 7) natural-log compression.
            for (int m = 0; m < _numMel; m++)
            {
                float acc = 0f;
                for (int k = 0; k < _numFftBins; k++)
                {
                    float w = _melBanks[m, k];
                    if (w != 0f) acc += w * (_re[k] * _re[k] + _im[k] * _im[k]);
                }
                output[t, m] = MathF.Log(MathF.Max(acc, Epsilon));
            }
        }
        return output;
    }

    private static float[] PoveyWindow(int n)
    {
        // Kaldi Povey window: (0.5 - 0.5·cos(2π·i/(n-1)))^0.85.
        float[] w = new float[n];
        double a = 2.0 * Math.PI / (n - 1);
        for (int i = 0; i < n; i++) w[i] = (float)Math.Pow(0.5 - 0.5 * Math.Cos(a * i), 0.85);
        return w;
    }

    private static float[,] KaldiMelBanks(int numMel, int fftSize, int sampleRate, double lowFreq, double highFreq)
    {
        int numFftBins = fftSize / 2 + 1;
        double fftBinWidth = (double)sampleRate / fftSize;
        double melLow = MelScale(lowFreq);
        double melHigh = MelScale(highFreq);
        double delta = (melHigh - melLow) / (numMel + 1);

        float[,] banks = new float[numMel, numFftBins];
        for (int m = 0; m < numMel; m++)
        {
            double leftMel = melLow + m * delta;
            double centerMel = melLow + (m + 1) * delta;
            double rightMel = melLow + (m + 2) * delta;
            // The Nyquist bin (k = fftSize/2) is left at zero, mirroring torchaudio's padded mel banks.
            for (int k = 0; k < fftSize / 2; k++)
            {
                double mel = MelScale(k * fftBinWidth);
                if (mel <= leftMel || mel >= rightMel) continue;
                double weight = mel <= centerMel
                    ? (mel - leftMel) / (centerMel - leftMel)
                    : (rightMel - mel) / (rightMel - centerMel);
                banks[m, k] = (float)weight;
            }
        }
        return banks;
    }

    private static double MelScale(double freqHz) => 1127.0 * Math.Log(1.0 + freqHz / 700.0);
}

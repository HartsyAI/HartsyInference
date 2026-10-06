using HartsyInference.Audio.Preprocessing;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Log-mel front end of <c>lib.mel_converter.MelConverter</c> (<c>get_mel_converter('44k')</c>): the waveform is
/// clamped to [-1, 1], reflect-padded by <c>(n_fft - hop) / 2</c>, framed without centering, and the magnitude
/// <c>sqrt(re^2 + im^2 + 1e-9)</c> goes through a Slaney mel bank and a floored (1e-5) logarithm. It is the mel the
/// vocoder was trained on, so wav to mel to vocode round-trips.</summary>
public sealed class ControlFoleyMelConverter
{
    private const float MagnitudeEpsilon = 1e-9f;
    private const float LogFloor = 1e-5f;

    private readonly int _nFft;
    private readonly int _hop;
    private readonly int _numMels;
    private readonly bool _log10;
    private readonly float[] _window;
    private readonly float[,] _melBasis;

    public ControlFoleyMelConverter(int sampleRate, int nFft, int numMels, int hopSize, double fmin, double fmax, bool log10)
    {
        if (nFft < 2 || (nFft & (nFft - 1)) != 0) throw new ArgumentOutOfRangeException(nameof(nFft), "n_fft must be a power of two.");
        if (hopSize < 1 || hopSize > nFft) throw new ArgumentOutOfRangeException(nameof(hopSize));
        SampleRate = sampleRate;
        _nFft = nFft;
        _hop = hopSize;
        _numMels = numMels;
        _log10 = log10;
        _window = HannWindow.Get(nFft);
        _melBasis = MelFilterbank.Get(sampleRate, nFft, numMels, fmin, fmax);
    }

    /// <summary>The 44.1 kHz converter: 2048-point FFT, hop 512, 128 mel bins to Nyquist, natural log.</summary>
    public static ControlFoleyMelConverter Create44k() => new(44_100, 2_048, 128, 512, 0.0, 22_050.0, log10: false);

    /// <summary>The 16 kHz converter: 1024-point FFT, hop 256, 80 mel bins to 8 kHz, log10.</summary>
    public static ControlFoleyMelConverter Create16k() => new(16_000, 1_024, 80, 256, 0.0, 8_000.0, log10: true);

    public int SampleRate { get; }

    public int NumMels => _numMels;

    /// <summary>The Slaney mel filter bank <c>[numMels, nFft / 2 + 1]</c>.</summary>
    public float[,] MelBasis => _melBasis;

    /// <summary>Mel frames produced for <paramref name="sampleCount"/> samples.</summary>
    public int FrameCount(int sampleCount) => sampleCount / _hop;

    /// <summary>Returns the mel <c>[numMels, frames]</c> row-major.</summary>
    public float[] Compute(ReadOnlySpan<float> waveform)
    {
        int pad = (_nFft - _hop) / 2;
        if (waveform.Length <= pad) throw new ArgumentException($"Waveform needs more than {pad} samples.", nameof(waveform));
        float[] clamped = new float[waveform.Length];
        for (int i = 0; i < clamped.Length; i++) clamped[i] = Math.Clamp(waveform[i], -1f, 1f);
        float[] padded = SignalPadding.Reflect(clamped, pad);
        int frames = (padded.Length - _nFft) / _hop + 1;
        int bins = _nFft / 2 + 1;
        float[] mel = new float[_numMels * frames];
        float[] frame = new float[_nFft];
        float[] re = new float[bins];
        float[] im = new float[bins];
        float[] magnitude = new float[bins];
        for (int f = 0; f < frames; f++)
        {
            int start = f * _hop;
            for (int i = 0; i < _nFft; i++) frame[i] = padded[start + i] * _window[i];
            Fft.RealTransform(frame, re, im, _nFft);
            for (int k = 0; k < bins; k++) magnitude[k] = MathF.Sqrt(re[k] * re[k] + im[k] * im[k] + MagnitudeEpsilon);
            for (int m = 0; m < _numMels; m++)
            {
                double acc = 0.0;
                for (int k = 0; k < bins; k++) acc += (double)_melBasis[m, k] * magnitude[k];
                float value = MathF.Max((float)acc, LogFloor);
                mel[m * frames + f] = _log10 ? MathF.Log10(value) : MathF.Log(value);
            }
        }
        return mel;
    }
}

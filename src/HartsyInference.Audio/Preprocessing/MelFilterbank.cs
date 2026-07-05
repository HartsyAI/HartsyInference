namespace HartsyInference.Audio.Preprocessing;

/// <summary>Mel scale convention. <see cref="Slaney"/> is linear below 1 kHz then logarithmic
/// (<c>librosa htk=False</c>); <see cref="Htk"/> is the O'Shaughnessy <c>2595·log10(1+f/700)</c> form
/// (<c>librosa htk=True</c> / torchaudio <c>mel_scale="htk"</c>).</summary>
public enum MelScale
{
    Slaney,
    Htk,
}

/// <summary>Mel filterbank generator. Defaults to the Slaney mel scale with Slaney area normalization
/// (<c>librosa.filters.mel(htk=False, norm='slaney')</c> / HF <c>mel_filter_bank(mel_scale='slaney',
/// norm='slaney')</c>, element-wise within 1e-6) but also supports the HTK scale and unnormalized
/// (<c>norm=None</c>) filters that torchaudio's <c>MelSpectrogram(mel_scale="htk", norm=None)</c> produces —
/// required by F5-TTS / Vocos, whose front-end uses HTK + no area norm.
///
/// <para>Used by Whisper (80 / 128 mel bins, n_fft=400, 0-8kHz), Kokoro (80 bins, n_fft=2048, 0-12kHz),
/// HiFiGAN (80 bins, n_fft=1024, 0-8kHz), and every other mel-input audio model in our scope.</para>
///
/// <para>Filterbanks are computed once at startup and cached on the static type keyed by the parameter
/// tuple. The cost is O(milliseconds) so generating at runtime (matching HuggingFace's behavior) is cheaper
/// than shipping pre-baked .npz files (matching the OpenAI Whisper convention).</para></summary>
public static class MelFilterbank
{
    private static readonly Dictionary<(int Sr, int NFft, int NMels, double Fmin, double Fmax, MelScale Scale, bool SlaneyNorm), float[,]> _cache = new();
    private static readonly object _cacheLock = new();

    /// <summary>Builds a mel filterbank of shape <c>[n_mels, n_fft/2 + 1]</c>. Both frequency arguments are
    /// zero-based; <paramref name="fmax"/>=null defaults to Nyquist (<c>sampleRate/2</c>). Defaults to the
    /// Slaney scale + Slaney area normalization; pass <paramref name="scale"/>=<see cref="MelScale.Htk"/> and
    /// <paramref name="slaneyNorm"/>=false for the torchaudio HTK/no-norm variant.</summary>
    public static float[,] Get(int sampleRate, int nFft, int nMels, double fmin = 0.0, double? fmax = null,
        MelScale scale = MelScale.Slaney, bool slaneyNorm = true)
    {
        double fmaxResolved = fmax ?? sampleRate / 2.0;
        (int, int, int, double, double, MelScale, bool) key = (sampleRate, nFft, nMels, fmin, fmaxResolved, scale, slaneyNorm);

        lock (_cacheLock)
        {
            if (_cache.TryGetValue(key, out float[,]? cached)) return cached;
            float[,] fb = Build(sampleRate, nFft, nMels, fmin, fmaxResolved, scale, slaneyNorm);
            _cache[key] = fb;
            return fb;
        }
    }

    private static float[,] Build(int sampleRate, int nFft, int nMels, double fmin, double fmax, MelScale scale, bool slaneyNorm)
    {
        int numBins = nFft / 2 + 1;

        // 1. Compute n_mels+2 mel-spaced center frequencies, then back to Hz.
        double melMin = HzToMel(fmin, scale);
        double melMax = HzToMel(fmax, scale);
        double[] melPoints = new double[nMels + 2];
        for (int i = 0; i < melPoints.Length; i++)
            melPoints[i] = melMin + (melMax - melMin) * i / (melPoints.Length - 1);
        double[] hzPoints = new double[melPoints.Length];
        for (int i = 0; i < melPoints.Length; i++) hzPoints[i] = MelToHz(melPoints[i], scale);

        // 2. FFT bin -> frequency mapping (linear, half-open up to Nyquist).
        double[] binFreqs = new double[numBins];
        for (int k = 0; k < numBins; k++) binFreqs[k] = (double)k * sampleRate / nFft;

        // 3. Triangular filters. Slaney area-normalization divides each filter by its bandwidth (constant
        //    energy per filter); with norm=None the triangles peak at 1.0 (torchaudio's default).
        float[,] fb = new float[nMels, numBins];
        for (int i = 0; i < nMels; i++)
        {
            double left = hzPoints[i];
            double center = hzPoints[i + 1];
            double right = hzPoints[i + 2];
            double enorm = slaneyNorm ? 2.0 / (hzPoints[i + 2] - hzPoints[i]) : 1.0;

            for (int k = 0; k < numBins; k++)
            {
                double f = binFreqs[k];
                double v;
                if (f <= left || f >= right) v = 0.0;
                else if (f <= center) v = (f - left) / (center - left);
                else v = (right - f) / (right - center);
                fb[i, k] = (float)(v * enorm);
            }
        }

        return fb;
    }

    /// <summary>Hz → mel on the requested scale.</summary>
    public static double HzToMel(double f, MelScale scale = MelScale.Slaney)
    {
        if (scale == MelScale.Htk) return 2595.0 * Math.Log10(1.0 + f / 700.0);
        double linMax = 15.0;      // mel value at 1000 Hz
        double logStep = Math.Log(6.4) / 27.0;
        if (f < 1000.0) return 3.0 * f / 200.0;
        return linMax + Math.Log(f / 1000.0) / logStep;
    }

    /// <summary>Inverse of <see cref="HzToMel"/> on the requested scale.</summary>
    public static double MelToHz(double m, MelScale scale = MelScale.Slaney)
    {
        if (scale == MelScale.Htk) return 700.0 * (Math.Pow(10.0, m / 2595.0) - 1.0);
        double linMax = 15.0;
        double logStep = Math.Log(6.4) / 27.0;
        if (m < linMax) return 200.0 * m / 3.0;
        return 1000.0 * Math.Exp((m - linMax) * logStep);
    }
}

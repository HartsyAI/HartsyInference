namespace HartsyInference.Audio.Preprocessing;

/// <summary>Composes <see cref="HannWindow"/>, <see cref="Fft"/>, and
/// <see cref="MelFilterbank"/> into a single audio → log-mel pipeline that matches
/// the model-specific preprocessing exactly.
///
/// <para>Three preset configurations are provided as static factories — these are the
/// parameter sets needed by every audio model in scope. Custom configurations are
/// constructed directly via the constructor.</para>
///
/// <para><b>Allocation contract:</b> the extractor pre-allocates all scratch buffers
/// at construction. <see cref="Compute"/> writes into a caller-provided output buffer
/// of shape <c>[n_mels, n_frames]</c> (row-major; channel-first layout matching every
/// model's expected input). No allocations happen on the hot path.</para>
///
/// <para>Validation target: each preset must match its Python reference within 1e-4
/// per element (the standard whisper.cpp tolerance) on a fixed sine-wave clip.</para></summary>
public sealed class MelSpectrogramExtractor
{
    /// <summary>Parameter set defining a specific mel preprocessing pipeline. The trailing
    /// <paramref name="Scale"/> / <paramref name="SlaneyNorm"/> / <paramref name="Center"/> parameters default
    /// to the librosa Slaney convention used by most presets; F5-TTS/Vocos overrides them to HTK + no-norm +
    /// center padding (see <see cref="F5VocosConfig"/>).</summary>
    public readonly record struct Config(
        int SampleRate,
        int NFft,
        int WinLength,
        int HopLength,
        int NMels,
        double Fmin,
        double Fmax,
        Normalization Norm,
        bool DropLastStftFrame,
        LogBase LogBase,
        float? LogFloor,
        float DynamicRangeDb,
        float NormOffset,
        float NormScale,
        bool PowerSpectrum,
        MelScale Scale = MelScale.Slaney,
        bool SlaneyNorm = true,
        bool Center = false);

    /// <summary>Whisper preset: 16kHz, n_fft=400, hop=160, 80 mel bins, log10,
    /// power spectrum, drop-last-frame, +4/4 normalization. Used by all Whisper
    /// sizes through large-v2 and turbo (80 bins). Pass <paramref name="nMels"/>=128
    /// for Whisper-large-v3+ which uses 128 bins.</summary>
    public static Config WhisperConfig(int nMels = 80) => new(
        SampleRate: 16_000,
        NFft: 400,
        WinLength: 400,
        HopLength: 160,
        NMels: nMels,
        Fmin: 0.0,
        Fmax: 8000.0,
        Norm: Normalization.WhisperDynamicRange,
        DropLastStftFrame: true,
        LogBase: LogBase.Log10,
        LogFloor: 1e-10f,
        DynamicRangeDb: 8.0f,
        NormOffset: 4.0f,
        NormScale: 4.0f,
        PowerSpectrum: true);

    /// <summary>StyleTTS 2 / Kokoro preset: 24kHz, n_fft=2048, hop=300, win=1200,
    /// 80 mel bins, magnitude spectrum, no normalization.</summary>
    public static Config Kokoro24kConfig() => new(
        SampleRate: 24_000,
        NFft: 2048,
        WinLength: 1200,
        HopLength: 300,
        NMels: 80,
        Fmin: 0.0,
        Fmax: 12_000.0,
        Norm: Normalization.None,
        DropLastStftFrame: false,
        LogBase: LogBase.Natural,
        LogFloor: null,
        DynamicRangeDb: 0f,
        NormOffset: 0f,
        NormScale: 1f,
        PowerSpectrum: false);

    /// <summary>CosyVoice 2 flow-matching mel conditioning (matcha <c>mel_spectrogram</c>): 24kHz, n_fft=1920,
    /// hop=480, win=1920, 80 mel bins, fmax=8000, magnitude spectrum, natural log clamped at 1e-5. The reference
    /// reflect-pads the audio by <c>(n_fft - hop)/2</c> (center=False); callers should pre-pad to match.</summary>
    public static Config CosyVoice2FlowConfig() => new(
        SampleRate: 24_000,
        NFft: 1_920,
        WinLength: 1_920,
        HopLength: 480,
        NMels: 80,
        Fmin: 0.0,
        Fmax: 8000.0,
        Norm: Normalization.None,
        DropLastStftFrame: false,
        LogBase: LogBase.Natural,
        LogFloor: 1e-5f,
        DynamicRangeDb: 0f,
        NormOffset: 0f,
        NormScale: 1f,
        PowerSpectrum: false);

    /// <summary>F5-TTS / Vocos mel front-end: torchaudio <c>MelSpectrogram(sr=24k, n_fft=1024, hop=256,
    /// win=1024, n_mels=100, power=1, center=True, norm=None, mel_scale="htk")</c> then <c>clamp(1e-5).log()</c>.
    /// HTK scale + no area norm + center reflect-padding + magnitude (not power) spectrum + natural log — every
    /// axis differs from the Slaney presets, which is why F5 output was distorted before this existed.</summary>
    public static Config F5VocosConfig() => new(
        SampleRate: 24_000,
        NFft: 1024,
        WinLength: 1024,
        HopLength: 256,
        NMels: 100,
        Fmin: 0.0,
        Fmax: 12_000.0,
        Norm: Normalization.None,
        DropLastStftFrame: false,
        LogBase: LogBase.Natural,
        LogFloor: 1e-5f,
        DynamicRangeDb: 0f,
        NormOffset: 0f,
        NormScale: 1f,
        PowerSpectrum: false,
        Scale: MelScale.Htk,
        SlaneyNorm: false,
        Center: true);

    /// <summary>Standard HiFiGAN preset: 22.05kHz, n_fft=1024, hop=256, 80 mel bins.
    /// Magnitude spectrum, no normalization.</summary>
    public static Config HifiGan22kConfig() => new(
        SampleRate: 22_050,
        NFft: 1024,
        WinLength: 1024,
        HopLength: 256,
        NMels: 80,
        Fmin: 0.0,
        Fmax: 8000.0,
        Norm: Normalization.None,
        DropLastStftFrame: false,
        LogBase: LogBase.Natural,
        LogFloor: null,
        DynamicRangeDb: 0f,
        NormOffset: 0f,
        NormScale: 1f,
        PowerSpectrum: false);

    public enum Normalization
    {
        None,
        /// <summary>Whisper-specific: clamp to (max - 8.0) in log10 units, then
        /// apply <c>(x + 4) / 4</c>. Yields output ~[0, 1].</summary>
        WhisperDynamicRange,
    }

    public enum LogBase { Log10, Natural }

    private readonly Config _cfg;
    private readonly float[] _window;
    private readonly float[,] _filterbank;
    private readonly int _numBins;
    private readonly int _fftSize;

    // Reusable scratch arrays (one set per extractor instance).
    private readonly float[] _frame;       // [_fftSize] zero-padded windowed frame
    private readonly float[] _stftRe;      // [_numBins]
    private readonly float[] _stftIm;      // [_numBins]
    private readonly float[] _power;       // [_numBins]
    private readonly float[] _melCol;      // [n_mels]

    public MelSpectrogramExtractor(Config cfg)
    {
        _cfg = cfg;
        _window = HannWindow.Get(cfg.WinLength);
        // FFT size must be a power of two; round up if win_length is not.
        _fftSize = NextPow2(cfg.NFft);
        _numBins = _fftSize / 2 + 1;
        _filterbank = MelFilterbank.Get(cfg.SampleRate, _fftSize, cfg.NMels, cfg.Fmin, cfg.Fmax, cfg.Scale, cfg.SlaneyNorm);

        _frame = new float[_fftSize];
        _stftRe = new float[_numBins];
        _stftIm = new float[_numBins];
        _power = new float[_numBins];
        _melCol = new float[cfg.NMels];
    }

    /// <summary>The number of mel frames a given input audio length will produce.</summary>
    public int OutputFrames(int audioLength)
    {
        // torch.stft(center=True) reflect-pads by n_fft/2 each side and yields (len // hop) + 1 frames.
        // Without centering the caller is responsible for any pre-padding (e.g. Whisper zero-pads to 30s).
        int frames = _cfg.Center
            ? 1 + audioLength / _cfg.HopLength
            : 1 + (audioLength - _cfg.WinLength) / _cfg.HopLength;
        if (frames < 1) frames = 1;
        if (_cfg.DropLastStftFrame) frames--;
        return frames;
    }

    /// <summary>Computes the log-mel spectrogram of the given audio buffer into
    /// the provided output array of shape <c>[n_mels, n_frames]</c>, row-major.</summary>
    public void Compute(ReadOnlySpan<float> audio, float[,] output)
    {
        int frames = OutputFrames(audio.Length);
        if (output.GetLength(0) != _cfg.NMels || output.GetLength(1) < frames)
            throw new ArgumentException($"output must be [{_cfg.NMels}, >={frames}]");

        // torch.stft(center=True): reflect-pad by n_fft/2 so frame t is centered at t*hop.
        float[]? padded = _cfg.Center ? ReflectPad(audio, _cfg.NFft / 2) : null;
        ReadOnlySpan<float> src = padded ?? audio;

        float globalMax = float.MinValue;

        for (int t = 0; t < frames; t++)
        {
            int start = t * _cfg.HopLength;
            // Windowed frame, zero-padded to FFT size.
            for (int i = 0; i < _cfg.WinLength; i++)
            {
                float sample = (start + i) < src.Length ? src[start + i] : 0f;
                _frame[i] = sample * _window[i];
            }
            for (int i = _cfg.WinLength; i < _fftSize; i++) _frame[i] = 0f;

            // Real FFT.
            Fft.RealTransform(_frame, _stftRe, _stftIm, _fftSize);

            // Magnitude or power.
            if (_cfg.PowerSpectrum)
            {
                for (int k = 0; k < _numBins; k++)
                    _power[k] = _stftRe[k] * _stftRe[k] + _stftIm[k] * _stftIm[k];
            }
            else
            {
                for (int k = 0; k < _numBins; k++)
                    _power[k] = MathF.Sqrt(_stftRe[k] * _stftRe[k] + _stftIm[k] * _stftIm[k]);
            }

            // Mel filterbank: [n_mels, n_bins] × [n_bins] -> [n_mels]
            // Plain row-major matmul; the filterbank is sparse (triangular) but at
            // [80, 201] sizes a dense matmul is cache-friendly enough that the
            // sparse-skip codepath isn't worth the branch.
            for (int m = 0; m < _cfg.NMels; m++)
            {
                float acc = 0f;
                for (int k = 0; k < _numBins; k++) acc += _filterbank[m, k] * _power[k];
                _melCol[m] = acc;
            }

            // Log compression.
            float floor = _cfg.LogFloor ?? 1e-10f;
            for (int m = 0; m < _cfg.NMels; m++)
            {
                float v = MathF.Max(_melCol[m], floor);
                float l = _cfg.LogBase == LogBase.Log10 ? MathF.Log10(v) : MathF.Log(v);
                output[m, t] = l;
                if (l > globalMax) globalMax = l;
            }
        }

        // Post-pass normalization (Whisper: dynamic-range clamp + (+4)/4 shift).
        if (_cfg.Norm == Normalization.WhisperDynamicRange)
        {
            float clampMin = globalMax - _cfg.DynamicRangeDb;
            float invScale = 1f / _cfg.NormScale;
            for (int m = 0; m < _cfg.NMels; m++)
                for (int t = 0; t < frames; t++)
                    output[m, t] = (MathF.Max(output[m, t], clampMin) + _cfg.NormOffset) * invScale;
        }
    }

    /// <summary>Computes a single mel frame from <c>WinLength</c> samples of audio. Used
    /// by <see cref="HartsyInference.Audio.Streaming.StreamingMelExtractor"/>. No global
    /// normalization is applied — streaming has no "global max" to clamp against; the
    /// caller layers any per-feature normalization on top.
    ///
    /// <para><paramref name="windowAudio"/> must be exactly <c>WinLength</c> samples (or
    /// shorter, in which case zeros pad to <c>FFT size</c>). <paramref name="melColumn"/>
    /// must be exactly <c>NMels</c> values.</para></summary>
    public void ComputeFrame(ReadOnlySpan<float> windowAudio, Span<float> melColumn)
    {
        if (melColumn.Length != _cfg.NMels)
            throw new ArgumentException($"melColumn must be length {_cfg.NMels}, got {melColumn.Length}.", nameof(melColumn));

        // Windowed frame, zero-padded to FFT size.
        int n = Math.Min(windowAudio.Length, _cfg.WinLength);
        for (int i = 0; i < n; i++) _frame[i] = windowAudio[i] * _window[i];
        for (int i = n; i < _fftSize; i++) _frame[i] = 0f;

        Fft.RealTransform(_frame, _stftRe, _stftIm, _fftSize);

        if (_cfg.PowerSpectrum)
        {
            for (int k = 0; k < _numBins; k++)
                _power[k] = _stftRe[k] * _stftRe[k] + _stftIm[k] * _stftIm[k];
        }
        else
        {
            for (int k = 0; k < _numBins; k++)
                _power[k] = MathF.Sqrt(_stftRe[k] * _stftRe[k] + _stftIm[k] * _stftIm[k]);
        }

        for (int m = 0; m < _cfg.NMels; m++)
        {
            float acc = 0f;
            for (int k = 0; k < _numBins; k++) acc += _filterbank[m, k] * _power[k];
            _melCol[m] = acc;
        }

        float floor = _cfg.LogFloor ?? 1e-10f;
        for (int m = 0; m < _cfg.NMels; m++)
        {
            float v = MathF.Max(_melCol[m], floor);
            float l = _cfg.LogBase == LogBase.Log10 ? MathF.Log10(v) : MathF.Log(v);
            melColumn[m] = l;
        }
    }

    /// <summary>Exposes the config for streaming clients that need WinLength / HopLength
    /// / NMels without owning the extractor's internal scratch.</summary>
    public Config Configuration => _cfg;

    /// <summary>Convenience overload that allocates the output array.</summary>
    public float[,] Compute(ReadOnlySpan<float> audio)
    {
        int frames = OutputFrames(audio.Length);
        float[,] result = new float[_cfg.NMels, frames];
        Compute(audio, result);
        return result;
    }

    private static int NextPow2(int n)
    {
        int p = 1;
        while (p < n) p <<= 1;
        return p;
    }

    /// <summary>Reflect-pads <paramref name="audio"/> by <paramref name="pad"/> samples each side using the
    /// edge-excluding ("reflect-101") convention that <c>torch.stft(center=True, pad_mode="reflect")</c> uses.</summary>
    private static float[] ReflectPad(ReadOnlySpan<float> audio, int pad)
    {
        int len = audio.Length;
        float[] outp = new float[len + 2 * pad];
        for (int i = 0; i < outp.Length; i++)
            outp[i] = audio[ReflectIndex(i - pad, len)];
        return outp;
    }

    private static int ReflectIndex(int j, int len)
    {
        if (len <= 1) return 0;
        int period = 2 * (len - 1);
        int m = ((j % period) + period) % period;
        return m < len ? m : period - m;
    }
}

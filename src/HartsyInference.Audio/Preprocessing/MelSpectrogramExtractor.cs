using HartsyInference.Core.Numerics;

namespace HartsyInference.Audio.Preprocessing;

/// <summary>Composes <see cref="HannWindow"/>, <see cref="Fft"/>, and
/// <see cref="MelFilterbank"/> into a single audio → log-mel pipeline that matches
/// the model-specific preprocessing exactly.
///
/// <para>Preset configurations are provided as static factories — one per model family in
/// scope. Custom configurations are constructed directly via the constructor.</para>
///
/// <para><b>Allocation contract:</b> the window, filterbank and FFT plan are built at construction.
/// <see cref="Compute(ReadOnlySpan{float}, float[,])"/> and <see cref="ComputeZeroPadded"/> write into a
/// caller-provided output of shape <c>[n_mels, n_frames]</c> (row-major; channel-first layout matching every model's
/// expected input) and fan frame blocks out through <see cref="CpuParallel"/>, each block renting its scratch from
/// <see cref="ArrayPool{T}.Shared"/>, so a warm extractor allocates nothing of its own per call. The blocks are fixed
/// by the FFT size alone and every frame is computed the same way inline or fanned out, so the output never depends on
/// the core count.</para>
///
/// <para>Validation target: each preset must match its Python reference within 1e-4
/// per element (the standard whisper.cpp tolerance).</para></summary>
public sealed class MelSpectrogramExtractor
{
    private static readonly Action<int, FrameBlocks> FrameBlockBody = RunFrameBlock;

    /// <summary>Parameter set defining a specific mel preprocessing pipeline. The trailing
    /// <paramref name="Scale"/> / <paramref name="SlaneyNorm"/> / <paramref name="Center"/> parameters default
    /// to the librosa Slaney convention used by most presets; F5-TTS/Vocos overrides them to HTK + no-norm +
    /// center padding (see <see cref="F5VocosConfig"/>).</summary>
    /// <param name="CenterWindowInFft">When WinLength &lt; NFft, torch.stft/torchaudio pad the analysis window to
    /// NFft *centered* ((NFft-WinLength)/2 zeros each side) rather than left-aligning it. Most presets here have
    /// WinLength == NFft so it is moot; StyleTTS2's reference mel (win 1200 &lt; n_fft 2048) needs it.</param>
    /// <param name="AdditiveLogFloor">Zonos's speaker front-end applies log(mel + floor) (additive) rather than
    /// log(max(mel, floor)) (clamp). The two agree for mel ≫ floor but diverge near-silent bins; Zonos needs the
    /// additive form.</param>
    /// <param name="ExactFftSize">Transforms at exactly NFft points with the filterbank on its NFft/2 + 1 bins, as
    /// torch.stft and numpy's rfft do; a size that is not a power of two runs through an <see cref="FftPlan"/>, so its
    /// prime factors must be 2, 3 and 5. Off, NFft is rounded up to a power of two and the frame zero-padded to it,
    /// the layout every preset but Whisper's was validated with.</param>
    public readonly record struct Config(int SampleRate, int NFft, int WinLength, int HopLength, int NMels, double Fmin,
        double Fmax, Normalization Norm, bool DropLastStftFrame, LogBase LogBase, float? LogFloor, float DynamicRangeDb,
        float NormOffset, float NormScale, bool PowerSpectrum, MelScale Scale = MelScale.Slaney, bool SlaneyNorm = true,
        bool Center = false, bool CenterWindowInFft = false, bool AdditiveLogFloor = false, bool ExactFftSize = false);

    /// <summary>Whisper preset, the log-mel of OpenAI's <c>log_mel_spectrogram</c> and HF's
    /// <c>WhisperFeatureExtractor</c>: 16 kHz, a 400-point STFT (201 bins) of a periodic Hann window at hop 160,
    /// centered by reflect padding of 200, last frame dropped (3000 frames for 30 s), power spectrum, Slaney filters to
    /// 8 kHz, log10 floored at 1e-10, then the max − 8 clamp and (x + 4) / 4. 80 bins through large-v2; pass
    /// <paramref name="nMels"/> = 128 for large-v3, large-v3-turbo and the distil-large-v3 models.</summary>
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
        PowerSpectrum: true,
        Center: true,
        ExactFftSize: true);

    /// <summary>The Whisper preset before it matched Whisper: the 400-sample window zero-padded into a 512-point FFT
    /// (257 bins) and no centering. Only for the S3 speech-tokenizer front-end (<c>S3GenReference</c>), which pads the
    /// reference itself and must keep its output until it gets its own parity check; new callers use
    /// <see cref="WhisperConfig"/>.</summary>
    public static Config WhisperLegacyPow2Config(int nMels = 80)
        => WhisperConfig(nMels) with { Center = false, ExactFftSize = false };

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

    /// <summary>Zonos speaker-encoder logFbank front-end: torchaudio <c>MelSpectrogram(sr=16k, n_fft=512,
    /// win=400, hop=160, n_mels=80, power=2, center=True, norm=None, mel_scale="htk")</c> then
    /// <c>log(mel + 1e-6)</c>. The per-mel time-mean subtraction that follows in the reference is applied by the
    /// caller (<c>ZonosSpeakerEncoder</c>) since it needs the full frame block. Win 400 &lt; n_fft 512 → the
    /// analysis window is centered in the FFT frame.</summary>
    public static Config Zonos16kConfig() => new(
        SampleRate: 16_000,
        NFft: 512,
        WinLength: 400,
        HopLength: 160,
        NMels: 80,
        Fmin: 0.0,
        Fmax: 8000.0,
        Norm: Normalization.None,
        DropLastStftFrame: false,
        LogBase: LogBase.Natural,
        LogFloor: 1e-6f,
        DynamicRangeDb: 0f,
        NormOffset: 0f,
        NormScale: 1f,
        PowerSpectrum: true,
        Scale: MelScale.Htk,
        SlaneyNorm: false,
        Center: true,
        CenterWindowInFft: true,
        AdditiveLogFloor: true);

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

    /// <summary><see cref="None"/> skips log compression entirely (raw amp/power mel, e.g. the Chatterbox
    /// voice-encoder front-end's <c>mel_type="amp"</c>).</summary>
    public enum LogBase { Log10, Natural, None }

    private readonly Config _cfg;
    private readonly float[] _window;
    private readonly int _numBins;
    private readonly int _fftSize;
    // Window position in the FFT frame: torch.stft centers a shorter window, the default left-aligns it.
    private readonly int _windowOffset;
    // Reflect padding of a centered STFT; frame t reads virtual samples from t·hop − _centerPad.
    private readonly int _centerPad;
    // The exact transform for a size that is not a power of two; null runs the radix-2 path.
    private readonly FftPlan? _plan;

    // The filterbank row-major and flat, with each row's nonzero bin range: a mel filter is a triangle a few bins
    // wide, so summing only [start, end) skips products that are exactly +0 and leaves every sum bit-identical for
    // finite input (a NaN or infinite bin times a zero weight is NaN, which the full sum would have carried).
    private readonly float[] _filterWeights;   // [n_mels * _numBins]
    private readonly int[] _filterStart;       // [n_mels]
    private readonly int[] _filterEnd;         // [n_mels]

    // The column of a frame whose samples are all zero: what transforming one produces, so it is filled instead.
    private readonly float _silent;

    // A frame's scratch: frame | re | im | power | mel | plan work. Blocks rent theirs; this one is ComputeFrame's.
    private readonly int _scratchLength;
    private readonly float[] _frameScratch;

    /// <summary>Builds the window, filterbank and FFT plan for <paramref name="cfg"/>.</summary>
    public MelSpectrogramExtractor(Config cfg)
    {
        _fftSize = cfg.ExactFftSize ? cfg.NFft : Fft.NextPow2(cfg.NFft);
        if (cfg.HopLength < 1 || cfg.WinLength < 1 || cfg.WinLength > _fftSize || cfg.NMels < 1)
            throw new ArgumentException(
                $"invalid STFT: win {cfg.WinLength}, hop {cfg.HopLength}, FFT {_fftSize}, mels {cfg.NMels}.", nameof(cfg));
        if (cfg.ExactFftSize && (_fftSize & (_fftSize - 1)) != 0)
        {
            if (!FftPlan.IsSupported(_fftSize))
                throw new ArgumentException(
                    $"an exact {_fftSize}-point STFT needs prime factors of 2, 3 and 5 only.", nameof(cfg));
            _plan = new FftPlan(_fftSize);
        }
        _cfg = cfg;
        _window = HannWindow.Get(cfg.WinLength);
        _numBins = _fftSize / 2 + 1;
        _windowOffset = cfg.CenterWindowInFft ? (_fftSize - cfg.WinLength) / 2 : 0;
        _centerPad = cfg.Center ? cfg.NFft / 2 : 0;
        float[,] filterbank = MelFilterbank.Get(cfg.SampleRate, _fftSize, cfg.NMels, cfg.Fmin, cfg.Fmax, cfg.Scale, cfg.SlaneyNorm);
        _filterWeights = new float[cfg.NMels * _numBins];
        _filterStart = new int[cfg.NMels];
        _filterEnd = new int[cfg.NMels];
        for (int m = 0; m < cfg.NMels; m++)
        {
            int start = -1, end = 0;
            for (int k = 0; k < _numBins; k++)
            {
                float w = filterbank[m, k];
                _filterWeights[m * _numBins + k] = w;
                if (w == 0f) continue;
                if (start < 0) start = k;
                end = k + 1;
            }
            _filterStart[m] = Math.Max(start, 0);
            _filterEnd[m] = end;
        }

        _silent = Compress(0f);
        _scratchLength = MelOffset + cfg.NMels + (_plan?.WorkLength ?? 0);
        _frameScratch = new float[_scratchLength];
    }

    /// <summary>The number of mel frames a given input audio length will produce.</summary>
    public int OutputFrames(int audioLength)
    {
        // torch.stft(center=True) reflect-pads by n_fft/2 each side and yields (len // hop) + 1 frames.
        // Without centering the caller is responsible for any pre-padding (e.g. Whisper zero-pads to 30s).
        int frames = _cfg.Center ? 1 + audioLength / _cfg.HopLength
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
        if (frames == 0)
        {
            return;
        }
        ComputeInto(audio, audio.Length, frames, MemoryMarshal.CreateSpan(ref output[0, 0], output.Length), output.GetLength(1));
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
        Span<float> scratch = _frameScratch;
        Span<float> frame = scratch[.._fftSize];
        int n = Math.Min(windowAudio.Length, _cfg.WinLength);
        for (int i = 0; i < n; i++) frame[i] = windowAudio[i] * _window[i];
        frame[n..].Clear();

        TransformToMel(scratch);
        ReadOnlySpan<float> mel = scratch.Slice(MelOffset, _cfg.NMels);
        for (int m = 0; m < _cfg.NMels; m++) melColumn[m] = Compress(mel[m]);
    }

    /// <summary>The log-mel of <paramref name="audio"/> zero-padded to <paramref name="paddedLength"/> samples, written
    /// row-major <c>[n_mels, OutputFrames(paddedLength)]</c> into <paramref name="output"/>: bit-for-bit what
    /// <see cref="Compute(ReadOnlySpan{float}, float[,])"/> returns for the padded buffer, without building it. A frame
    /// whose window reads only padding, through the reflection at either edge too for a centered preset, is all zeros,
    /// so its column is one constant and skips the transform — most of Whisper's 30 s window for a short
    /// utterance.</summary>
    public void ComputeZeroPadded(ReadOnlySpan<float> audio, int paddedLength, Span<float> output)
    {
        if (audio.Length > paddedLength)
            throw new ArgumentException($"audio has {audio.Length} samples, more than the padded length {paddedLength}.", nameof(audio));
        int frames = OutputFrames(paddedLength);
        if (output.Length < _cfg.NMels * frames)
            throw new ArgumentException($"output must hold [{_cfg.NMels}, {frames}] values.", nameof(output));
        ComputeInto(audio, paddedLength, frames, output, frames);
    }

    /// <summary>The log-mel of <paramref name="frames"/> frames into row-major <paramref name="output"/> (row stride
    /// <paramref name="stride"/>), normalized per the preset. The signal is <paramref name="signalLength"/> samples, of
    /// which <paramref name="src"/> holds the leading ones and the rest are zero; a centered preset reflects at the
    /// signal's edges as <see cref="SignalPadding.Reflect"/> would, without materializing anything. Frames fan out in
    /// size-fixed blocks; each frame's column depends only on its own samples, and the max the Whisper clamp needs is
    /// taken afterwards, so the result is the same however the blocks are scheduled.</summary>
    private unsafe void ComputeInto(ReadOnlySpan<float> src, int signalLength, int frames, Span<float> output, int stride)
    {
        int framesPerBlock = FramePartition.FramesPerBlock(_fftSize, FramePartition.TransformBudget);
        int blocks = FramePartition.BlockCount(frames, framesPerBlock);
        fixed (float* srcPtr = src)
        fixed (float* outPtr = output)
        {
            FrameBlocks job = new(this, srcPtr, src.Length, signalLength, frames, framesPerBlock, outPtr, stride);
            CpuParallel.For(blocks, frames * FramePartition.TransformWork(_fftSize), job, FrameBlockBody);
        }

        // Post-pass normalization (Whisper: dynamic-range clamp + (+4)/4 shift).
        if (_cfg.Norm == Normalization.WhisperDynamicRange)
        {
            float globalMax = float.MinValue;
            for (int m = 0; m < _cfg.NMels; m++)
            {
                ReadOnlySpan<float> row = output.Slice(m * stride, frames);
                for (int t = 0; t < frames; t++)
                    if (row[t] > globalMax) globalMax = row[t];
            }
            float clampMin = globalMax - _cfg.DynamicRangeDb;
            float invScale = 1f / _cfg.NormScale;
            for (int m = 0; m < _cfg.NMels; m++)
            {
                Span<float> row = output.Slice(m * stride, frames);
                for (int t = 0; t < frames; t++)
                    row[t] = (MathF.Max(row[t], clampMin) + _cfg.NormOffset) * invScale;
            }
        }
    }

    /// <summary>One block of <see cref="ComputeInto"/>'s frames, in scratch rented for the block.</summary>
    private static unsafe void RunFrameBlock(int block, FrameBlocks job)
    {
        MelSpectrogramExtractor self = job.Extractor;
        float[] scratch = ArrayPool<float>.Shared.Rent(self._scratchLength);
        try
        {
            ReadOnlySpan<float> src = new(job.Source, job.SourceLength);
            int end = Math.Min(job.Frames, (block + 1) * job.FramesPerBlock);
            for (int t = block * job.FramesPerBlock; t < end; t++)
                self.ComputeColumn(src, job.SignalLength, t, scratch, job.Output, job.Stride);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scratch);
        }
    }

    /// <summary>Writes frame <paramref name="t"/>'s compressed mel column into <paramref name="output"/>.</summary>
    private unsafe void ComputeColumn(ReadOnlySpan<float> src, int signalLength, int t, Span<float> scratch, float* output,
        int stride)
    {
        // Virtual index of the window's first sample: negative or past the signal end means reflected.
        int first = t * _cfg.HopLength - _centerPad + _windowOffset;
        int last = first + _cfg.WinLength - 1;
        if (ReadsOnlyPadding(first, last, src.Length, signalLength))
        {
            for (int m = 0; m < _cfg.NMels; m++) output[(long)m * stride + t] = _silent;
            return;
        }

        Span<float> frame = scratch[.._fftSize];
        frame[.._windowOffset].Clear();
        frame[(_windowOffset + _cfg.WinLength)..].Clear();
        Span<float> windowed = frame.Slice(_windowOffset, _cfg.WinLength);
        if (first >= 0 && (!_cfg.Center || last < signalLength))
        {
            // No reflection: samples past the end of src are the zero padding (a +0 product with the window).
            int real = Math.Clamp(src.Length - first, 0, _cfg.WinLength);
            for (int i = 0; i < real; i++) windowed[i] = src[first + i] * _window[i];
            windowed[real..].Clear();
        }
        else
        {
            // torch.stft(center=True) reflects at both signal edges; past the reflect padding a read is zero.
            int limit = signalLength + _centerPad;
            for (int i = 0; i < _cfg.WinLength; i++)
            {
                int j = first + i;
                int index = j < limit ? SignalPadding.ReflectIndex(j, signalLength) : int.MaxValue;
                float sample = index < src.Length ? src[index] : 0f;
                windowed[i] = sample * _window[i];
            }
        }

        TransformToMel(scratch);
        ReadOnlySpan<float> mel = scratch.Slice(MelOffset, _cfg.NMels);
        for (int m = 0; m < _cfg.NMels; m++) output[(long)m * stride + t] = Compress(mel[m]);
    }

    /// <summary>Whether every sample the window <c>[first, last]</c> reads is padding: at or past
    /// <paramref name="realLength"/>, including the ones a centered preset mirrors back from past the signal end.
    /// False wherever the read would mirror more than once; transforming zeros gives the same column anyway.</summary>
    private bool ReadsOnlyPadding(int first, int last, int realLength, int signalLength)
    {
        if (first < realLength) return false;
        if (!_cfg.Center || last < signalLength) return true;
        int limit = signalLength + _centerPad;
        if (first >= limit) return true;
        return 2L * (signalLength - 1) - Math.Min(last, limit - 1) >= realLength;
    }

    /// <summary>Transforms the frame at the start of <paramref name="scratch"/> and leaves the power or magnitude
    /// spectrum through the filterbank in its mel slot. Reads only immutable state, so blocks run it
    /// concurrently.</summary>
    private void TransformToMel(Span<float> scratch)
    {
        Span<float> frame = scratch[.._fftSize];
        Span<float> re = scratch.Slice(_fftSize, _numBins);
        Span<float> im = scratch.Slice(_fftSize + _numBins, _numBins);
        Span<float> power = scratch.Slice(_fftSize + 2 * _numBins, _numBins);
        Span<float> mel = scratch.Slice(MelOffset, _cfg.NMels);
        if (_plan is not null)
        {
            _plan.ForwardReal(frame, re, im, scratch.Slice(MelOffset + _cfg.NMels, _plan.WorkLength));
        }
        else
        {
            Fft.RealTransform(frame, re, im, _fftSize);
        }

        if (_cfg.PowerSpectrum)
        {
            for (int k = 0; k < _numBins; k++)
                power[k] = re[k] * re[k] + im[k] * im[k];
        }
        else
        {
            for (int k = 0; k < _numBins; k++)
                power[k] = MathF.Sqrt(re[k] * re[k] + im[k] * im[k]);
        }

        for (int m = 0; m < _cfg.NMels; m++)
        {
            float acc = 0f;
            int row = m * _numBins;
            for (int k = _filterStart[m]; k < _filterEnd[m]; k++) acc += _filterWeights[row + k] * power[k];
            mel[m] = acc;
        }
    }

    /// <summary>Offset of the mel slot in a frame's scratch.</summary>
    private int MelOffset => _fftSize + 3 * _numBins;

    /// <summary>Log compression of one mel value (identity for <see cref="LogBase.None"/>).</summary>
    private float Compress(float mel)
    {
        if (_cfg.LogBase == LogBase.None)
        {
            return mel;
        }
        float floor = _cfg.LogFloor ?? 1e-10f;
        float v = _cfg.AdditiveLogFloor ? mel + floor : MathF.Max(mel, floor);
        return _cfg.LogBase == LogBase.Log10 ? MathF.Log10(v) : MathF.Log(v);
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

    /// <summary>One <see cref="ComputeInto"/> call, passed to the static block body rather than captured, so the
    /// delegate is built once and a call that runs inline allocates nothing.</summary>
    private readonly unsafe struct FrameBlocks(MelSpectrogramExtractor extractor, float* source, int sourceLength,
        int signalLength, int frames, int framesPerBlock, float* output, int stride)
    {
        public readonly MelSpectrogramExtractor Extractor = extractor;
        public readonly float* Source = source;
        public readonly int SourceLength = sourceLength;
        public readonly int SignalLength = signalLength;
        public readonly int Frames = frames;
        public readonly int FramesPerBlock = framesPerBlock;
        public readonly float* Output = output;
        public readonly int Stride = stride;
    }
}

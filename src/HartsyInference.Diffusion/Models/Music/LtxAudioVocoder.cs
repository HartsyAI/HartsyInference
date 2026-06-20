using System.Collections.Generic;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Music;

/// <summary>LTX-2 audio vocoder (<c>vocoder.*</c>) — turns the audio VAE's log-mel spectrogram into a stereo
/// waveform. Two checkpoint variants are auto-detected from the weight keys:</summary>
///
/// <remarks><para><b>BWE dual-stage</b> (diffusers <c>LTX2VocoderWithBWE</c>, the LTX-2.3 22B checkpoint, 48 kHz):
/// (1) the main anti-aliased SnakeBeta <see cref="LtxBigVganGenerator"/> (<c>vocoder.vocoder.*</c>) renders a
/// low-rate waveform; (2) a causal STFT re-derives a 64-bin log-mel, the BWE generator (<c>vocoder.bwe_generator.*</c>)
/// predicts a high-rate residual, and a Hann-windowed sinc <see cref="Resampler"/> upsamples the stage-1 waveform as
/// the low-band skip. Output = <c>clamp(residual + skip, -1, 1)</c>, cropped to the exact 48 kHz length.</para>
/// <para><b>Single-stage</b> (diffusers <c>LTX2Vocoder</c>, the LTX-2 19B checkpoint, 24 kHz): one leaky-ReLU
/// HiFi-GAN generator (<c>vocoder.*</c>) with a final tanh; no BWE/STFT/resampler.</para>
/// <para>The mel input is <c>[1, audioChannels, frames, melBins]</c> (audioChannels·melBins = the generator's
/// flattened input channels). The STFT bases (<c>forward_basis</c>) and mel filterbank (<c>mel_basis</c>) are loaded
/// from the checkpoint. Batch-1; stereo is carried as audioChannels = 2. Numerics validation-pending (the anti-alias
/// lowpass filters are recomputed rather than loaded; no reference activations captured yet).</para></remarks>
public sealed unsafe class LtxAudioVocoder : IDisposable
{
    // mel_stft config (BWE only).
    private const int FilterLength = 512;
    private const int HopLength = 80;
    private const int WindowLength = 512;
    private const int NumMelChannels = 64;
    private const int NumFreqs = FilterLength / 2 + 1;  // 257
    private const int InputSamplingRate = 16000;
    private const int OutputSamplingRate = 48000;

    private readonly int _audioChannels;
    private LtxBigVganGenerator? _main;
    private LtxBigVganGenerator? _bwe;
    private LtxBigVganGenerator? _single;
    private Resampler? _resampler;

    private Tensor? _forwardBasis;   // [NumFreqs*2, 1, FilterLength]
    private Tensor? _melBasis;       // [NumMelChannels, NumFreqs]

    /// <summary>Output waveform sample rate, known after <see cref="LoadWeights"/> (48 kHz BWE / 24 kHz single).</summary>
    public int SampleRate { get; private set; }

    public LtxAudioVocoder(int audioChannels = 2) => _audioChannels = audioChannels;

    /// <summary>Detects the variant from the checkpoint keys and builds the matching generator stack. The BWE main
    /// generator lives under <c>vocoder.vocoder.*</c>; the single-stage generator lives under <c>vocoder.*</c> (so
    /// the BWE marker <c>vocoder.bwe_generator.*</c> must be probed first).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        if (w.ContainsKey("vocoder.bwe_generator.conv_pre.weight"))
        {
            _main = new LtxBigVganGenerator(
                inChannels: 128, hiddenChannels: 1536, outChannels: _audioChannels,
                upsampleFactors: [5, 2, 2, 2, 2, 2], upsampleKernels: [11, 4, 4, 4, 4, 4],
                resnetKernels: [3, 7, 11], resnetDilations: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                applyTanh: false);
            _bwe = new LtxBigVganGenerator(
                inChannels: 128, hiddenChannels: 512, outChannels: _audioChannels,
                upsampleFactors: [6, 5, 2, 2, 2], upsampleKernels: [12, 11, 4, 4, 4],
                resnetKernels: [3, 7, 11], resnetDilations: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                applyTanh: false);
            _resampler = new Resampler(OutputSamplingRate / InputSamplingRate, _audioChannels);
            _main.LoadWeights(w, "vocoder.vocoder");
            _bwe.LoadWeights(w, "vocoder.bwe_generator");
            _forwardBasis = w["vocoder.mel_stft.stft_fn.forward_basis"];
            _melBasis = AsF32(w["vocoder.mel_stft.mel_basis"]);
            SampleRate = OutputSamplingRate;
            return;
        }
        if (w.ContainsKey("vocoder.conv_pre.weight"))
        {
            _single = new LtxBigVganGenerator(
                inChannels: 128, hiddenChannels: 1024, outChannels: _audioChannels,
                upsampleFactors: [6, 5, 2, 2, 2], upsampleKernels: [16, 15, 8, 4, 4],
                resnetKernels: [3, 7, 11], resnetDilations: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                applyTanh: true, activation: LtxVocoderActivation.LeakyRelu);
            _single.LoadWeights(w, "vocoder");
            SampleRate = 24000;
            return;
        }
        throw new KeyNotFoundException(
            "No LTX-2 vocoder weights found: expected either 'vocoder.bwe_generator.conv_pre.weight' (BWE) " +
            "or 'vocoder.conv_pre.weight' (single-stage).");
    }

    /// <summary>Vocode a log-mel spectrogram <c>[1, audioChannels, frames, melBins]</c> into a waveform
    /// <c>[1, audioChannels, samples]</c> in [-1, 1] at <see cref="SampleRate"/>. Caller owns the returned tensor.</summary>
    public Tensor Forward(IBackend backend, Tensor mel)
    {
        // Single-stage: one generator (it applies the final tanh) and we are done.
        if (_single is not null)
        {
            Tensor sIn = FlattenMel(mel);
            Tensor sOut = _single.Forward(backend, sIn);
            sIn.Dispose();
            return sOut;
        }

        // 1. Stage-1 generator: flatten (audioChannels, melBins) → input channels, then render the low-rate waveform.
        Tensor mainIn = FlattenMel(mel);
        Tensor x = _main!.Forward(backend, mainIn);   // [1, audioChannels, numSamples]
        mainIn.Dispose();

        int numSamples = (int)x.Shape[2];
        // Pad to an exact multiple of hop_length so the STFT frame count is exact.
        int remainder = numSamples % HopLength;
        if (remainder != 0)
        {
            Tensor padded = PadRight(x, HopLength - remainder);
            x.Dispose();
            x = padded;
        }

        // 2. Causal log-mel of the stage-1 waveform → [1, audioChannels, NumMelChannels, frames].
        Tensor logMel = MelSpectrogram(backend, x);

        // 3. BWE generator on the new mel → high-rate residual [1, audioChannels, bweSamples].
        Tensor bweIn = FlattenMel(TransposeLast(logMel));   // [1, C, frames, mel] → [1, C*mel, frames]
        logMel.Dispose();
        Tensor residual = _bwe!.Forward(backend, bweIn);
        bweIn.Dispose();

        // 4. Residual + resampled stage-1 skip, clamped and cropped to the exact output length.
        Tensor skip = _resampler!.Forward(backend, x);
        int outputSamples = (int)((long)numSamples * OutputSamplingRate / InputSamplingRate);
        Tensor waveform = AddClampCrop(residual, skip, outputSamples);
        residual.Dispose();
        skip.Dispose();
        x.Dispose();
        return waveform;
    }

    /// <summary>Causal log-mel: <c>[1, C, samples] → magnitude STFT → mel filterbank → log</c>, producing
    /// <c>[1, C, NumMelChannels, frames]</c>. The per-channel STFT runs as a stride-hop conv with the loaded DFT
    /// bases (left-only causal padding).</summary>
    private Tensor MelSpectrogram(IBackend backend, Tensor x)
    {
        int c = (int)x.Shape[1], samples = (int)x.Shape[2];
        // CausalSTFT: input [C, 1, samples], left_pad = window_length - hop_length.
        Tensor stftIn = new(new TensorShape(c, 1, samples), DType.F32);
        Buffer.MemoryCopy((float*)x.DataPointer, (float*)stftIn.DataPointer, (long)c * samples * 4, (long)c * samples * 4);
        int leftPad = Math.Max(0, WindowLength - HopLength);
        int frames = (samples + leftPad - FilterLength) / HopLength + 1;
        Tensor spec = new(new TensorShape(c, NumFreqs * 2, frames), DType.F32);
        backend.Conv1d(spec, stftIn, _forwardBasis!, null, HopLength, leftPad, 0, 1, 1);
        stftIn.Dispose();

        // magnitude = sqrt(real^2 + imag^2), real = spec[:, :NumFreqs], imag = spec[:, NumFreqs:].
        Tensor mag = new(new TensorShape(c, NumFreqs, frames), DType.F32);
        float* sp = (float*)spec.DataPointer;
        float* mp = (float*)mag.DataPointer;
        long specCH = (long)(NumFreqs * 2) * frames;
        long magCH = (long)NumFreqs * frames;
        for (int ch = 0; ch < c; ch++)
        {
            float* sBase = sp + ch * specCH;
            float* mBase = mp + ch * magCH;
            for (int f = 0; f < NumFreqs; f++)
            {
                float* re = sBase + (long)f * frames;
                float* im = sBase + (long)(NumFreqs + f) * frames;
                float* o = mBase + (long)f * frames;
                for (int t = 0; t < frames; t++) o[t] = MathF.Sqrt(re[t] * re[t] + im[t] * im[t]);
            }
        }
        spec.Dispose();

        // mel = mel_basis [NumMel, NumFreqs] @ magnitude [C, NumFreqs, frames]; log_mel = log(clamp(mel, 1e-5)).
        Tensor logMel = new(new TensorShape(1, c, NumMelChannels, frames), DType.F32);
        float* wp = (float*)_melBasis!.DataPointer;
        float* lp = (float*)logMel.DataPointer;
        for (int ch = 0; ch < c; ch++)
        {
            float* magBase = mp + ch * magCH;
            float* outBase = lp + (long)ch * NumMelChannels * frames;
            for (int m = 0; m < NumMelChannels; m++)
            {
                float* wrow = wp + (long)m * NumFreqs;
                float* orow = outBase + (long)m * frames;
                for (int t = 0; t < frames; t++)
                {
                    double acc = 0.0;
                    for (int f = 0; f < NumFreqs; f++) acc += wrow[f] * magBase[(long)f * frames + t];
                    orow[t] = MathF.Log(MathF.Max((float)acc, 1e-5f));
                }
            }
        }
        mag.Dispose();
        return logMel;
    }

    /// <summary>Flattens a mel tensor <c>[1, C, frames, melBins]</c> into generator input <c>[1, C·melBins,
    /// frames]</c> (the reference transposes time-last then merges the channel and mel-bin axes).</summary>
    private static Tensor FlattenMel(Tensor mel)
    {
        int c = (int)mel.Shape[1], frames = (int)mel.Shape[2], melBins = (int)mel.Shape[3];
        Tensor o = new(new TensorShape(1, c * melBins, frames), DType.F32);
        float* xp = (float*)mel.DataPointer;
        float* op = (float*)o.DataPointer;
        // mel[c, frame, bin] → out[c*melBins + bin, frame].
        for (int ci = 0; ci < c; ci++)
            for (int t = 0; t < frames; t++)
                for (int bin = 0; bin < melBins; bin++)
                    op[(long)(ci * melBins + bin) * frames + t] = xp[((long)ci * frames + t) * melBins + bin];
        return o;
    }

    /// <summary>Transposes the last two axes of <c>[1, C, A, B] → [1, C, B, A]</c> (mel time/bin swap before BWE).</summary>
    private static Tensor TransposeLast(Tensor x)
    {
        int c = (int)x.Shape[1], a = (int)x.Shape[2], b = (int)x.Shape[3];
        Tensor o = new(new TensorShape(1, c, b, a), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)o.DataPointer;
        for (int ci = 0; ci < c; ci++)
            for (int ai = 0; ai < a; ai++)
                for (int bi = 0; bi < b; bi++)
                    op[((long)ci * b + bi) * a + ai] = xp[((long)ci * a + ai) * b + bi];
        return o;
    }

    /// <summary>Right-pads the time axis of <c>[1, C, T]</c> with zeros.</summary>
    private static Tensor PadRight(Tensor x, int pad)
    {
        int c = (int)x.Shape[1], t = (int)x.Shape[2], t2 = t + pad;
        Tensor o = new(new TensorShape(1, c, t2), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)o.DataPointer;
        for (int ci = 0; ci < c; ci++)
        {
            new Span<float>(op + (long)ci * t2, t2).Clear();
            Buffer.MemoryCopy(xp + (long)ci * t, op + (long)ci * t2, (long)t * 4, (long)t * 4);
        }
        return o;
    }

    /// <summary><c>clamp(residual + skip, -1, 1)</c> then crop to <paramref name="outputSamples"/>. The two inputs
    /// are aligned to their shared length defensively (exact-length parity is validation-pending).</summary>
    private Tensor AddClampCrop(Tensor residual, Tensor skip, int outputSamples)
    {
        int c = (int)residual.Shape[1];
        int len = Math.Min((int)residual.Shape[2], (int)skip.Shape[2]);
        int outLen = Math.Min(len, outputSamples);
        Tensor o = new(new TensorShape(1, c, outLen), DType.F32);
        float* rp = (float*)residual.DataPointer;
        float* kp = (float*)skip.DataPointer;
        float* op = (float*)o.DataPointer;
        int rT = (int)residual.Shape[2], kT = (int)skip.Shape[2];
        for (int ci = 0; ci < c; ci++)
            for (int t = 0; t < outLen; t++)
            {
                float v = rp[(long)ci * rT + t] + kp[(long)ci * kT + t];
                op[(long)ci * outLen + t] = Math.Clamp(v, -1f, 1f);
            }
        return o;
    }

    private static Tensor AsF32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    public void Dispose()
    {
        _main?.Dispose();
        _bwe?.Dispose();
        _single?.Dispose();
        _resampler?.Dispose();
    }

    /// <summary>Hann-windowed sinc resampler (the reference's <c>UpSample1d(window_type="hann")</c>): replicate-pad,
    /// depthwise transposed conv ×ratio with a computed Hann sinc lowpass, scale by ratio, crop. Used to lift the
    /// stage-1 waveform to the output rate for the residual skip connection.</summary>
    private sealed class Resampler : IDisposable
    {
        private readonly int _ratio;
        private readonly int _channels;
        private readonly int _kernel;
        private readonly int _pad, _cropLeft, _cropRight;
        private readonly Tensor _weight;   // [channels, 1, kernel]

        public Resampler(int ratio, int channels)
        {
            _ratio = ratio;
            _channels = channels;
            float[] filter = LtxVocoderDsp.HannSincFilter(ratio, out _pad, out _cropLeft, out _cropRight);
            _kernel = filter.Length;
            _weight = new Tensor(new TensorShape(channels, 1, _kernel), DType.F32);
            float* wp = (float*)_weight.DataPointer;
            for (int c = 0; c < channels; c++)
                for (int k = 0; k < _kernel; k++)
                    wp[(long)c * _kernel + k] = filter[k];
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            int c = (int)x.Shape[1], t = (int)x.Shape[2];
            // Replicate-pad both sides by _pad.
            int tp = t + 2 * _pad;
            Tensor padded = new(new TensorShape(1, c, tp), DType.F32);
            float* xp = (float*)x.DataPointer;
            float* pp = (float*)padded.DataPointer;
            for (int ci = 0; ci < c; ci++)
            {
                float* src = xp + (long)ci * t;
                float* dst = pp + (long)ci * tp;
                float left = src[0], right = src[t - 1];
                for (int i = 0; i < _pad; i++) dst[i] = left;
                Buffer.MemoryCopy(src, dst + _pad, (long)t * 4, (long)t * 4);
                for (int i = 0; i < _pad; i++) dst[_pad + t + i] = right;
            }

            int tOut = (tp - 1) * _ratio + (_kernel - 1) + 1;
            Tensor convt = new(new TensorShape(1, c, tOut), DType.F32);
            backend.ConvTranspose1d(convt, padded, _weight, null, _ratio, 0, 0, 1, c);
            padded.Dispose();
            backend.Scale(convt, convt, _ratio);

            // Crop [cropLeft : -cropRight].
            int cropped = tOut - _cropLeft - _cropRight;
            Tensor o = new(new TensorShape(1, c, cropped), DType.F32);
            float* cp = (float*)convt.DataPointer;
            float* op = (float*)o.DataPointer;
            for (int ci = 0; ci < c; ci++)
                Buffer.MemoryCopy(cp + (long)ci * tOut + _cropLeft, op + (long)ci * cropped, (long)cropped * 4, (long)cropped * 4);
            convt.Dispose();
            return o;
        }

        public void Dispose() => _weight.Dispose();
    }
}

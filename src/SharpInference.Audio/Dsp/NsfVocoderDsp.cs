using SharpInference.Audio.Preprocessing;
using SharpInference.Audio.Models.Vocoders;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Dsp;

/// <summary>Shared DSP for NSF (neural source-filter) iSTFT vocoders — the building blocks common to
/// Kokoro's iSTFTNet decoder and CosyVoice's HiFTNet (and any future iSTFT vocoder). Centralizing these
/// avoids a per-model copy of the same harmonic-source / STFT / overlap-add math: each model differs
/// only in parameters (upsample scale, n_fft, hop, harmonic count), passed as arguments here.</summary>
public static unsafe class NsfVocoderDsp
{
    /// <summary>SourceModuleHnNSF harmonic-plus-noise source: nearest-upsamples F0 by
    /// <paramref name="scale"/> to audio rate, sums <paramref name="harmonics"/> phase-accumulated sines
    /// (deterministic phase, voiced/unvoiced + fixed-seed Gaussian noise shaping), then merges via
    /// <c>tanh(Linear)</c> using <paramref name="mergeW"/> (<c>[1, harmonics]</c>) + <paramref name="mergeB"/>.
    /// <paramref name="f0"/> is <c>[1, 1, T0]</c> in Hz; returns a float[<c>T0 · scale</c>] waveform.</summary>
    public static float[] GenerateHarmonicSource(Tensor f0, int scale, int sampleRate, int harmonics,
        Tensor mergeW, Tensor mergeB, float sineAmp = 0.1f, float noiseStd = 0.003f,
        float voicedThreshold = 10f, int noiseSeed = 0)
    {
        int t0 = (int)f0.Shape[2];
        int audioLen = t0 * scale;
        float* fp = (float*)f0.DataPointer;
        float* mW = (float*)mergeW.DataPointer;
        float mB = ((float*)mergeB.DataPointer)[0];
        double[] cum = new double[harmonics];
        float[] merged = new float[audioLen];
        uint rng = noiseSeed == 0 ? 0x9E3779B9u : DeterministicRng.Seed(noiseSeed);
        for (int i = 0; i < t0; i++)
        {
            float hz = fp[i];
            float uv = hz > voicedThreshold ? 1f : 0f;
            float noiseAmp = uv * noiseStd + (1f - uv) * (sineAmp / 3f);
            for (int rep = 0; rep < scale; rep++)
            {
                float lin = mB;
                for (int h = 0; h < harmonics; h++)
                {
                    cum[h] += (double)hz * (h + 1) / sampleRate;
                    cum[h] -= Math.Floor(cum[h]);
                    float sine = (float)Math.Sin(2.0 * Math.PI * cum[h]) * sineAmp;
                    float noise = noiseAmp * DeterministicRng.NextGaussian(ref rng);
                    lin += mW[h] * (sine * uv + noise);
                }
                merged[i * scale + rep] = MathF.Tanh(lin);
            }
        }
        return merged;
    }

    /// <summary>Forward STFT (Hann window, <c>center=True</c> reflect padding, <c>normalized=False</c>)
    /// producing magnitude in channels <c>[0, n_fft/2]</c> and phase angle in <c>[n_fft/2+1, n_fft+1]</c>
    /// of a <c>[1, n_fft+2, frames]</c> tensor — the <c>cat([|STFT|, angle(STFT)])</c> NSF source spectrogram.</summary>
    public static Tensor ForwardStftMagPhase(float[] signal, int nFft, int hop)
    {
        int half = nFft / 2;
        int numBins = half + 1;
        int pad = half;
        int paddedLen = signal.Length + 2 * pad;
        float[] padded = new float[paddedLen];
        for (int i = 0; i < pad; i++)
        {
            padded[i] = signal[Math.Min(pad - i, signal.Length - 1)];
            padded[paddedLen - 1 - i] = signal[Math.Max(signal.Length - 2 - i, 0)];
        }
        Array.Copy(signal, 0, padded, pad, signal.Length);
        int frames = 1 + (paddedLen - nFft) / hop;
        if (frames < 1) frames = 1;
        float[] window = HannWindow.Get(nFft);
        Tensor outT = new(new TensorShape(1, nFft + 2, frames), DType.F32);
        float* op = (float*)outT.DataPointer;
        float[] frame = new float[nFft];
        float[] re = new float[numBins];
        float[] im = new float[numBins];
        for (int f = 0; f < frames; f++)
        {
            int start = f * hop;
            for (int k = 0; k < nFft; k++) frame[k] = padded[start + k] * window[k];
            Fft.RealTransform(frame, re, im, nFft);
            for (int b = 0; b < numBins; b++)
            {
                op[b * frames + f] = MathF.Sqrt(re[b] * re[b] + im[b] * im[b]);
                op[(numBins + b) * frames + f] = MathF.Atan2(im[b], re[b]);
            }
        }
        return outT;
    }

    /// <summary>iSTFT output head: <c>magnitude = exp(post[0:nFft/2+1])</c>, <c>phase = sin(post[nFft/2+1:])</c>,
    /// then <c>iSTFT(magnitude·e^{j·phase})</c>. <paramref name="post"/> is channels-first
    /// <c>[1, n_fft+2, frames]</c>; returns the time-domain waveform.</summary>
    public static float[] IstftHead(Tensor post, int nFft, int hop)
    {
        int numBins = nFft / 2 + 1;
        int frames = (int)post.Shape[2];
        float* pp = (float*)post.DataPointer;
        float[] real = new float[frames * numBins];
        float[] imag = new float[frames * numBins];
        for (int f = 0; f < frames; f++)
            for (int b = 0; b < numBins; b++)
            {
                float mag = MathF.Exp(pp[b * frames + f]);
                float ang = MathF.Sin(pp[(numBins + b) * frames + f]);
                real[f * numBins + b] = mag * MathF.Cos(ang);
                imag[f * numBins + b] = mag * MathF.Sin(ang);
            }
        return IStft.Apply(real, imag, frames, nFft, hop);
    }

    /// <summary>ReflectionPad1d((1,0)) on a channels-first <c>[1, C, T]</c>: prepends one left sample by
    /// reflection (out[0] = x[1]).</summary>
    public static Tensor ReflectionPadLeft1(Tensor x)
    {
        int c = (int)x.Shape[1];
        int t = (int)x.Shape[2];
        Tensor outT = new(new TensorShape(1, c, t + 1), DType.F32);
        float* ip = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int cc = 0; cc < c; cc++)
        {
            long src = (long)cc * t;
            long dst = (long)cc * (t + 1);
            op[dst] = ip[src + Math.Min(1, t - 1)];
            for (int j = 0; j < t; j++) op[dst + 1 + j] = ip[src + j];
        }
        return outT;
    }

    /// <summary>In-place <c>dst += src</c> over the overlapping channel/time prefix (crops to the shorter
    /// extent to absorb ±1 conv-length rounding between branches).</summary>
    public static void AddInPlaceCropped(Tensor dst, Tensor src)
    {
        int dc = (int)dst.Shape[1], sc = (int)src.Shape[1];
        int c = Math.Min(dc, sc);
        int td = (int)dst.Shape[2], ts = (int)src.Shape[2];
        int t = Math.Min(td, ts);
        float* dp = (float*)dst.DataPointer;
        float* sp = (float*)src.DataPointer;
        for (int cc = 0; cc < c; cc++)
        {
            long db = (long)cc * td, sb = (long)cc * ts;
            for (int j = 0; j < t; j++) dp[db + j] += sp[sb + j];
        }
    }

    /// <summary>In-place scalar multiply of an entire tensor.</summary>
    public static void ScaleInPlace(Tensor x, float factor)
    {
        float* p = (float*)x.DataPointer;
        long n = x.ElementCount;
        for (long i = 0; i < n; i++) p[i] *= factor;
    }
}

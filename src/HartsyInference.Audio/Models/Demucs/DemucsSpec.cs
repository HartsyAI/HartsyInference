using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Demucs;

/// <summary>HTDemucs spectrogram front/back ends: the STFT → complex-as-channels (cac) encoding and its inverse.
/// <c>Spec</c> reflect-pads, runs <c>torch.stft(nfft, hop, hann, center=False, normalized=True)</c>, drops the
/// Nyquist bin, and lays the complex spectrogram out as channels (<c>[B,C,Fq,T,2] → [B,2C,Fq,T]</c>).
/// <c>InverseSpec</c> reverses it: restore the Nyquist bin, rebuild the interleaved complex spectrum, and run the
/// inverse STFT. n_fft 4096, hop 1024, periodic Hann. Reuses the backend <c>Stft</c> (forward) and
/// <see cref="IStft"/> (inverse).</summary>
public static unsafe class DemucsSpec
{
    /// <summary>Demucs trims 2 frames from each side of the center=False STFT (the reflect-pad overhang).</summary>
    private const int FrameTrim = 2;

    /// <summary>Forward spectrogram + cac. Input waveform <c>[1, C, L]</c> (channels-first, C audio channels);
    /// returns <c>[1, 2C, Fq, T]</c> (real/imag interleaved as channels, Nyquist dropped) and reports the
    /// frequency / time extents.</summary>
    public static Tensor Spec(IBackend backend, Tensor wav, int channels, int length, int nfft, int hop, out int freq, out int time)
    {
        int half = nfft / 2;
        freq = half;                                  // Nyquist bin dropped → nfft/2 freq rows
        int le = (length + hop - 1) / hop;            // ceil(length / hop)
        time = le;
        // demucs `_spec`: manual reflect-pad (pad = hop//2*3 left; pad + le*hop - L right) THEN `spectro` runs
        // torch.stft(center=True) which reflect-pads ANOTHER nfft/2 each side. We replicate the two successive
        // reflects exactly (a single combined reflect is NOT identical), then frame center=False, then trim the
        // first 2 frames (z[..., 2:2+le]).
        int manualL = hop / 2 * 3;
        int manualR = manualL + le * hop - length;
        int m = length + manualL + manualR;           // length after the manual pad
        int centerPad = half;                         // torch.stft(center=True) reflect pad
        int padded = m + 2 * centerPad;
        float* wp = (float*)wav.DataPointer;

        Tensor window = HannTensor(nfft);
        float norm = 1f / MathF.Sqrt(nfft);           // torch.stft(normalized=True)
        Tensor outT = new(new TensorShape(1, 2 * channels, freq, time), DType.F32);
        float* op = (float*)outT.DataPointer;

        Tensor manual = new(new TensorShape(m), DType.F32);
        float* mp = (float*)manual.DataPointer;
        Tensor frame = new(new TensorShape(padded), DType.F32);
        float* fp = (float*)frame.DataPointer;
        int rawFrames = (padded - nfft) / hop + 1;
        Tensor stftOut = new(new TensorShape(rawFrames, nfft * 2), DType.F32);
        float* sp = (float*)stftOut.DataPointer;

        for (int c = 0; c < channels; c++)
        {
            // Two-step reflect: original → manual pad → center=True pad.
            ReflectPad(wp + (long)c * length, length, mp, manualL, manualR);
            ReflectPad(mp, m, fp, centerPad, centerPad);
            backend.Stft(stftOut, frame, nfft, hop, window);
            // Keep frames [FrameTrim, FrameTrim+time) and freq bins [0, freq) (Nyquist at index `half` dropped).
            for (int t = 0; t < time; t++)
            {
                int srcFrame = Math.Min(t + FrameTrim, rawFrames - 1);
                float* row = sp + (long)srcFrame * nfft * 2;
                for (int k = 0; k < freq; k++)
                {
                    float re = row[2 * k] * norm;
                    float im = row[2 * k + 1] * norm;
                    op[(((long)(2 * c) * freq + k) * time) + t] = re;
                    op[(((long)(2 * c + 1) * freq + k) * time) + t] = im;
                }
            }
        }

        window.Dispose(); frame.Dispose(); stftOut.Dispose(); manual.Dispose();
        return outT;
    }

    /// <summary>Inverse of <see cref="Spec"/>. Input <c>[1, 2C, Fq, T]</c> (cac), restores the Nyquist bin and the
    /// trimmed temporal frames, then runs the inverse STFT per channel into a <c>[1, C, length]</c> waveform.</summary>
    public static Tensor InverseSpec(Tensor cac, int channels, int length, int nfft, int hop)
    {
        int freq = (int)cac.Shape[2];                 // = nfft/2
        int time = (int)cac.Shape[3];
        int numBins = nfft / 2 + 1;                    // iSTFT expects the [0, Nyquist] half
        float* cp = (float*)cac.DataPointer;
        float invNorm = MathF.Sqrt(nfft);              // undo normalized=True

        // iSTFT center=True trims n_fft/2 off each edge; we pad the frame count to recover `length` then crop.
        int frames = time + 2 * FrameTrim;
        float[] re = new float[(long)frames * numBins];
        float[] im = new float[(long)frames * numBins];
        Tensor outT = new(new TensorShape(1, channels, length), DType.F32);
        float* op = (float*)outT.DataPointer;

        // The decoder's reconstructed freq axis may not land exactly on nfft/2 (the encode/decode conv strides are
        // not bit-exact inverses without the reference padding); clamp to the iSTFT's non-negative half.
        int usableFreq = Math.Min(freq, numBins - 1);
        for (int c = 0; c < channels; c++)
        {
            Array.Clear(re); Array.Clear(im);
            for (int t = 0; t < time; t++)
            {
                int dstFrame = t + FrameTrim;
                int rowOff = dstFrame * numBins;
                for (int k = 0; k < usableFreq; k++)
                {
                    re[rowOff + k] = cp[(((long)(2 * c) * freq + k) * time) + t] * invNorm;
                    im[rowOff + k] = cp[(((long)(2 * c + 1) * freq + k) * time) + t] * invNorm;
                }
                // Nyquist bin (index `freq`) restored as zero — Demucs nets predict no Nyquist energy.
            }
            float[] wave = IStft.Apply(re, im, frames, nfft, hop);
            int n = Math.Min(length, wave.Length);
            for (int i = 0; i < n; i++) op[(long)c * length + i] = wave[i];
        }
        return outT;
    }

    private static void ReflectPad(float* src, int length, float* dst, int padLeft, int padRight)
    {
        for (int i = 0; i < padLeft; i++)
        {
            int idx = padLeft - i;                    // reflect (no edge repeat): src[1..]
            dst[i] = src[Reflect(idx, length)];
        }
        for (int i = 0; i < length; i++) dst[padLeft + i] = src[i];
        for (int i = 0; i < padRight; i++)
        {
            int idx = length - 2 - i;
            dst[padLeft + length + i] = src[Reflect(idx, length)];
        }
    }

    private static int Reflect(int idx, int length)
    {
        if (length <= 1) return 0;
        int period = 2 * (length - 1);
        int m = ((idx % period) + period) % period;
        return m < length ? m : period - m;
    }

    private static Tensor HannTensor(int nfft)
    {
        float[] w = HannWindow.Get(nfft);
        Tensor t = new(new TensorShape(nfft), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < nfft; i++) p[i] = w[i];
        return t;
    }
}

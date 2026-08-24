using HartsyInference.Audio.Preprocessing;
using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Dsp;

/// <summary>Shared DSP for NSF (neural source-filter) iSTFT vocoders — the building blocks common to
/// Kokoro's iSTFTNet decoder and CosyVoice's HiFTNet (and any future iSTFT vocoder). Centralizing these
/// avoids a per-model copy of the same harmonic-source / STFT / overlap-add math: each model differs
/// only in parameters (upsample scale, n_fft, hop, harmonic count), passed as arguments here.</summary>
// TODO(gpu-residency): every method here is host C# DSP over `(float*)DataPointer` (harmonic source, FFT/STFT,
// iSTFT overlap-add). On CUDA these force device→host syncs and break GPU residency for the HiFTNet vocoder.
// Port the harmonic-source + STFT + iSTFT to PTX kernels for a fully on-device mel→wav path.
public static unsafe class NsfVocoderDsp
{
    /// <summary>SourceModuleHnNSF harmonic-plus-noise source: nearest-upsamples F0 by
    /// <paramref name="scale"/> to audio rate, sums <paramref name="harmonics"/> phase-accumulated sines
    /// (deterministic phase, voiced/unvoiced + fixed-seed Gaussian noise shaping), then merges via
    /// <c>tanh(Linear)</c> using <paramref name="mergeW"/> (<c>[1, harmonics]</c>) + <paramref name="mergeB"/>.
    /// <paramref name="f0"/> is <c>[1, 1, T0]</c> in Hz; returns a float[<c>T0 · scale</c>] waveform.</summary>
    public static float[] GenerateHarmonicSource(Tensor f0, int scale, int sampleRate, int harmonics, Tensor mergeW,
        Tensor mergeB, float sineAmp = 0.1f, float noiseStd = 0.003f, float voicedThreshold = 10f, int noiseSeed = 0)
    {
        int t0 = (int)f0.Shape[2];
        float* fp = (float*)f0.DataPointer;
        float[] f0Array = new float[t0];
        for (int i = 0; i < t0; i++) f0Array[i] = fp[i];

        double[] cum = new double[harmonics];
        // noiseSeed < 0 → deterministic (no NSF noise), used by the parity harness; otherwise stochastic.
        bool addNoise = noiseSeed >= 0;
        uint rng = noiseSeed == 0 ? 0x9E3779B9u : DeterministicRng.Seed(Math.Abs(noiseSeed));
        return GenerateHarmonicSourceChunk(f0Array, cum, ref rng, scale, sampleRate, harmonics, mergeW, mergeB,
            sineAmp, noiseStd, voicedThreshold, addNoise);
    }

    /// <summary>Incremental counterpart to <see cref="GenerateHarmonicSource"/>: advances the SAME phase
    /// accumulators (<paramref name="phase"/>, one running sum per harmonic, mutated in place) and noise RNG
    /// state (<paramref name="rngState"/>) forward using only the NEW F0 values in <paramref name="f0Chunk"/> —
    /// never re-derives phase/noise for previously-consumed F0. Both are pure running sequences (phase is a
    /// cumulative sum, the RNG is a deterministic sequential walk), so threading them through successive calls
    /// with successive F0 chunks reproduces bit-identical results to one monolithic
    /// <see cref="GenerateHarmonicSource"/> call over the concatenation of all chunks — this is what makes the
    /// NSF source safe to stream (see <c>CosyVoice.HiFTStreamState</c>'s doc comment for why recompute-with-margin
    /// alone is NOT safe for this specific piece of the vocoder).</summary>
    public static float[] GenerateHarmonicSourceChunk(float[] f0Chunk, double[] phase, ref uint rngState,
        int scale, int sampleRate, int harmonics, Tensor mergeW, Tensor mergeB, float sineAmp, float noiseStd,
        float voicedThreshold, bool addNoise)
    {
        float* mW = (float*)mergeW.DataPointer;
        float mB = ((float*)mergeB.DataPointer)[0];
        float[] merged = new float[f0Chunk.Length * scale];
        uint rng = rngState;
        for (int i = 0; i < f0Chunk.Length; i++)
        {
            float hz = f0Chunk[i];
            float uv = hz > voicedThreshold ? 1f : 0f;
            float noiseAmp = uv * noiseStd + (1f - uv) * (sineAmp / 3f);
            for (int rep = 0; rep < scale; rep++)
            {
                float lin = mB;
                for (int h = 0; h < harmonics; h++)
                {
                    phase[h] += (double)hz * (h + 1) / sampleRate;
                    phase[h] -= Math.Floor(phase[h]);
                    float sine = (float)Math.Sin(2.0 * Math.PI * phase[h]) * sineAmp;
                    float noise = addNoise ? noiseAmp * DeterministicRng.NextGaussian(ref rng) : 0f;
                    lin += mW[h] * (sine * uv + noise);
                }
                merged[i * scale + rep] = MathF.Tanh(lin);
            }
        }
        rngState = rng;
        return merged;
    }

    /// <summary>Forward STFT (Hann window, <c>center=True</c> reflect padding, <c>normalized=False</c>)
    /// producing magnitude in channels <c>[0, n_fft/2]</c> and phase angle in <c>[n_fft/2+1, n_fft+1]</c>
    /// of a <c>[1, n_fft+2, frames]</c> tensor — the <c>cat([|STFT|, angle(STFT)])</c> NSF source spectrogram.</summary>
    public static Tensor ForwardStftMagPhase(float[] signal, int nFft, int hop)
        => ForwardStft(signal, nFft, hop, magPhase: true);

    /// <summary>Forward STFT (periodic Hann, <c>center=True</c> reflect padding) producing the real part in
    /// channels <c>[0, n_fft/2]</c> and the imaginary part in <c>[n_fft/2+1, n_fft+1]</c> of a
    /// <c>[1, n_fft+2, frames]</c> tensor — the <c>cat([Re(STFT), Im(STFT)])</c> NSF source spectrogram that
    /// HiFTGenerator's <c>source_downs</c> convs consume (NOT magnitude/phase).</summary>
    public static Tensor ForwardStftRealImag(float[] signal, int nFft, int hop)
        => ForwardStft(signal, nFft, hop, magPhase: false);

    /// <param name="magPhase">Writes magnitude/phase when true, real/imaginary when false — the only difference
    /// between the two published forms.</param>
    private static Tensor ForwardStft(float[] signal, int nFft, int hop, bool magPhase)
    {
        int half = nFft / 2;
        int numBins = half + 1;
        int pad = half;
        int paddedLen = signal.Length + 2 * pad;
        float[] padded = new float[paddedLen];
        // center=True reflection padding (torch default).
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
            if (magPhase)
            {
                for (int b = 0; b < numBins; b++)
                {
                    op[b * frames + f] = MathF.Sqrt(re[b] * re[b] + im[b] * im[b]);
                    op[(numBins + b) * frames + f] = MathF.Atan2(im[b], re[b]);
                }
            }
            else
            {
                for (int b = 0; b < numBins; b++)
                {
                    op[b * frames + f] = re[b];
                    op[(numBins + b) * frames + f] = im[b];
                }
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
                float mag = MathF.Min(MathF.Exp(pp[b * frames + f]), 1e2f);   // torch _istft clips magnitude to 1e2
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

using HartsyInference.Audio.Preprocessing;
using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Dsp;

/// <summary>Shared DSP for NSF (neural source-filter) iSTFT vocoders — the building blocks common to
/// Kokoro's iSTFTNet decoder and CosyVoice's HiFTNet (and any future iSTFT vocoder). Centralizing these
/// avoids a per-model copy of the same harmonic-source / STFT / overlap-add math: each model differs
/// only in parameters (upsample scale, n_fft, hop, harmonic count), passed as arguments here.
///
/// <para>Everything here is host DSP over <c>(float*)DataPointer</c>: the F0 read and the final spectrogram read
/// are the two device→host syncs of an iSTFT vocoder, and the work between them is spread over the cores
/// (frames are independent for the transforms; the harmonic source is split by jumping the phase and the noise
/// generator to each worker's start, so its output is bit-for-bit the sequential walk).</para></summary>
public static unsafe class NsfVocoderDsp
{
    /// <summary>Frames per worker below which a parallel split costs more than it saves.</summary>
    private const int MinFramesPerWorker = 64;

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
        => GenerateHarmonicSourceChunk(f0Chunk, phase, ref rngState, scale, sampleRate, harmonics, mergeW, mergeB,
            sineAmp, noiseStd, voicedThreshold, addNoise, Environment.ProcessorCount);

    /// <summary>The worker-count-explicit body of <see cref="GenerateHarmonicSourceChunk(float[], double[], ref uint, int, int, int, Tensor, Tensor, float, float, float, bool)"/>.
    /// Frames are split across workers; each starts from the phase the sequential walk would have reached
    /// (the per-harmonic sum is <c>(h+1)</c> times one cumulative F0 sum, taken mod 1) and from the noise state
    /// <see cref="DeterministicRng.Advance"/> jumps to, so every worker reproduces exactly the samples the
    /// sequential loop would have written there. <paramref name="maxWorkers"/> = 1 is that sequential loop.</summary>
    internal static float[] GenerateHarmonicSourceChunk(float[] f0Chunk, double[] phase, ref uint rngState,
        int scale, int sampleRate, int harmonics, Tensor mergeW, Tensor mergeB, float sineAmp, float noiseStd,
        float voicedThreshold, bool addNoise, int maxWorkers)
    {
        if (phase.Length < harmonics) throw new ArgumentException($"phase holds {phase.Length} accumulators, need {harmonics}.");
        int frames = f0Chunk.Length;
        float* mW = (float*)mergeW.DataPointer;
        float mB = ((float*)mergeB.DataPointer)[0];
        float[] merged = new float[(long)frames * scale];
        int workers = Math.Clamp(Math.Min(maxWorkers, frames / MinFramesPerWorker), 1, Math.Max(1, frames));
        if (workers == 1)
        {
            uint rng = rngState;
            RunHarmonicFrames(f0Chunk, 0, frames, phase, ref rng, scale, sampleRate, harmonics, mW, mB, sineAmp, noiseStd,
                voicedThreshold, addNoise, merged);
            rngState = rng;
            return merged;
        }

        // Cumulative F0 sum (in cycles at the fundamental) at each worker's first frame, sequential and cheap.
        int framesPerWorker = (frames + workers - 1) / workers;
        double[] startCycles = new double[workers];
        double cycles = 0;
        for (int w = 0, frame = 0; w < workers; w++)
        {
            startCycles[w] = cycles;
            int end = Math.Min(frames, frame + framesPerWorker);
            for (; frame < end; frame++) cycles += (double)f0Chunk[frame] * scale / sampleRate;
        }
        double[] initialPhase = new double[harmonics];
        Array.Copy(phase, initialPhase, harmonics);
        uint initialRng = rngState;
        double[][] finalPhase = new double[workers][];
        Parallel.For(0, workers, w =>
        {
            int start = w * framesPerWorker;
            int end = Math.Min(frames, start + framesPerWorker);
            double[] localPhase = new double[harmonics];
            for (int h = 0; h < harmonics; h++)
            {
                double p = initialPhase[h] + (h + 1) * startCycles[w];
                localPhase[h] = p - Math.Floor(p);
            }
            uint localRng = addNoise ? DeterministicRng.Advance(initialRng, 2L * start * scale * harmonics) : initialRng;
            RunHarmonicFrames(f0Chunk, start, end, localPhase, ref localRng, scale, sampleRate, harmonics, mW, mB, sineAmp,
                noiseStd, voicedThreshold, addNoise, merged);
            finalPhase[w] = localPhase;
        });
        Array.Copy(finalPhase[workers - 1], phase, harmonics);
        if (addNoise) rngState = DeterministicRng.Advance(initialRng, 2L * frames * scale * harmonics);
        return merged;
    }

    /// <summary>The per-frame harmonic-source loop over frames <c>[start, end)</c>, writing samples
    /// <c>[start·scale, end·scale)</c> of <paramref name="merged"/>.</summary>
    private static void RunHarmonicFrames(float[] f0Chunk, int start, int end, double[] phase, ref uint rng, int scale,
        int sampleRate, int harmonics, float* mW, float mB, float sineAmp, float noiseStd, float voicedThreshold,
        bool addNoise, float[] merged)
    {
        for (int i = start; i < end; i++)
        {
            float hz = f0Chunk[i];
            float uv = hz > voicedThreshold ? 1f : 0f;
            float noiseAmp = uv * noiseStd + (1f - uv) * (sineAmp / 3f);
            long sampleBase = (long)i * scale;
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
                merged[sampleBase + rep] = MathF.Tanh(lin);
            }
        }
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
        // Frames are independent and each writes its own column, so they split across the cores.
        Parallel.ForEach(FrameBlocks(frames), () => (Frame: new float[nFft], Re: new float[numBins], Im: new float[numBins]),
            (block, _, scratch) =>
            {
                for (int f = block.Start; f < block.End; f++)
                {
                    int start = f * hop;
                    for (int k = 0; k < nFft; k++) scratch.Frame[k] = padded[start + k] * window[k];
                    Fft.RealTransform(scratch.Frame, scratch.Re, scratch.Im, nFft);
                    if (magPhase)
                    {
                        for (int b = 0; b < numBins; b++)
                        {
                            float re = scratch.Re[b], im = scratch.Im[b];
                            op[(long)b * frames + f] = MathF.Sqrt(re * re + im * im);
                            op[(long)(numBins + b) * frames + f] = MathF.Atan2(im, re);
                        }
                    }
                    else
                    {
                        for (int b = 0; b < numBins; b++)
                        {
                            op[(long)b * frames + f] = scratch.Re[b];
                            op[(long)(numBins + b) * frames + f] = scratch.Im[b];
                        }
                    }
                }
                return scratch;
            },
            _ => { });
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
        float[] real = new float[(long)frames * numBins];
        float[] imag = new float[(long)frames * numBins];
        Parallel.ForEach(FrameBlocks(frames), block =>
        {
            for (int f = block.Start; f < block.End; f++)
                for (int b = 0; b < numBins; b++)
                {
                    float mag = MathF.Min(MathF.Exp(pp[(long)b * frames + f]), 1e2f);   // torch _istft clips magnitude to 1e2
                    float ang = MathF.Sin(pp[(long)(numBins + b) * frames + f]);
                    real[(long)f * numBins + b] = mag * MathF.Cos(ang);
                    imag[(long)f * numBins + b] = mag * MathF.Sin(ang);
                }
        });
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

    /// <summary>Splits <c>[0, frames)</c> into contiguous blocks sized for one worker each.</summary>
    internal static IEnumerable<(int Start, int End)> FrameBlocks(int frames)
    {
        int workers = Math.Clamp(frames / MinFramesPerWorker, 1, Environment.ProcessorCount);
        int perBlock = (frames + workers - 1) / workers;
        for (int start = 0; start < frames; start += perBlock)
        {
            yield return (start, Math.Min(frames, start + perBlock));
        }
    }
}

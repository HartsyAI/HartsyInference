using System.Buffers;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Dsp;

/// <summary>Shared DSP for NSF (neural source-filter) iSTFT vocoders — the building blocks common to
/// Kokoro's iSTFTNet decoder and CosyVoice's HiFTNet (and any future iSTFT vocoder). Centralizing these
/// avoids a per-model copy of the same harmonic-source / STFT / overlap-add math: each model differs
/// only in parameters (upsample scale, n_fft, hop, harmonic count), passed as arguments here.
///
/// <para>Everything here is host DSP over <c>(float*)DataPointer</c>: the F0 read and the final spectrogram read
/// are the two device→host syncs of an iSTFT vocoder. The work between them fans out through
/// <see cref="CpuParallel"/> over a <see cref="FramePartition"/> fixed by the input size, and every block
/// computes exactly what the sequential loop computes for its frames, so the output is bit-for-bit the
/// single-threaded result at any core count, any <c>numerics.cpuThreads</c> cap and inside
/// <see cref="CpuParallel.EnterInline"/>.</para></summary>
public static unsafe class NsfVocoderDsp
{
    /// <summary>Per-block budget of the harmonic source, in synthesized sample-harmonics.</summary>
    private const int HarmonicBlockBudget = 32768;

    /// <summary>Rough scalar cost of one sample-harmonic (a sine, a Box-Muller draw, the merge), for the serial threshold.</summary>
    private const int HarmonicWork = 32;

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
    /// alone is NOT safe for this specific piece of the vocoder).
    ///
    /// <para>Blocks of frames run in parallel. A sequential pass first walks the phase accumulators through every
    /// sample with the synthesis loop's own arithmetic and records each block's starting phases — additions and
    /// floors only, a small fraction of the sines and Gaussian draws — and each block's noise state is jumped to
    /// with <see cref="DeterministicRng.Advance"/>, so every block reproduces the sequential loop exactly.</para></summary>
    public static float[] GenerateHarmonicSourceChunk(float[] f0Chunk, double[] phase, ref uint rngState,
        int scale, int sampleRate, int harmonics, Tensor mergeW, Tensor mergeB, float sineAmp, float noiseStd,
        float voicedThreshold, bool addNoise)
    {
        if (phase.Length < harmonics) throw new ArgumentException($"phase holds {phase.Length} accumulators, need {harmonics}.");
        int frames = f0Chunk.Length;
        float* mW = (float*)mergeW.DataPointer;
        float mB = ((float*)mergeB.DataPointer)[0];
        float[] merged = new float[(long)frames * scale];
        int framesPerBlock = FramePartition.FramesPerBlock(scale * harmonics, HarmonicBlockBudget);
        int blocks = FramePartition.BlockCount(frames, framesPerBlock);
        if (blocks <= 1)
        {
            uint rng = rngState;
            RunHarmonicFrames(f0Chunk, 0, frames, phase.AsSpan(0, harmonics), ref rng, scale, sampleRate, harmonics, mW, mB,
                sineAmp, noiseStd, voicedThreshold, addNoise, merged);
            rngState = rng;
            return merged;
        }

        double[] blockPhase = ArrayPool<double>.Shared.Rent(blocks * harmonics);
        try
        {
            WalkPhases(f0Chunk, phase, blockPhase, framesPerBlock, blocks, scale, sampleRate, harmonics);
            uint initialRng = rngState;
            CpuParallel.For(blocks, (long)frames * scale * harmonics * HarmonicWork, b =>
            {
                int start = b * framesPerBlock;
                int end = Math.Min(frames, start + framesPerBlock);
                uint localRng = addNoise ? DeterministicRng.Advance(initialRng, 2L * start * scale * harmonics) : initialRng;
                RunHarmonicFrames(f0Chunk, start, end, blockPhase.AsSpan(b * harmonics, harmonics), ref localRng, scale,
                    sampleRate, harmonics, mW, mB, sineAmp, noiseStd, voicedThreshold, addNoise, merged);
            });
            if (addNoise) rngState = DeterministicRng.Advance(initialRng, 2L * frames * scale * harmonics);
        }
        finally
        {
            ArrayPool<double>.Shared.Return(blockPhase);
        }
        return merged;
    }

    /// <summary>Advances <paramref name="phase"/> through every sample of <paramref name="f0Chunk"/> exactly as
    /// <see cref="RunHarmonicFrames"/> does, copying the accumulators into <paramref name="blockPhase"/> at the
    /// first frame of each block.</summary>
    private static void WalkPhases(float[] f0Chunk, double[] phase, double[] blockPhase, int framesPerBlock, int blocks,
        int scale, int sampleRate, int harmonics)
    {
        double[] increment = ArrayPool<double>.Shared.Rent(harmonics);
        try
        {
            for (int b = 0; b < blocks; b++)
            {
                Array.Copy(phase, 0, blockPhase, b * harmonics, harmonics);
                int end = Math.Min(f0Chunk.Length, (b + 1) * framesPerBlock);
                for (int i = b * framesPerBlock; i < end; i++)
                {
                    float hz = f0Chunk[i];
                    // The synthesis loop's own expression, evaluated once per frame: the same IEEE operations on the
                    // same operands, so the same doubles.
                    for (int h = 0; h < harmonics; h++) increment[h] = (double)hz * (h + 1) / sampleRate;
                    for (int rep = 0; rep < scale; rep++)
                    {
                        for (int h = 0; h < harmonics; h++)
                        {
                            phase[h] += increment[h];
                            phase[h] -= Math.Floor(phase[h]);
                        }
                    }
                }
            }
        }
        finally
        {
            ArrayPool<double>.Shared.Return(increment);
        }
    }

    /// <summary>The per-frame harmonic-source loop over frames <c>[start, end)</c>, writing samples
    /// <c>[start·scale, end·scale)</c> of <paramref name="merged"/>.</summary>
    private static void RunHarmonicFrames(float[] f0Chunk, int start, int end, Span<double> phase, ref uint rng, int scale,
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
        float[] padded = ArrayPool<float>.Shared.Rent(paddedLen);
        try
        {
            // center=True reflection padding (torch default). Every element of [0, paddedLen) is written below.
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
            int framesPerBlock = FramePartition.FramesPerBlock(nFft, FramePartition.TransformBudget);
            int blocks = FramePartition.BlockCount(frames, framesPerBlock);
            // Frames are independent and each writes its own column.
            CpuParallel.For(blocks, frames * FramePartition.TransformWork(nFft), b =>
            {
                float[] scratch = ArrayPool<float>.Shared.Rent(nFft + 2 * numBins);
                try
                {
                    Span<float> frame = scratch.AsSpan(0, nFft);
                    Span<float> re = scratch.AsSpan(nFft, numBins);
                    Span<float> im = scratch.AsSpan(nFft + numBins, numBins);
                    int end = Math.Min(frames, (b + 1) * framesPerBlock);
                    for (int f = b * framesPerBlock; f < end; f++)
                    {
                        int start = f * hop;
                        for (int k = 0; k < nFft; k++) frame[k] = padded[start + k] * window[k];
                        Fft.RealTransform(frame, re, im, nFft);
                        if (magPhase)
                        {
                            for (int bin = 0; bin < numBins; bin++)
                            {
                                op[(long)bin * frames + f] = MathF.Sqrt(re[bin] * re[bin] + im[bin] * im[bin]);
                                op[(long)(numBins + bin) * frames + f] = MathF.Atan2(im[bin], re[bin]);
                            }
                        }
                        else
                        {
                            for (int bin = 0; bin < numBins; bin++)
                            {
                                op[(long)bin * frames + f] = re[bin];
                                op[(long)(numBins + bin) * frames + f] = im[bin];
                            }
                        }
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(scratch);
                }
            });
            return outT;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(padded);
        }
    }

    /// <summary>iSTFT output head: <c>magnitude = exp(post[0:nFft/2+1])</c>, <c>phase = sin(post[nFft/2+1:])</c>,
    /// then <c>iSTFT(magnitude·e^{j·phase})</c>. <paramref name="post"/> is channels-first
    /// <c>[1, n_fft+2, frames]</c>; returns the time-domain waveform.</summary>
    public static float[] IstftHead(Tensor post, int nFft, int hop)
    {
        int numBins = nFft / 2 + 1;
        int frames = (int)post.Shape[2];
        float* pp = (float*)post.DataPointer;
        int count = frames * numBins;
        float[] real = ArrayPool<float>.Shared.Rent(count);
        float[] imag = ArrayPool<float>.Shared.Rent(count);
        try
        {
            int framesPerBlock = FramePartition.FramesPerBlock(nFft, FramePartition.TransformBudget);
            int blocks = FramePartition.BlockCount(frames, framesPerBlock);
            CpuParallel.For(blocks, 8L * count, b =>
            {
                int end = Math.Min(frames, (b + 1) * framesPerBlock);
                for (int f = b * framesPerBlock; f < end; f++)
                {
                    for (int bin = 0; bin < numBins; bin++)
                    {
                        float mag = MathF.Min(MathF.Exp(pp[(long)bin * frames + f]), 1e2f);   // torch _istft clips magnitude to 1e2
                        float ang = MathF.Sin(pp[(long)(numBins + bin) * frames + f]);
                        real[f * numBins + bin] = mag * MathF.Cos(ang);
                        imag[f * numBins + bin] = mag * MathF.Sin(ang);
                    }
                }
            });
            return IStft.Apply(real, imag, frames, nFft, hop);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(real);
            ArrayPool<float>.Shared.Return(imag);
        }
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

using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using Xunit;
using static HartsyInference.Tests.Common.CpuSchedules;

namespace HartsyInference.Audio.Tests;

/// <summary>The NSF vocoder DSP fans its frames out through <see cref="CpuParallel"/> over a partition fixed by the
/// input size. These pin every piece to the single-threaded loop it replaced, bit for bit, and pin the output to be
/// the same whatever the schedule: the default fan-out, a <c>numerics.cpuThreads</c> cap of 1, and
/// <see cref="CpuParallel.EnterInline"/> (what a real-time audio thread runs under). A difference here would mean
/// a machine's core count or a host's knob changes the audio — or that streaming the harmonic source in chunks no
/// longer reproduces one monolithic call, which CosyVoice's streaming vocoder depends on.</summary>
public sealed unsafe class NsfVocoderDspTests
{
    private const int Scale = 120, Harmonics = 9, SampleRate = 24_000;

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(7)]
    [InlineData(1000)]
    [InlineData(123_457)]
    public void Advance_MatchesSequentialWalk(int steps)
    {
        uint walked = 0x9E3779B9u;
        for (int i = 0; i < steps; i++) DeterministicRng.NextUniform(ref walked);
        Assert.Equal(walked, DeterministicRng.Advance(0x9E3779B9u, steps));
        Assert.Equal(0x9E3779B9u, DeterministicRng.Advance(0x9E3779B9u, 0));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void HarmonicSource_BitIdenticalToSequentialLoop(bool addNoise)
    {
        float[] f0 = VoicedF0(700, seed: 7);
        (Tensor mergeW, Tensor mergeB) = MergeWeights();
        try
        {
            double[] phaseRef = new double[Harmonics];
            uint rngRef = 0x9E3779B9u;
            float[] expected = SequentialHarmonicSource(f0, phaseRef, ref rngRef, mergeW.AsSpan<float>().ToArray(),
                mergeB.AsSpan<float>()[0], addNoise);

            double[] phase = new double[Harmonics];
            uint rng = 0x9E3779B9u;
            float[] actual = NsfVocoderDsp.GenerateHarmonicSourceChunk(f0, phase, ref rng, Scale, SampleRate, Harmonics,
                mergeW, mergeB, 0.1f, 0.003f, 10f, addNoise);

            AssertBitIdentical(expected, actual);
            Assert.Equal(rngRef, rng);
            for (int h = 0; h < Harmonics; h++) Assert.Equal(phaseRef[h], phase[h]);
        }
        finally
        {
            mergeW.Dispose();
            mergeB.Dispose();
        }
    }

    [Fact]
    public void HarmonicSource_StreamedChunks_MatchOneMonolithicCall()
    {
        float[] f0 = VoicedF0(700, seed: 11);
        (Tensor mergeW, Tensor mergeB) = MergeWeights();
        try
        {
            double[] phaseWhole = new double[Harmonics];
            uint rngWhole = 0x9E3779B9u;
            float[] whole = NsfVocoderDsp.GenerateHarmonicSourceChunk(f0, phaseWhole, ref rngWhole, Scale, SampleRate,
                Harmonics, mergeW, mergeB, 0.1f, 0.003f, 10f, addNoise: true);

            // Uneven chunks: a single frame (the serial branch) between two multi-block ones.
            double[] phase = new double[Harmonics];
            uint rng = 0x9E3779B9u;
            List<float> streamed = [];
            foreach ((int start, int length) in new[] { (0, 250), (250, 1), (251, 449) })
            {
                streamed.AddRange(NsfVocoderDsp.GenerateHarmonicSourceChunk(f0[start..(start + length)], phase, ref rng, Scale,
                    SampleRate, Harmonics, mergeW, mergeB, 0.1f, 0.003f, 10f, addNoise: true));
            }

            AssertBitIdentical(whole, streamed.ToArray());
            Assert.Equal(rngWhole, rng);
            for (int h = 0; h < Harmonics; h++) Assert.Equal(phaseWhole[h], phase[h]);
        }
        finally
        {
            mergeW.Dispose();
            mergeB.Dispose();
        }
    }

    [Fact]
    public void HarmonicSource_SameBytesUnderEverySchedule()
    {
        float[] f0 = VoicedF0(700, seed: 13);
        (Tensor mergeW, Tensor mergeB) = MergeWeights();
        try
        {
            (float[] parallel, float[] capped, float[] inline) = UnderEverySchedule(() =>
            {
                double[] phase = new double[Harmonics];
                uint rng = 0x9E3779B9u;
                return NsfVocoderDsp.GenerateHarmonicSourceChunk(f0, phase, ref rng, Scale, SampleRate, Harmonics, mergeW,
                    mergeB, 0.1f, 0.003f, 10f, addNoise: true);
            });
            AssertBitIdentical(parallel, capped);
            AssertBitIdentical(parallel, inline);
        }
        finally
        {
            mergeW.Dispose();
            mergeB.Dispose();
        }
    }

    [Theory]
    [InlineData(20, 5, true)]
    [InlineData(20, 5, false)]
    [InlineData(16, 4, false)]
    public void ForwardStft_SameBytesUnderEverySchedule(int nFft, int hop, bool magPhase)
    {
        float[] signal = Noise(40_000, seed: nFft + hop);
        (float[] parallel, float[] capped, float[] inline) = UnderEverySchedule(() =>
        {
            using Tensor spec = magPhase ? NsfVocoderDsp.ForwardStftMagPhase(signal, nFft, hop)
                : NsfVocoderDsp.ForwardStftRealImag(signal, nFft, hop);
            return spec.AsSpan<float>().ToArray();
        });
        AssertBitIdentical(parallel, capped);
        AssertBitIdentical(parallel, inline);
    }

    [Fact]
    public void IstftHead_SameBytesUnderEverySchedule()
    {
        const int nFft = 20, hop = 5, frames = 8000;
        using Tensor post = new(new TensorShape(1, nFft + 2, frames), DType.F32);
        Noise(post.AsSpan<float>().Length, seed: 29).CopyTo(post.AsSpan<float>());
        (float[] parallel, float[] capped, float[] inline) = UnderEverySchedule(() => NsfVocoderDsp.IstftHead(post, nFft, hop));
        AssertBitIdentical(parallel, capped);
        AssertBitIdentical(parallel, inline);
    }

    [Theory]
    [InlineData(20, 5, 6000)]
    [InlineData(1024, 256, 300)]
    public void IStftApply_BitIdenticalToSequentialLoop_UnderEverySchedule(int nFft, int hop, int frames)
    {
        int numBins = nFft / 2 + 1;
        float[] re = Noise(frames * numBins, seed: 31);
        float[] im = Noise(frames * numBins, seed: 37);
        float[] expected = SequentialIStft(re, im, frames, nFft, hop);
        (float[] parallel, float[] capped, float[] inline) = UnderEverySchedule(() => IStft.Apply(re, im, frames, nFft, hop));
        AssertBitIdentical(expected, parallel);
        AssertBitIdentical(expected, capped);
        AssertBitIdentical(expected, inline);
    }

    /// <summary>The harmonic source as it was before the fan-out, verbatim: one walk over every sample.</summary>
    private static float[] SequentialHarmonicSource(float[] f0, double[] phase, ref uint rng, float[] mW, float mB, bool addNoise)
    {
        const float sineAmp = 0.1f, noiseStd = 0.003f, voicedThreshold = 10f;
        float[] merged = new float[f0.Length * Scale];
        for (int i = 0; i < f0.Length; i++)
        {
            float hz = f0[i];
            float uv = hz > voicedThreshold ? 1f : 0f;
            float noiseAmp = uv * noiseStd + (1f - uv) * (sineAmp / 3f);
            for (int rep = 0; rep < Scale; rep++)
            {
                float lin = mB;
                for (int h = 0; h < Harmonics; h++)
                {
                    phase[h] += (double)hz * (h + 1) / SampleRate;
                    phase[h] -= Math.Floor(phase[h]);
                    float sine = (float)Math.Sin(2.0 * Math.PI * phase[h]) * sineAmp;
                    float noise = addNoise ? noiseAmp * DeterministicRng.NextGaussian(ref rng) : 0f;
                    lin += mW[h] * (sine * uv + noise);
                }
                merged[i * Scale + rep] = MathF.Tanh(lin);
            }
        }
        return merged;
    }

    /// <summary><see cref="IStft.Apply"/> as it was before the fan-out, verbatim: transform and overlap-add frame by frame.</summary>
    private static float[] SequentialIStft(float[] spectReal, float[] spectImag, int frames, int nFft, int hopLength)
    {
        int half = nFft / 2;
        int numBins = half + 1;
        float[] window = HannWindow.Get(nFft);
        long rawLen = (long)(frames - 1) * hopLength + nFft;
        float[] outRaw = new float[rawLen];
        float[] winSq = new float[rawLen];
        float[] frameRe = new float[nFft];
        float[] frameIm = new float[nFft];
        for (int f = 0; f < frames; f++)
        {
            int rowOff = f * numBins;
            for (int k = 0; k < numBins; k++)
            {
                frameRe[k] = spectReal[rowOff + k];
                frameIm[k] = spectImag[rowOff + k];
            }
            for (int k = 1; k < half; k++)
            {
                frameRe[nFft - k] = spectReal[rowOff + k];
                frameIm[nFft - k] = -spectImag[rowOff + k];
            }
            for (int i = 0; i < nFft; i++) frameIm[i] = -frameIm[i];
            Fft.Transform(frameRe, frameIm, nFft);
            float invN = 1f / nFft;
            for (int i = 0; i < nFft; i++) frameRe[i] *= invN;
            long start = (long)f * hopLength;
            for (int i = 0; i < nFft; i++)
            {
                outRaw[start + i] += frameRe[i] * window[i];
                winSq[start + i] += window[i] * window[i];
            }
        }
        for (long i = 0; i < rawLen; i++)
        {
            if (winSq[i] > 1e-11f) outRaw[i] /= winSq[i];
        }
        long trimmedLen = Math.Max(0, rawLen - 2L * half);
        float[] result = new float[trimmedLen];
        Array.Copy(outRaw, half, result, 0, trimmedLen);
        return result;
    }

    /// <summary>Voiced runs with unvoiced gaps, so both branches of the amplitude shaping run.</summary>
    private static float[] VoicedF0(int frames, int seed)
    {
        Random rng = new(seed);
        float[] f0 = new float[frames];
        for (int i = 0; i < frames; i++) f0[i] = (i / 50) % 3 == 2 ? 0f : 90f + 200f * (float)rng.NextDouble();
        return f0;
    }

    private static (Tensor MergeW, Tensor MergeB) MergeWeights()
    {
        Tensor mergeW = new(new TensorShape(1, Harmonics), DType.F32);
        Tensor mergeB = new(new TensorShape(1), DType.F32);
        Span<float> w = mergeW.AsSpan<float>();
        for (int h = 0; h < Harmonics; h++) w[h] = 0.3f - 0.05f * h;
        mergeB.AsSpan<float>()[0] = 0.01f;
        return (mergeW, mergeB);
    }

    private static float[] Noise(int length, int seed)
    {
        Random rng = new(seed);
        float[] values = new float[length];
        for (int i = 0; i < length; i++) values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }

    private static void AssertBitIdentical(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
            {
                Assert.Fail($"element {i}: {expected[i]:R} vs {actual[i]:R}");
            }
        }
    }
}

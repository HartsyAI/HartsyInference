using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Cpu;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Two 10 ms frames denoised together, layer by layer, must give exactly what two frames one at a time did.
/// <see cref="RnnoiseStream"/> pairs two whole frames when one call holds them, and
/// <see cref="RnnoiseModel.ProcessPair"/> then reads the weights shared by the pair once. That is only a memory-traffic
/// change if every output sample and every speech probability keeps its bits. This compares them byte for byte on real
/// speech in noise, with digital silence spliced in at lengths that put the silence edges on both frames of a pair, so
/// every branch runs: both frames paired, and either one silent while the other runs alone.</summary>
public sealed class RnnoisePairTests(ITestOutputHelper log)
{
    private const float Int16Scale = 32768f;

    [Theory]
    [Trait("Category", "Integration")]
    [InlineData(16_000)]
    [InlineData(48_000)]
    public void PairedFrames_MatchOneFrameAtATime_BitForBit(int rate)
    {
        string weightsPath = RnnoiseRealSpeechTests.WeightsPath();
        string clipPath = Path.Combine(RepoRoot.Path, "tests", "python-reference", "silerovad_reference", "jfk.wav");
        if (!RealWeightGate.Require(log.WriteLine, weightsPath, clipPath)) return;

        WavFile.DecodedAudio clip = WavFile.Read(clipPath);
        float[] speech = clip.ToMono();
        if (clip.SampleRate != rate) speech = Resampler.Create(clip.SampleRate, rate).Resample(speech);
        float[] audio = WithNoiseAndGaps(speech, rate);

        using RnnoiseWeights weights = RnnoiseRealSpeechTests.LoadWeights(weightsPath);
        using CpuBackend backend = new();
        using RnnoiseStream single = new(weights, rate);
        using RnnoiseStream paired = new(weights, rate);
        int frame = single.FrameSize;
        int frames = audio.Length / frame;

        // One frame per call never pairs; two frames per call always do.
        float[] singleOut = new float[frames * frame];
        float[] pairedOut = new float[frames * frame];
        float[] buffer = new float[3 * frame];
        int singleWritten = 0, pairedWritten = 0, probabilitiesCompared = 0;
        for (int f = 0; f + 1 < frames; f += 2)
        {
            for (int k = 0; k < 2; k++)
            {
                int n = single.Process(backend, audio.AsSpan((f + k) * frame, frame), buffer);
                buffer.AsSpan(0, n).CopyTo(singleOut.AsSpan(singleWritten));
                singleWritten += n;
            }
            int m = paired.Process(backend, audio.AsSpan(f * frame, 2 * frame), buffer);
            buffer.AsSpan(0, m).CopyTo(pairedOut.AsSpan(pairedWritten));
            pairedWritten += m;
            Assert.Equal(BitConverter.SingleToUInt32Bits(single.SpeechProbability),
                BitConverter.SingleToUInt32Bits(paired.SpeechProbability));
            probabilitiesCompared++;
        }

        Assert.Equal(singleWritten, pairedWritten);
        int firstMismatch = FirstMismatch(singleOut.AsSpan(0, singleWritten), pairedOut.AsSpan(0, pairedWritten));
        RnnoiseDenoiser denoiser = paired.Denoiser;
        log.WriteLine($"{rate} Hz: {singleWritten} samples and {probabilitiesCompared} speech probabilities compared; "
            + $"pairs with the first frame silent {denoiser.PairsWithFirstSilent}, with the second silent "
            + $"{denoiser.PairsWithSecondSilent}");
        Assert.True(firstMismatch < 0, $"{rate} Hz: sample {firstMismatch} differs between paired and single frames");
        Assert.True(denoiser.PairsWithFirstSilent > 0, "no pair had only its first frame silent");
        Assert.True(denoiser.PairsWithSecondSilent > 0, "no pair had only its second frame silent");
    }

    /// <summary>Speech at int16 scale with white noise about 30 dB under it, interrupted by stretches of digital
    /// silence of 100 to 190 ms. The high-pass filter rings for several frames after a loud one, so the first frame
    /// RNNoise reads as silent varies with the audio before it; ten gap lengths put the edges on both frames of a pair.</summary>
    private static float[] WithNoiseAndGaps(float[] speech, int rate)
    {
        Random rng = new(7);
        int frameSamples = rate / 100;
        List<float> audio = [];
        int position = 0;
        int[] gapFrames = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
        int segment = speech.Length / (gapFrames.Length + 1);
        foreach (int gap in gapFrames)
        {
            for (int i = 0; i < segment; i++)
                audio.Add((speech[position + i] + (float)((rng.NextDouble() * 2 - 1) * 0.01)) * Int16Scale);
            position += segment;
            for (int i = 0; i < gap * frameSamples; i++) audio.Add(0f);
        }
        for (int i = position; i < speech.Length; i++)
            audio.Add((speech[i] + (float)((rng.NextDouble() * 2 - 1) * 0.01)) * Int16Scale);
        return [.. audio];
    }

    private static int FirstMismatch(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        for (int i = 0; i < a.Length; i++)
        {
            if (BitConverter.SingleToUInt32Bits(a[i]) != BitConverter.SingleToUInt32Bits(b[i])) return i;
        }
        return -1;
    }
}

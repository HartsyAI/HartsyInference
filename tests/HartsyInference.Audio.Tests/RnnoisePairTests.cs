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
/// change if every output sample and every speech probability keeps its bits.
///
/// <para>Each case runs a reference stream one frame per call, which never pairs, against a second stream fed in
/// chunks (<see cref="Chunking"/>), and compares every output sample bit for bit. After every chunked call, the
/// second stream's speech probability must also match the reference's after the same number of frames. Stretches of
/// sound alternate with digital silence, spaced so the silence edges fall on both frames of a pair. That covers the
/// branches where one frame is silent and the other runs alone, and the test asserts both ran.</para>
///
/// <para>The synthetic-weight cases are in the unit lane, so the comparison runs without the checkpoint. The
/// real-weight cases repeat it with the shipped model on real speech. Both run at each
/// <see cref="RnnoisePrecision"/>: at int8 the paired path quantizes two rows and runs one product over them, and
/// each row must still get its own frame's bits.</para></summary>
public sealed class RnnoisePairTests(ITestOutputHelper log)
{
    private const float Int16Scale = 32768f;

    /// <summary>Digital silence between stretches of sound, in 10 ms frames. The chunk patterns group frames into
    /// pairs in cycles of 2, 3 and 5 frames. A stretch of sound plus its gap is 31 frames for noise and 91 for speech,
    /// and neither shares a factor with those cycles. So successive stretches start at every position in each cycle,
    /// and their ends move round it too.</summary>
    private const int GapFrames = 19;

    private const int NoiseFrames = 12;
    private const int SpeechFrames = 72;

    /// <summary>How the chunked stream is fed.</summary>
    public enum Chunking
    {
        /// <summary>Two whole frames per call, so every call pairs, as the voice front end's 20 ms frames do.</summary>
        TwoFrames,

        /// <summary>A frame and a half per call. Every other call completes the half frame the previous one left
        /// pending and pairs it with the next whole frame; the calls between run one frame alone.</summary>
        FrameAndAHalf,

        /// <summary>Two and a half frames per call. Within single calls this pairs two fresh frames, completes a
        /// pending frame and pairs it, and runs a lone frame.</summary>
        TwoAndAHalfFrames,

        /// <summary>Seeded random sizes from one sample to three frames.</summary>
        RandomSizes,
    }

    public static TheoryData<int, Chunking, RnnoisePrecision> Cases()
    {
        TheoryData<int, Chunking, RnnoisePrecision> data = new();
        foreach (RnnoisePrecision precision in Enum.GetValues<RnnoisePrecision>())
            foreach (int rate in new[] { 16_000, 48_000 })
                foreach (Chunking chunking in Enum.GetValues<Chunking>())
                    data.Add(rate, chunking, precision);
        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void SyntheticWeights_ChunkedStreamMatchesOneFrameAtATime_BitForBit(int rate, Chunking chunking,
        RnnoisePrecision precision)
    {
        using RnnoiseWeights weights = new();
        VoiceFrontendAllocationTests.Load(VoiceFrontendAllocationTests.RnnoiseLayout, seed: 1, weights.Load);
        if (precision == RnnoisePrecision.Int8) VoiceFrontendAllocationTests.LoadInt8Tables(weights, seed: 11);
        AssertChunkedMatchesSingle(weights, NoiseBurstsWithGaps(rate), rate, chunking);
    }

    [Theory]
    [Trait("Category", "Integration")]
    [MemberData(nameof(Cases))]
    public void RealWeights_ChunkedStreamMatchesOneFrameAtATime_BitForBit(int rate, Chunking chunking,
        RnnoisePrecision precision)
    {
        string weightsPath = RnnoiseRealSpeechTests.WeightsPath();
        string clipPath = Path.Combine(RepoRoot.Path, "tests", "python-reference", "silerovad_reference", "jfk.wav");
        string[] required = precision == RnnoisePrecision.Int8
            ? [weightsPath, RnnoiseRealSpeechTests.Int8TablesPath(), clipPath]
            : [weightsPath, clipPath];
        if (!RealWeightGate.Require(log.WriteLine, required)) return;

        WavFile.DecodedAudio clip = WavFile.Read(clipPath);
        float[] speech = clip.ToMono();
        if (clip.SampleRate != rate) speech = Resampler.Create(clip.SampleRate, rate).Resample(speech);
        using RnnoiseWeights weights = RnnoiseRealSpeechTests.LoadWeights(weightsPath, precision);
        AssertChunkedMatchesSingle(weights, SpeechWithNoiseAndGaps(speech, rate), rate, chunking);
    }

    private void AssertChunkedMatchesSingle(RnnoiseWeights weights, float[] audio, int rate, Chunking chunking)
    {
        using CpuBackend backend = new();
        using RnnoiseStream single = new(weights, rate);
        using RnnoiseStream chunked = new(weights, rate);
        int frame = single.FrameSize;
        int frames = audio.Length / frame;
        int total = frames * frame;
        float[] buffer = new float[4 * frame];

        // The reference: one frame per call never pairs. Its speech probability is kept after every frame, so the
        // chunked stream can be checked wherever a call leaves it.
        float[] expected = new float[total];
        uint[] probabilityAfter = new uint[frames + 1];
        probabilityAfter[0] = BitConverter.SingleToUInt32Bits(single.SpeechProbability);
        int expectedWritten = 0;
        for (int f = 0; f < frames; f++)
        {
            int n = single.Process(backend, audio.AsSpan(f * frame, frame), buffer);
            buffer.AsSpan(0, n).CopyTo(expected.AsSpan(expectedWritten));
            expectedWritten += n;
            probabilityAfter[f + 1] = BitConverter.SingleToUInt32Bits(single.SpeechProbability);
        }
        Assert.Equal(total, expectedWritten);

        float[] actual = new float[total];
        int fed = 0, written = 0, calls = 0;
        foreach (int size in ChunkSizes(chunking, frame, total))
        {
            int n = chunked.Process(backend, audio.AsSpan(fed, size), buffer);
            buffer.AsSpan(0, n).CopyTo(actual.AsSpan(written));
            fed += size;
            written += n;
            calls++;
            Assert.Equal(0, written % frame);
            Assert.True(probabilityAfter[written / frame] == BitConverter.SingleToUInt32Bits(chunked.SpeechProbability),
                $"{rate} Hz {chunking}: speech probability differs after call {calls} ({written / frame} frames)");
        }
        Assert.Equal(total, written);

        RnnoiseDenoiser denoiser = chunked.Denoiser;
        log.WriteLine($"{rate} Hz {chunking}: {total} samples in {calls} calls; pairs with the first frame silent "
            + $"{denoiser.PairsWithFirstSilent}, with the second silent {denoiser.PairsWithSecondSilent}");
        int mismatch = FirstMismatch(expected, actual);
        Assert.True(mismatch < 0, $"{rate} Hz {chunking}: sample {mismatch} differs from one frame at a time");
        Assert.True(denoiser.PairsWithFirstSilent > 0, $"{rate} Hz {chunking}: no pair had only its first frame silent");
        Assert.True(denoiser.PairsWithSecondSilent > 0, $"{rate} Hz {chunking}: no pair had only its second frame silent");
    }

    /// <summary>Sizes of successive calls, in source samples, ending exactly at <paramref name="total"/>.</summary>
    private static List<int> ChunkSizes(Chunking chunking, int frame, int total)
    {
        Random rng = new(11);
        List<int> sizes = [];
        for (int fed = 0; fed < total;)
        {
            int size = chunking switch
            {
                Chunking.TwoFrames => 2 * frame,
                Chunking.FrameAndAHalf => frame + frame / 2,
                Chunking.TwoAndAHalfFrames => 2 * frame + frame / 2,
                _ => rng.Next(1, 3 * frame + 1),
            };
            size = Math.Min(size, total - fed);
            sizes.Add(size);
            fed += size;
        }
        return sizes;
    }

    /// <summary>Eleven 120 ms bursts of white noise at int16 scale, separated by the gaps. Whether a frame is silent
    /// depends only on its own energy, not on the weights, so synthetic weights cover the same branches as real
    /// ones.</summary>
    private static float[] NoiseBurstsWithGaps(int rate)
    {
        Random rng = new(5);
        int frame = rate / 100;
        List<float> audio = [];
        for (int burst = 0; burst < 11; burst++)
        {
            if (burst > 0) audio.AddRange(new float[GapFrames * frame]);
            for (int i = 0; i < NoiseFrames * frame; i++) audio.Add((float)((rng.NextDouble() * 2 - 1) * 3000));
        }
        return [.. audio];
    }

    /// <summary>Speech at int16 scale with white noise about 30 dB under it, cut into 720 ms stretches separated by the
    /// gaps. The noise keeps every frame of a stretch above the silence threshold, so only the gaps are silent. The
    /// high-pass filter rings for several frames after a loud one, so the first frame RNNoise reads as silent varies
    /// with the audio before it.</summary>
    private static float[] SpeechWithNoiseAndGaps(float[] speech, int rate)
    {
        Random rng = new(7);
        int frame = rate / 100;
        int stretch = SpeechFrames * frame;
        List<float> audio = [];
        for (int position = 0; position < speech.Length; position += stretch)
        {
            if (position > 0) audio.AddRange(new float[GapFrames * frame]);
            int end = Math.Min(position + stretch, speech.Length);
            for (int i = position; i < end; i++)
                audio.Add((speech[i] + (float)((rng.NextDouble() * 2 - 1) * 0.01)) * Int16Scale);
        }
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

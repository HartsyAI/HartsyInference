using HartsyInference.Audio.Dsp.Windowing;
using Xunit;

namespace HartsyInference.Audio.Tests.Dsp.Windowing;

/// <summary>Cut placement at 16 kHz, where a 20 ms frame is 320 samples.</summary>
public sealed class SpeechCutFinderTests
{
    private const int SampleRate = 16_000;

    [Fact]
    public void Find_CutsInsidePlantedSilence()
    {
        float[] band = Concat(Tone(6_400, 0.5f), Silence(3_200), Tone(6_400, 0.5f));

        SpeechCut cut = SpeechCutFinder.Find(band, SampleRate);

        Assert.False(cut.HardCut);
        Assert.InRange(cut.Offset, 6_400, 9_600);
    }

    [Fact]
    public void Find_LoudBandWithNoQuietRun_IsAHardCut()
    {
        float[] band = Tone(16_000, 0.5f);

        SpeechCut cut = SpeechCutFinder.Find(band, SampleRate);

        Assert.True(cut.HardCut);
        Assert.InRange(cut.Offset, 0, band.Length - 1);
    }

    [Fact]
    public void Find_AllSilentBand_IsQuietThroughoutNotSpeech()
    {
        // -80 dBFS noise is below the silence floor, so every frame is quiet and the cut lands mid-band.
        float[] band = Noise(16_000, 1e-4f);

        SpeechCut cut = SpeechCutFinder.Find(band, SampleRate);

        Assert.False(cut.HardCut);
        Assert.InRange(cut.Offset, 7_000, 9_000);
    }

    [Fact]
    public void Find_SingleQuietFrame_CutsInItsMiddleNotAtItsEdge()
    {
        // Ten loud frames, one silent frame (320 samples), ten loud frames: the silent frame spans 3200-3520.
        float[] band = Concat(Tone(3_200, 0.5f), Silence(320), Tone(3_200, 0.5f));

        SpeechCut cut = SpeechCutFinder.Find(band, SampleRate);

        Assert.False(cut.HardCut);
        Assert.Equal(3_360, cut.Offset);
    }

    [Fact]
    public void Find_TrailingPartialFrame_IsNotAQuietRun()
    {
        // Ten silent samples do not fill a frame, so they must not read as a clean cut inside loud audio.
        float[] band = Concat(Tone(16_000, 0.5f), Silence(10));

        SpeechCut cut = SpeechCutFinder.Find(band, SampleRate);

        Assert.True(cut.HardCut);
    }

    [Fact]
    public void Find_SampleRateBelowTheFloor_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => SpeechCutFinder.Find(new float[100], 40));
    }

    [Fact]
    public void Find_EmptyBand_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => SpeechCutFinder.Find(ReadOnlySpan<float>.Empty, SampleRate));
    }

    private static float[] Tone(int length, float amplitude)
    {
        float[] samples = new float[length];
        for (int i = 0; i < length; i++)
        {
            samples[i] = amplitude * (float)Math.Sin(2 * Math.PI * 440 * i / SampleRate);
        }
        return samples;
    }

    private static float[] Silence(int length) => new float[length];

    private static float[] Noise(int length, float amplitude)
    {
        Random random = new(7);
        float[] samples = new float[length];
        for (int i = 0; i < length; i++)
        {
            samples[i] = amplitude * (float)(random.NextDouble() * 2 - 1);
        }
        return samples;
    }

    private static float[] Concat(params float[][] parts)
    {
        float[] result = new float[parts.Sum(p => p.Length)];
        int at = 0;
        foreach (float[] part in parts)
        {
            part.CopyTo(result, at);
            at += part.Length;
        }
        return result;
    }
}

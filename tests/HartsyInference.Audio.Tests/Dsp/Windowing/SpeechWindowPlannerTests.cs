using HartsyInference.Audio.Dsp.Windowing;
using Xunit;

namespace HartsyInference.Audio.Tests.Dsp.Windowing;

/// <summary>Window plans at 16 kHz and 24 kHz; windows are contiguous and every window is at or under the limit.</summary>
public sealed class SpeechWindowPlannerTests
{
    [Theory]
    [InlineData(16_000)]
    [InlineData(24_000)]
    public void Plan_CoversEverySampleOnceWithinTheLimit(int sampleRate)
    {
        int limit = 30 * sampleRate;
        float[] clip = Tone(sampleRate * 91, sampleRate, 0.5f);

        IReadOnlyList<SpeechWindow> windows = SpeechWindowPlanner.Plan(clip, sampleRate, limit);

        int expectedStart = 0;
        foreach (SpeechWindow window in windows)
        {
            Assert.Equal(expectedStart, window.Start);
            Assert.InRange(window.Length, sampleRate, limit);
            expectedStart += window.Length;
        }
        Assert.Equal(clip.Length, expectedStart);
    }

    [Fact]
    public void Plan_ClipJustOverTheLimit_LeavesNoSubSecondTail()
    {
        int sampleRate = 16_000;
        int limit = 30 * sampleRate;
        float[] clip = Tone(limit + sampleRate / 10, sampleRate, 0.5f);

        IReadOnlyList<SpeechWindow> windows = SpeechWindowPlanner.Plan(clip, sampleRate, limit);

        Assert.Equal(2, windows.Count);
        Assert.True(windows[^1].Length >= sampleRate, "the last window must hold at least one second");
    }

    [Fact]
    public void Plan_ClipAtTheLimit_IsOneWindow()
    {
        int sampleRate = 16_000;
        float[] clip = Tone(30 * sampleRate, sampleRate, 0.5f);

        IReadOnlyList<SpeechWindow> windows = SpeechWindowPlanner.Plan(clip, sampleRate, 30 * sampleRate);

        SpeechWindow only = Assert.Single(windows);
        Assert.Equal(clip.Length, only.Length);
        Assert.False(only.HardCut);
    }

    [Fact]
    public void Plan_CutsInsideASilenceNearTheLimit()
    {
        int sampleRate = 16_000;
        int limit = 30 * sampleRate;
        // Speech for 26 s, silence from 26 s to 29 s, speech again to 40 s.
        float[] clip = Concat(Tone(26 * sampleRate, sampleRate, 0.5f), Silence(3 * sampleRate), Tone(11 * sampleRate, sampleRate, 0.5f));

        IReadOnlyList<SpeechWindow> windows = SpeechWindowPlanner.Plan(clip, sampleRate, limit);

        SpeechWindow first = windows[0];
        Assert.False(first.HardCut);
        Assert.InRange(first.Start + first.Length, 26 * sampleRate, 29 * sampleRate);
    }

    [Fact]
    public void Plan_SilentTail_IsFlaggedSilent()
    {
        int sampleRate = 16_000;
        // Speech for 20 s, then silence for 20 s; the 30 s window searches 25-30 s, all silent, and cuts at its centre.
        float[] clip = Concat(Tone(20 * sampleRate, sampleRate, 0.5f), Silence(20 * sampleRate));

        IReadOnlyList<SpeechWindow> windows = SpeechWindowPlanner.Plan(clip, sampleRate, 30 * sampleRate);

        Assert.False(windows[0].Silent);
        Assert.InRange(windows[0].Length, 25 * sampleRate, 30 * sampleRate);
        Assert.True(windows[^1].Silent);
    }

    [Fact]
    public void Plan_ContinuousSpeechPastTheLimit_EndsTheWindowOnAHardCut()
    {
        int sampleRate = 16_000;
        float[] clip = Tone(61 * sampleRate, sampleRate, 0.5f);

        IReadOnlyList<SpeechWindow> windows = SpeechWindowPlanner.Plan(clip, sampleRate, 30 * sampleRate);

        Assert.True(windows[0].HardCut);
        Assert.False(windows[^1].HardCut);
    }

    [Fact]
    public void Plan_MinimumWindow_LeavesAtLeastOneSecondBeforeAndAfterTheCut()
    {
        int sampleRate = 16_000;
        // A silent head would otherwise pull the cut to the first few samples of the window.
        float[] clip = Concat(Silence(2_000), Tone(2 * sampleRate + 1 - 2_000, sampleRate, 0.5f));

        IReadOnlyList<SpeechWindow> windows = SpeechWindowPlanner.Plan(clip, sampleRate, 2 * sampleRate);

        Assert.Equal(2, windows.Count);
        Assert.All(windows, window => Assert.True(window.Length >= sampleRate, "every window must hold at least one second"));
    }

    [Fact]
    public void Plan_QuietButAudibleSpeech_IsNotFlaggedSilent()
    {
        int sampleRate = 16_000;
        // Amplitude 0.01 is about -43 dBFS RMS, above the -55 dBFS silence floor.
        float[] clip = Tone(5 * sampleRate, sampleRate, 0.01f);

        SpeechWindow only = Assert.Single(SpeechWindowPlanner.Plan(clip, sampleRate, 30 * sampleRate));

        Assert.False(only.Silent);
    }

    [Fact]
    public void Plan_EmptyAudio_IsEmpty()
    {
        Assert.Empty(SpeechWindowPlanner.Plan(ReadOnlySpan<float>.Empty, 16_000, 30 * 16_000));
    }

    [Fact]
    public void Plan_SampleRateBelowTheFloor_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => SpeechWindowPlanner.Plan(new float[1_000], 40, 2_000));
    }

    [Fact]
    public void Plan_WindowShorterThanTwoSeconds_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => SpeechWindowPlanner.Plan(new float[16_000 * 10], 16_000, 16_000));
    }

    private static float[] Tone(int length, int sampleRate, float amplitude)
    {
        float[] samples = new float[length];
        for (int i = 0; i < length; i++)
        {
            samples[i] = amplitude * (float)Math.Sin(2 * Math.PI * 440 * i / sampleRate);
        }
        return samples;
    }

    private static float[] Silence(int length) => new float[length];

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

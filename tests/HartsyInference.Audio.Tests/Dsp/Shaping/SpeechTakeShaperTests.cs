using HartsyInference.Audio.Dsp.Shaping;
using Xunit;

namespace HartsyInference.Audio.Tests.Dsp.Shaping;

/// <summary>Hand-computed cases for <see cref="SpeechTakeShaper"/> at 16 kHz, where a 20 ms frame is 320 samples.</summary>
public sealed class SpeechTakeShaperTests
{
    private const int SampleRate = 16_000;

    [Fact]
    public void Trim_SilenceOnly_ReturnsACopyOfTheInput()
    {
        float[] silence = new float[SampleRate];
        float[] trimmed = SpeechTakeShaper.Trim(silence, SampleRate);
        Assert.Equal(silence.Length, trimmed.Length);
        Assert.All(trimmed, v => Assert.Equal(0f, v));
        Assert.NotSame(silence, trimmed);
    }

    [Fact]
    public void Trim_EmptyInput_ReturnsEmpty()
    {
        Assert.Empty(SpeechTakeShaper.Trim(Array.Empty<float>(), SampleRate));
    }

    [Fact]
    public void Trim_OneFrame_KeepsTheWholeFrame()
    {
        float[] one = Tone(320, 0.5f);
        float[] trimmed = SpeechTakeShaper.Trim(one, SampleRate);
        Assert.Equal(one, trimmed);
    }

    [Fact]
    public void Trim_LeadingAndTrailingSilence_KeepsTwoFramesBeforeAndThreeAfter()
    {
        // Speech occupies frames 25..49, so the output runs from frame 23 (sample 7360) to sample 16640.
        float[] take = Concat(Silence(8_000), Tone(8_000, 0.5f), Silence(8_000));
        float[] trimmed = SpeechTakeShaper.Trim(take, SampleRate);
        Assert.Equal(9_280, trimmed.Length);
        Assert.Equal(take[7_360..16_640], trimmed);
    }

    [Fact]
    public void Squeeze_ShortInternalPause_IsKept()
    {
        float[] take = Concat(Tone(8_000, 0.5f), Silence(4_800), Tone(8_000, 0.5f));
        float[] squeezed = SpeechTakeShaper.Squeeze(take, SampleRate);
        Assert.Equal(20_800, squeezed.Length);
    }

    [Fact]
    public void Squeeze_OneSecondPause_IsReducedToThirtyFiveHundredths()
    {
        // The 1.0 s run is 50 frames, above 0.5 s; 0.175 s is kept at each end, 0.35 s in total.
        float[] take = Concat(Tone(8_000, 0.5f), Silence(16_000), Tone(8_000, 0.5f));
        float[] squeezed = SpeechTakeShaper.Squeeze(take, SampleRate);
        Assert.Equal(8_000 + 5_600 + 8_000, squeezed.Length);
        Assert.Equal(take[..8_000], squeezed[..8_000]);
        Assert.Equal(take[^8_000..], squeezed[^8_000..]);
    }

    [Fact]
    public void Level_SineTake_HitsTargetRmsWithinTenthOfADecibel()
    {
        float[] sine = Tone(SampleRate, 0.5f);
        float[] leveled = SpeechTakeShaper.Level(sine, SampleRate);
        double rmsDb = 20.0 * Math.Log10(Rms(leveled));
        Assert.True(Math.Abs(rmsDb - (-20.0)) < 0.1, $"RMS was {rmsDb} dBFS.");
    }

    [Fact]
    public void Level_PeakAboveCap_IsCappedAtPeak()
    {
        float[] sine = Tone(SampleRate, 0.5f);
        float[] leveled = SpeechTakeShaper.Level(sine, SampleRate, rmsDb: 0.0, peak: 0.89);
        float max = 0f;
        foreach (float v in leveled) max = Math.Max(max, Math.Abs(v));
        Assert.Equal(0.89f, max, 4);
    }

    [Fact]
    public void Level_SilentInput_ReturnsACopy()
    {
        float[] silence = Silence(SampleRate);
        float[] leveled = SpeechTakeShaper.Level(silence, SampleRate);
        Assert.Equal(silence, leveled);
        Assert.NotSame(silence, leveled);
    }

    [Fact]
    public void Shape_RunsTrimThenSqueezeWithDefaults()
    {
        // No outer silence to trim, so only the 1.0 s pause shrinks: 8000 + 5600 + 8000 samples.
        float[] take = Concat(Tone(8_000, 0.5f), Silence(16_000), Tone(8_000, 0.5f));
        SpeechTakeShaper.SpeechTakeShapingOptions options = new();
        float[] shaped = SpeechTakeShaper.Shape(take, SampleRate, options);
        Assert.Equal(21_600, shaped.Length);
        Assert.NotSame(take, shaped);
    }

    private static float[] Silence(int samples) => new float[samples];

    private static float[] Tone(int samples, float amplitude)
    {
        float[] y = new float[samples];
        for (int i = 0; i < samples; i++)
            y[i] = (float)(amplitude * Math.Sin(2.0 * Math.PI * 1000.0 * i / SampleRate));
        return y;
    }

    private static float[] Concat(params float[][] parts)
    {
        int total = 0;
        foreach (float[] part in parts) total += part.Length;
        float[] joined = new float[total];
        int at = 0;
        foreach (float[] part in parts)
        {
            part.CopyTo(joined, at);
            at += part.Length;
        }

        return joined;
    }

    private static double Rms(float[] y)
    {
        double sum = 0.0;
        foreach (float v in y) sum += (double)v * v;
        return Math.Sqrt(sum / y.Length);
    }
}

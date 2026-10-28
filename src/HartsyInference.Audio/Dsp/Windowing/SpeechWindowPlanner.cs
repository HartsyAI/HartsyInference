namespace HartsyInference.Audio.Dsp.Windowing;

/// <summary>Splits a long clip into contiguous windows, cutting only in quiet audio so no word is decoded twice.</summary>
public static class SpeechWindowPlanner
{
    /// <summary>Shortest window or tail a cut may leave; Whisper invents text on sub-second windows.</summary>
    public const double MinTailSeconds = 1.0;

    /// <summary>Longest band a cut is searched in, counted back from the window end.</summary>
    public const double MaxBandSeconds = 5.0;

    /// <summary>Shortest window the planner accepts; it must hold a minimum head and a minimum tail.</summary>
    public const double MinWindowSeconds = 2.0;

    /// <summary>Plans windows of at most <paramref name="maxWindowSamples"/> samples covering every sample once.</summary>
    /// <remarks>Input at or under the window gives one window. Silent windows are flagged so callers can skip them.</remarks>
    public static IReadOnlyList<SpeechWindow> Plan(ReadOnlySpan<float> audio, int sampleRate, int maxWindowSamples)
    {
        FrameEnergy.RequireSampleRate(sampleRate);
        int minTail = (int)(sampleRate * MinTailSeconds);
        if (maxWindowSamples < (int)(sampleRate * MinWindowSeconds))
        {
            throw new ArgumentOutOfRangeException(nameof(maxWindowSamples), maxWindowSamples, $"Windows must be at least {MinWindowSeconds} s.");
        }

        List<SpeechWindow> windows = [];
        if (audio.IsEmpty)
        {
            return windows;
        }

        int bandLength = Math.Min((int)(sampleRate * MaxBandSeconds), maxWindowSamples / 2);
        int pos = 0;
        while (audio.Length - pos > maxWindowSamples)
        {
            int limit = Math.Min(pos + maxWindowSamples, audio.Length - minTail);
            int bandStart = Math.Max(pos + minTail, limit - bandLength);
            SpeechCut cut = SpeechCutFinder.Find(audio.Slice(bandStart, limit - bandStart), sampleRate);
            int cutAt = bandStart + cut.Offset;
            windows.Add(Describe(audio, pos, cutAt - pos, cut.HardCut, sampleRate));
            pos = cutAt;
        }

        windows.Add(Describe(audio, pos, audio.Length - pos, false, sampleRate));
        return windows;
    }

    private static SpeechWindow Describe(ReadOnlySpan<float> audio, int start, int length, bool hardCut, int sampleRate)
    {
        int frame = FrameEnergy.FrameSamples(sampleRate);
        int frames = (length + frame - 1) / frame;
        double[] rms = FrameEnergy.FrameRms(audio.Slice(start, length), frame, frames);
        bool silent = FrameEnergy.MaxValue(rms) < FrameEnergy.SilenceFloorAmplitude;
        return new SpeechWindow(start, length, silent, hardCut);
    }
}

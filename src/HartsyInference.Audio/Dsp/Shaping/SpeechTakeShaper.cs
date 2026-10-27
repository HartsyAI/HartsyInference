namespace HartsyInference.Audio.Dsp.Shaping;

/// <summary>Trims silence, shortens long pauses and levels speech takes, matching the reference take shaping.</summary>
public static class SpeechTakeShaper
{
    private const double FrameSeconds = 0.02;
    private const double SilenceFloorDb = -55.0;
    private const double RmsEpsilon = 1e-9;
    private const double QuietFraction = 0.03;
    private const double ActiveFraction = 0.1;
    private const int MinSampleRate = 50;

    /// <summary>Settings for <see cref="Shape"/>. The defaults match the reference take shaping.</summary>
    public sealed record SpeechTakeShapingOptions
    {
        /// <summary>Depth in dB below the loudest frame that still counts as speech.</summary>
        public double TopDb { get; init; } = 40.0;

        /// <summary>Quiet runs longer than this many seconds are shortened.</summary>
        public double MaxPauseSeconds { get; init; } = 0.5;

        /// <summary>Seconds of a shortened pause kept in total, split evenly between its two ends.</summary>
        public double KeepSeconds { get; init; } = 0.35;
    }

    /// <summary>Drops leading and trailing silence, keeping two frames before and three after the speech.</summary>
    /// <remarks>Allocates the frame levels and the returned array. Returns a copy when no frame is above the floor.</remarks>
    public static float[] Trim(float[] y, int sampleRate, double topDb = 40.0)
    {
        ValidateArguments(y, sampleRate);
        ValidateNonNegative(topDb, nameof(topDb));
        if (y.Length == 0) return Array.Empty<float>();

        int frame = FrameSamples(sampleRate);
        double[] levelDb = FrameRms(y, frame, ReferenceFrameCount(y.Length, frame));
        for (int k = 0; k < levelDb.Length; k++) levelDb[k] = 20.0 * Math.Log10(levelDb[k] + RmsEpsilon);
        double maxDb = MaxValue(levelDb);

        double threshold = Math.Max(maxDb - topDb, SilenceFloorDb);
        int first = -1;
        int last = -1;
        for (int k = 0; k < levelDb.Length; k++)
        {
            if (levelDb[k] <= threshold) continue;
            if (first < 0) first = k;
            last = k;
        }

        if (first < 0) return (float[])y.Clone();
        int from = Math.Max(0, (first - 2) * frame);
        int to = Math.Min(y.Length, (last + 3) * frame);
        float[] trimmed = new float[to - from];
        Array.Copy(y, from, trimmed, 0, trimmed.Length);
        return trimmed;
    }

    /// <summary>Shortens each quiet run longer than <paramref name="maxPause"/> seconds to <paramref name="keep"/> seconds.</summary>
    /// <remarks>Allocates the frame levels, one intermediate buffer and the returned array. Samples after the last
    /// whole 20 ms frame are appended unchanged.</remarks>
    public static float[] Squeeze(float[] y, int sampleRate, double maxPause = 0.5, double keep = 0.35)
    {
        ValidateArguments(y, sampleRate);
        ValidateNonNegative(maxPause, nameof(maxPause));
        ValidateNonNegative(keep, nameof(keep));

        int frame = FrameSamples(sampleRate);
        int frames = y.Length / frame;
        if (frames == 0) return (float[])y.Clone();

        double[] rms = FrameRms(y, frame, frames);
        double quietBelow = MaxValue(rms) * QuietFraction;
        int keepEach = (int)(sampleRate * keep / 2.0);
        float[] buffer = new float[y.Length];
        int written = 0;
        int i = 0;
        while (i < frames)
        {
            bool quiet = rms[i] < quietBelow;
            int j = i + 1;
            while (j < frames && (rms[j] < quietBelow) == quiet) j++;

            int start = i * frame;
            int end = j * frame;
            int runSamples = end - start;
            if (quiet && (j - i) * FrameSeconds > maxPause)
            {
                int head = Math.Min(keepEach, runSamples);
                int tail = Math.Min(keepEach, runSamples);
                Array.Copy(y, start, buffer, written, head);
                written += head;
                Array.Copy(y, end - tail, buffer, written, tail);
                written += tail;
            }
            else
            {
                Array.Copy(y, start, buffer, written, runSamples);
                written += runSamples;
            }

            i = j;
        }

        int rest = y.Length - frames * frame;
        Array.Copy(y, frames * frame, buffer, written, rest);
        written += rest;
        Array.Resize(ref buffer, written);
        return buffer;
    }

    /// <summary>Scales a take so its active frames reach <paramref name="rmsDb"/>, then caps its peak.</summary>
    /// <remarks>Allocates the frame levels and the returned array. Frames below a tenth of the loudest frame are
    /// excluded from the loudness estimate. Returns a copy when the take is silent.</remarks>
    public static float[] Level(float[] y, int sampleRate, double rmsDb = -20.0, double peak = 0.89)
    {
        ValidateArguments(y, sampleRate);
        if (peak <= 0.0) throw new ArgumentOutOfRangeException(nameof(peak), peak, "Must be positive.");
        if (y.Length == 0) return Array.Empty<float>();

        int frame = FrameSamples(sampleRate);
        double[] rms = FrameRms(y, frame, ReferenceFrameCount(y.Length, frame));
        double maxRms = MaxValue(rms);
        if (maxRms <= 0.0) return (float[])y.Clone();

        double activeFloor = maxRms * ActiveFraction;
        double sumSquares = 0.0;
        int active = 0;
        for (int k = 0; k < rms.Length; k++)
        {
            if (rms[k] <= activeFloor) continue;
            sumSquares += rms[k] * rms[k];
            active++;
        }

        double gain = Math.Pow(10.0, rmsDb / 20.0) / Math.Sqrt(sumSquares / active);
        double scaledPeak = 0.0;
        for (int k = 0; k < y.Length; k++) scaledPeak = Math.Max(scaledPeak, Math.Abs(y[k] * gain));
        double cap = scaledPeak > peak ? peak / scaledPeak : 1.0;

        float[] leveled = new float[y.Length];
        for (int k = 0; k < y.Length; k++) leveled[k] = (float)(y[k] * gain * cap);
        return leveled;
    }

    /// <summary>Runs <see cref="Trim"/> then <see cref="Squeeze"/>. Leveling is left to the caller on the joined take.</summary>
    /// <remarks>Allocates the intermediate trimmed array and the returned array.</remarks>
    public static float[] Shape(float[] y, int sampleRate, SpeechTakeShapingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        // Reference order: trim(squeeze(take)). Squeeze first so edge frames are judged on the squeezed take.
        float[] squeezed = Squeeze(y, sampleRate, options.MaxPauseSeconds, options.KeepSeconds);
        return Trim(squeezed, sampleRate, options.TopDb);
    }

    private static void ValidateArguments(float[] y, int sampleRate)
    {
        ArgumentNullException.ThrowIfNull(y);
        if (sampleRate < MinSampleRate)
            throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "Must be at least 50 Hz.");
    }

    private static void ValidateNonNegative(double value, string name)
    {
        if (value < 0.0) throw new ArgumentOutOfRangeException(name, value, "Must be non-negative.");
    }

    private static int FrameSamples(int sampleRate) => (int)(sampleRate * FrameSeconds);

    // The reference enumerates frame starts below max(1, length - frame), so its last frame can be dropped.
    private static int ReferenceFrameCount(int length, int frame)
    {
        int limit = Math.Max(1, length - frame);
        return (limit + frame - 1) / frame;
    }

    private static double[] FrameRms(float[] y, int frame, int count)
    {
        double[] rms = new double[count];
        for (int k = 0; k < count; k++)
        {
            int start = k * frame;
            int end = Math.Min(start + frame, y.Length);
            double sum = 0.0;
            for (int i = start; i < end; i++) sum += (double)y[i] * y[i];
            rms[k] = Math.Sqrt(sum / (end - start));
        }

        return rms;
    }

    private static double MaxValue(double[] values)
    {
        double max = values[0];
        for (int k = 1; k < values.Length; k++) max = Math.Max(max, values[k]);
        return max;
    }
}

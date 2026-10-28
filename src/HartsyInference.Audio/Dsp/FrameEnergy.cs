namespace HartsyInference.Audio.Dsp;

/// <summary>Short-frame RMS shared by the take shaper and the speech window planner, so their silence rules agree.</summary>
public static class FrameEnergy
{
    /// <summary>Analysis frame length in seconds.</summary>
    public const double FrameSeconds = 0.02;

    /// <summary>Frames below this level in dBFS count as silence.</summary>
    public const double SilenceFloorDb = -55.0;

    /// <summary>A frame is quiet when its RMS is below this fraction of the loudest frame's RMS.</summary>
    public const double QuietFraction = 0.03;

    /// <summary>Frame length in samples at <paramref name="sampleRate"/>.</summary>
    public static int FrameSamples(int sampleRate) => (int)(sampleRate * FrameSeconds);

    /// <summary>RMS of each of the first <paramref name="count"/> frames; the last frame may be shorter than <paramref name="frame"/>.</summary>
    public static double[] FrameRms(ReadOnlySpan<float> y, int frame, int count)
    {
        double[] rms = new double[count];
        for (int k = 0; k < count; k++)
        {
            int start = k * frame;
            int end = Math.Min(start + frame, y.Length);
            double sum = 0.0;
            for (int i = start; i < end; i++)
            {
                sum += (double)y[i] * y[i];
            }
            rms[k] = Math.Sqrt(sum / (end - start));
        }
        return rms;
    }

    /// <summary>Largest value in <paramref name="values"/>; the array must not be empty.</summary>
    public static double MaxValue(double[] values)
    {
        double max = values[0];
        for (int k = 1; k < values.Length; k++)
        {
            max = Math.Max(max, values[k]);
        }
        return max;
    }

    /// <summary>Amplitude of the silence floor, the linear form of <see cref="SilenceFloorDb"/>.</summary>
    public static double SilenceFloorAmplitude => Math.Pow(10.0, SilenceFloorDb / 20.0);
}

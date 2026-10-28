namespace HartsyInference.Audio.Dsp.Windowing;

/// <summary>Finds the quietest place in a band of audio to cut a long clip.</summary>
public static class SpeechCutFinder
{
    /// <summary>Returns the centre of the longest quiet run, or the quietest frame as a hard cut when nothing is quiet.</summary>
    /// <remarks>Quiet is under the larger of the relative and absolute floors. Only whole frames count; a shorter band is one frame.</remarks>
    public static SpeechCut Find(ReadOnlySpan<float> band, int sampleRate)
    {
        if (band.IsEmpty)
        {
            throw new ArgumentOutOfRangeException(nameof(band), "The band must hold at least one sample.");
        }
        FrameEnergy.RequireSampleRate(sampleRate);

        int frame = FrameEnergy.FrameSamples(sampleRate);
        int frames = Math.Max(1, band.Length / frame);
        double[] rms = FrameEnergy.FrameRms(band, frame, frames);
        double quietBelow = Math.Max(FrameEnergy.MaxValue(rms) * FrameEnergy.QuietFraction, FrameEnergy.SilenceFloorAmplitude);

        int bestStart = 0;
        int bestLength = 0;
        int runStart = -1;
        for (int k = 0; k <= frames; k++)
        {
            bool quiet = k < frames && rms[k] < quietBelow;
            if (quiet && runStart < 0)
            {
                runStart = k;
            }
            else if (!quiet && runStart >= 0)
            {
                if (k - runStart > bestLength)
                {
                    bestStart = runStart;
                    bestLength = k - runStart;
                }
                runStart = -1;
            }
        }

        if (bestLength > 0)
        {
            int centre = bestStart * frame + bestLength * frame / 2;
            return new SpeechCut(Math.Min(centre, band.Length - 1), HardCut: false);
        }

        int quietest = 0;
        for (int k = 1; k < frames; k++)
        {
            if (rms[k] < rms[quietest])
            {
                quietest = k;
            }
        }
        return new SpeechCut(Math.Min(quietest * frame + frame / 2, band.Length - 1), HardCut: true);
    }
}

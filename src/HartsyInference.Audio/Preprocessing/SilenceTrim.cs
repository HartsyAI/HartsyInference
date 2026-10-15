namespace HartsyInference.Audio.Preprocessing;

/// <summary>Leading and trailing silence trim, the port of <c>librosa.effects.trim</c> (librosa 0.10+): frame RMS over a
/// zero-padded centered framing, converted to dB against the loudest frame, frames above <c>-topDb</c> are kept, and the
/// kept span runs from the first such frame's start to the end of the last one.</summary>
public static class SilenceTrim
{
    /// <summary>librosa's <c>amplitude_to_db</c> floor on the RMS (amin 1e-5, so a power floor of 1e-10).</summary>
    private const double AmplitudeFloor = 1e-5;

    /// <summary>The <c>[Start, End)</c> sample span <c>librosa.effects.trim(y, top_db=topDb, frame_length, hop_length)</c>
    /// keeps; <c>(0, 0)</c> when the signal is empty.</summary>
    public static (int Start, int End) Bounds(ReadOnlySpan<float> y, float topDb = 60f, int frameLength = 2048,
        int hopLength = 512)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(frameLength, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(hopLength, 1);
        if (y.IsEmpty) return (0, 0);

        // feature.rms(center=True, pad_mode="constant"): frame_length // 2 zeros each side, then util.frame.
        int pad = frameLength / 2;
        int frames = 1 + (y.Length + 2 * pad - frameLength) / hopLength;
        double[] rms = new double[frames];
        double loudest = 0;
        for (int f = 0; f < frames; f++)
        {
            int first = f * hopLength - pad;
            double sum = 0;
            int from = Math.Max(first, 0), to = Math.Min(first + frameLength, y.Length);
            for (int i = from; i < to; i++) sum += (double)y[i] * y[i];
            rms[f] = Math.Sqrt(sum / frameLength);
            loudest = Math.Max(loudest, rms[f]);
        }

        double refDb = 20.0 * Math.Log10(Math.Max(AmplitudeFloor, loudest));
        int firstKept = -1, lastKept = -1;
        for (int f = 0; f < frames; f++)
        {
            double db = 20.0 * Math.Log10(Math.Max(AmplitudeFloor, rms[f])) - refDb;
            if (db <= -topDb) continue;
            if (firstKept < 0) firstKept = f;
            lastKept = f;
        }
        if (firstKept < 0) return (0, 0);
        return (firstKept * hopLength, Math.Min(y.Length, (lastKept + 1) * hopLength));
    }
}

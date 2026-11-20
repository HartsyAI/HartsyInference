namespace HartsyInference.Audio.Models.Haru;

/// <summary>Preparation of a voice reference: mono 24 kHz audio with its silence trimmed and the last stretch kept.</summary>
internal static class HaruAudio
{
    /// <summary>Longest reference the model is prompted with.</summary>
    internal const double MaxReferenceSeconds = 180.0;

    private const double SilenceDbfs = -45.0;
    private const double MarginSeconds = 0.2;

    /// <summary>Cuts leading and trailing audio quieter than -45 dBFS (20 ms RMS windows) with a 0.2 s margin, then keeps at most the last
    /// <paramref name="maxSeconds"/>.</summary>
    internal static float[] PrepareReference(ReadOnlySpan<float> audio, int sampleRate, double maxSeconds = MaxReferenceSeconds)
    {
        ReadOnlySpan<float> trimmed = TrimSilence(audio, sampleRate);
        int limit = (int)(maxSeconds * sampleRate);
        if (trimmed.Length > limit)
        {
            trimmed = trimmed[^limit..];
        }
        return trimmed.ToArray();
    }

    /// <summary>The span between the first and last 20 ms window louder than -45 dBFS, with a 0.2 s margin each side; empty when the whole
    /// clip is quieter, the clip itself when it is shorter than one window.</summary>
    internal static ReadOnlySpan<float> TrimSilence(ReadOnlySpan<float> audio, int sampleRate)
    {
        int window = sampleRate / 50;
        int count = audio.Length / window;
        if (count == 0)
        {
            return audio;
        }
        int first = -1;
        int last = -1;
        for (int w = 0; w < count; w++)
        {
            double sum = 0;
            foreach (float v in audio.Slice(w * window, window))
            {
                sum += (double)v * v;
            }
            double db = 20.0 * Math.Log10(Math.Sqrt(sum / window + 1e-12));
            if (db > SilenceDbfs)
            {
                if (first < 0)
                {
                    first = w;
                }
                last = w;
            }
        }
        if (first < 0)
        {
            return audio[..0];
        }
        int margin = (int)(MarginSeconds * sampleRate);
        int start = Math.Max(0, first * window - margin);
        int end = Math.Min(audio.Length, (last + 1) * window + margin);
        return audio[start..end];
    }
}

namespace HartsyInference.Audio.Preprocessing;

/// <summary>Post-synthesis loudness normalization for TTS waveforms. This is a deliberate product step,
/// NOT a parity step: neither the reference implementations nor the vocoders apply output gain, so every
/// parity test must run with normalization disabled. Free-running AR TTS models can drift toward low-energy
/// codes (e.g. Fish-Speech renders ~10x quiet), and this restores a consistent, audible level.</summary>
public static class LoudnessNormalizer
{
    /// <summary>Below this peak the buffer is treated as silence and left untouched (avoids amplifying noise
    /// floor / DC in an empty clip).</summary>
    public const float SilenceThreshold = 1e-4f;

    /// <summary>Scales <paramref name="audio"/> in place so its maximum absolute sample equals
    /// <paramref name="peak"/>. Returns the same array for chaining. No-op on silence.</summary>
    public static float[] PeakNormalize(float[] audio, float peak = 0.99f)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (audio.Length == 0) return audio;
        float maxAbs = 0f;
        for (int i = 0; i < audio.Length; i++)
        {
            float a = MathF.Abs(audio[i]);
            if (a > maxAbs) maxAbs = a;
        }
        if (maxAbs < SilenceThreshold) return audio;
        float gain = peak / maxAbs;
        for (int i = 0; i < audio.Length; i++) audio[i] *= gain;
        return audio;
    }

    /// <summary>Scales <paramref name="audio"/> in place to a target RMS, then clamps the applied gain so the
    /// peak never exceeds <paramref name="peakCeiling"/> (RMS targeting gives perceptually even loudness across
    /// clips; the ceiling prevents clipping on transient-heavy content). Returns the same array. No-op on silence.</summary>
    public static float[] RmsNormalize(float[] audio, float targetRms = 0.1f, float peakCeiling = 0.99f)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (audio.Length == 0) return audio;
        double sumSq = 0.0;
        float maxAbs = 0f;
        for (int i = 0; i < audio.Length; i++)
        {
            float a = audio[i];
            sumSq += (double)a * a;
            float abs = MathF.Abs(a);
            if (abs > maxAbs) maxAbs = abs;
        }
        if (maxAbs < SilenceThreshold) return audio;
        float rms = (float)Math.Sqrt(sumSq / audio.Length);
        if (rms < SilenceThreshold) return audio;
        float gain = targetRms / rms;
        if (maxAbs * gain > peakCeiling) gain = peakCeiling / maxAbs;
        for (int i = 0; i < audio.Length; i++) audio[i] *= gain;
        return audio;
    }
}

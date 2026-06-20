namespace HartsyInference.Audio.Models.Rvc;

/// <summary>RVC pitch helpers — the mel-scale coarse quantization that turns continuous per-frame F0 (Hz) into
/// the 1..255 bins fed to <c>emb_pitch</c>, and the optional pitch shift. Pure, unit-tested.</summary>
public static class RvcPitch
{
    /// <summary>Shifts F0 by <paramref name="semitones"/> (the RVC <c>f0_up_key</c>): <c>f0 · 2^(n/12)</c>.</summary>
    public static float[] Shift(ReadOnlySpan<float> f0, float semitones)
    {
        float factor = MathF.Pow(2f, semitones / 12f);
        float[] outArr = new float[f0.Length];
        for (int i = 0; i < f0.Length; i++) outArr[i] = f0[i] * factor;
        return outArr;
    }

    /// <summary>Coarse-quantizes F0 (Hz) to integer bins in <c>[1, 255]</c> via the mel formula
    /// <c>1127·ln(1+f/700)</c> mapped over <c>[f0Min, f0Max]</c>.</summary>
    public static int[] CoarseQuantize(ReadOnlySpan<float> f0, float f0Min, float f0Max)
    {
        float melMin = 1127f * MathF.Log(1f + f0Min / 700f);
        float melMax = 1127f * MathF.Log(1f + f0Max / 700f);
        int[] bins = new int[f0.Length];
        for (int i = 0; i < f0.Length; i++)
        {
            float mel = 1127f * MathF.Log(1f + f0[i] / 700f);
            float scaled = mel > 0 ? (mel - melMin) * 254f / (melMax - melMin) + 1f : mel;
            int b = (int)MathF.Round(scaled);
            bins[i] = Math.Clamp(b, 1, 255);
        }
        return bins;
    }
}

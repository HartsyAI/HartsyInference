namespace HartsyInference.Audio.Preprocessing;

/// <summary>Waveform edge padding shared by the STFT front ends.</summary>
public static class SignalPadding
{
    /// <summary>Reflect-pads <paramref name="audio"/> by <paramref name="pad"/> samples each side using the
    /// edge-excluding ("reflect-101") convention that <c>torch.stft(center=True, pad_mode="reflect")</c> uses.</summary>
    public static float[] Reflect(ReadOnlySpan<float> audio, int pad)
    {
        int len = audio.Length;
        float[] outp = new float[len + 2 * pad];
        for (int i = 0; i < outp.Length; i++)
            outp[i] = audio[ReflectIndex(i - pad, len)];
        return outp;
    }

    // Folds an out-of-range index back into [0, len) by mirroring about both edges without repeating them,
    // so a pad wider than the signal keeps reflecting periodically instead of throwing.
    private static int ReflectIndex(int j, int len)
    {
        if (len <= 1) return 0;
        int period = 2 * (len - 1);
        int m = ((j % period) + period) % period;
        return m < len ? m : period - m;
    }
}

namespace HartsyInference.Audio.Dsp;

/// <summary>Pure-C# fundamental-frequency (F0) estimation via the YIN algorithm (de Cheveigné &amp; Kawahara,
/// 2002). Produces one F0 value (Hz; 0 = unvoiced) per hop — the per-frame pitch contour RVC / OpenVoice need
/// from a raw waveform. The engine ships no neural pitch tracker (RMVPE); YIN is the classical, dependency-free
/// equivalent ("pm"/"harvest"-class quality). For RVC: 16 kHz mono, hop 320 → 50 Hz, aligned with HuBERT frames.</summary>
public static class F0Estimator
{
    /// <summary>Estimates the F0 contour of mono <paramref name="audio"/> at <paramref name="sampleRate"/>.
    /// Returns <c>audio.Length / hopSize</c> values (Hz, 0 where unvoiced) — one per hop, matching a HuBERT
    /// content stream at the same hop.</summary>
    public static float[] EstimateYin(ReadOnlySpan<float> audio, int sampleRate, int hopSize = 320,
        float fMin = 50f, float fMax = 1100f, float threshold = 0.1f)
    {
        if (audio.Length == 0 || hopSize <= 0)
        {
            return [];
        }
        int tauMin = Math.Max(1, (int)(sampleRate / fMax));
        int tauMax = Math.Min(audio.Length, (int)(sampleRate / fMin) + 1);
        if (tauMax <= tauMin)
        {
            return new float[audio.Length / hopSize];
        }
        int window = 2 * tauMax;                  // two max-periods → stable difference function
        int frames = audio.Length / hopSize;
        float[] f0 = new float[frames];
        float[] diff = new float[tauMax];
        float[] cmnd = new float[tauMax];
        for (int frame = 0; frame < frames; frame++)
        {
            int start = frame * hopSize - window / 2; // centre the window on the hop
            f0[frame] = EstimateFrame(audio, start, window, tauMin, tauMax, sampleRate, threshold, diff, cmnd);
        }
        return f0;
    }

    /// <summary>One YIN frame: difference function → cumulative-mean-normalized difference → absolute threshold
    /// → parabolic interpolation. Returns Hz, or 0 if no period passes the threshold (unvoiced).</summary>
    private static float EstimateFrame(ReadOnlySpan<float> audio, int start, int window, int tauMin, int tauMax,
        int sampleRate, float threshold, float[] diff, float[] cmnd)
    {
        for (int tau = 0; tau < tauMax; tau++)
        {
            float sum = 0f;
            for (int j = 0; j < window; j++)
            {
                int a = start + j;
                int b = a + tau;
                float x = (uint)a < (uint)audio.Length ? audio[a] : 0f;
                float y = (uint)b < (uint)audio.Length ? audio[b] : 0f;
                float d = x - y;
                sum += d * d;
            }
            diff[tau] = sum;
        }

        cmnd[0] = 1f;
        float running = 0f;
        for (int tau = 1; tau < tauMax; tau++)
        {
            running += diff[tau];
            cmnd[tau] = running > 0f ? diff[tau] * tau / running : 1f;
        }

        int chosen = -1;
        for (int tau = tauMin; tau < tauMax; tau++)
        {
            if (cmnd[tau] < threshold)
            {
                while (tau + 1 < tauMax && cmnd[tau + 1] < cmnd[tau])
                {
                    tau++;
                }
                chosen = tau;
                break;
            }
        }
        if (chosen < 0)
        {
            return 0f; // unvoiced
        }

        float refinedTau = chosen;
        if (chosen > tauMin && chosen + 1 < tauMax)
        {
            float s0 = cmnd[chosen - 1], s1 = cmnd[chosen], s2 = cmnd[chosen + 1];
            float denom = 2f * (2f * s1 - s2 - s0);
            if (MathF.Abs(denom) > 1e-9f)
            {
                refinedTau = chosen + (s2 - s0) / denom;
            }
        }
        return refinedTau > 0f ? sampleRate / refinedTau : 0f;
    }
}

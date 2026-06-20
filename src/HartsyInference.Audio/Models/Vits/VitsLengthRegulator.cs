namespace HartsyInference.Audio.Models.Vits;

/// <summary>VITS monotonic length regulation — the inference-time equivalent of <c>generate_path</c> +
/// <c>matmul(attn, m_p)</c>: each text frame <c>i</c> is copied <c>w_ceil[i]</c> times along the output time
/// axis (a hard repeat/gather, not a real matmul). Also the VITS phoneme interspersing (a blank token 0
/// between every phoneme). Pure index bookkeeping — unit-tested without weights.</summary>
public static class VitsLengthRegulator
{
    /// <summary>Inserts <paramref name="blank"/> (id 0) between every phoneme and wraps with optional BOS/EOS
    /// (the VITS <c>add_blank=True</c> sequence the model is trained on).</summary>
    public static int[] Intersperse(ReadOnlySpan<int> phonemes, int blank = 0, int bos = -1, int eos = -1)
    {
        // Layout: [bos?, p0, blank, p1, blank, ..., pN, eos?] — blanks between phonemes, not trailing.
        int n = phonemes.Length;
        int len = (bos >= 0 ? 1 : 0) + (n == 0 ? 0 : 2 * n - 1) + (eos >= 0 ? 1 : 0);
        int[] outArr = new int[len];
        int w = 0;
        if (bos >= 0) outArr[w++] = bos;
        for (int i = 0; i < n; i++)
        {
            outArr[w++] = phonemes[i];
            if (i < n - 1) outArr[w++] = blank;
        }
        if (eos >= 0) outArr[w++] = eos;
        return outArr;
    }

    /// <summary>Per-text-frame durations from a log-duration prediction: <c>ceil(exp(logw) * lengthScale)</c>,
    /// clamped to ≥1 (matching <c>w_ceil</c> + the <c>clamp_min(...,1)</c> on the total).</summary>
    public static int[] Durations(ReadOnlySpan<float> logw, float lengthScale)
    {
        int[] w = new int[logw.Length];
        for (int i = 0; i < logw.Length; i++)
        {
            float v = MathF.Exp(logw[i]) * lengthScale;
            w[i] = Math.Max(1, (int)MathF.Ceiling(v));
        }
        return w;
    }

    /// <summary>Total output frames = Σ durations (≥1).</summary>
    public static int TotalFrames(ReadOnlySpan<int> durations)
    {
        int sum = 0;
        for (int i = 0; i < durations.Length; i++) sum += durations[i];
        return Math.Max(1, sum);
    }

    /// <summary>Expands a channels-first sequence <c>[C, Tx]</c> to <c>[C, Ty]</c> by repeating text frame
    /// <c>i</c> exactly <c>durations[i]</c> times along time.</summary>
    public static unsafe void Expand(float* src, float* dst, int channels, int tx, ReadOnlySpan<int> durations, int ty)
    {
        for (int c = 0; c < channels; c++)
        {
            int outT = 0;
            for (int i = 0; i < tx; i++)
            {
                float v = src[(long)c * tx + i];
                int reps = durations[i];
                for (int r = 0; r < reps && outT < ty; r++) dst[(long)c * ty + outT++] = v;
            }
        }
    }
}

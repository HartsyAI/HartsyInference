namespace SharpInference.Audio.Models.Music;

/// <summary>The MusicGen <b>delay pattern</b>: each residual codebook k is offset by <c>delay[k]</c>
/// timesteps so the AR decoder predicts one flat stream while the codebooks stay causally staggered.
/// <see cref="Apply"/> stages real codes into the delayed grid (special token fills the lead-in);
/// <see cref="Revert"/> reads them back after generation. <c>Revert(Apply(x)) == x</c>.</summary>
public static class MusicGenDelay
{
    /// <summary>Stages <paramref name="real"/> <c>[T, K]</c> into the delayed grid <c>[T + maxDelay, K]</c>.
    /// <c>delayed[s,k] = real[s-delay[k],k]</c> when in range, else <paramref name="special"/>.</summary>
    public static int[,] Apply(int[,] real, int[] delay, int special)
    {
        int t = real.GetLength(0);
        int k = real.GetLength(1);
        int max = Max(delay);
        int[,] delayed = new int[t + max, k];
        for (int s = 0; s < t + max; s++)
            for (int c = 0; c < k; c++)
            {
                int src = s - delay[c];
                delayed[s, c] = src >= 0 && src < t ? real[src, c] : special;
            }
        return delayed;
    }

    /// <summary>Reads <paramref name="tReal"/> real frames back out of the delayed grid
    /// <paramref name="delayed"/> <c>[Tg, K]</c>: <c>real[j,k] = delayed[j+delay[k],k]</c>.</summary>
    public static int[,] Revert(int[,] delayed, int[] delay, int tReal)
    {
        int k = delayed.GetLength(1);
        int[,] real = new int[tReal, k];
        for (int j = 0; j < tReal; j++)
            for (int c = 0; c < k; c++)
                real[j, c] = delayed[j + delay[c], c];
        return real;
    }

    /// <summary>Whether codebook <paramref name="c"/> emits a real token at decode step <paramref name="step"/>
    /// (it is still in its special-token lead-in while <c>step &lt; delay[c]</c>).</summary>
    public static bool IsActive(int step, int c, int[] delay) => step >= delay[c];

    public static int Max(int[] delay)
    {
        int m = 0;
        for (int i = 0; i < delay.Length; i++) if (delay[i] > m) m = delay[i];
        return m;
    }
}

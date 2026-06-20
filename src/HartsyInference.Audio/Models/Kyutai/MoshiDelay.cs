namespace HartsyInference.Audio.Models.Kyutai;

/// <summary>Per-codebook delay roll for the Moshi/Kyutai TTS audio streams — the pure index bookkeeping that
/// staggers each codebook by its delay (the depformer emits codes un-delayed; the temporal stream consumes
/// them delayed). Kept apart from the pipeline and unit-tested, like <c>MusicGenDelay</c>.</summary>
public static class MoshiDelay
{
    /// <summary>Applies per-stream delays to a <c>[numStreams, T]</c> grid: stream <c>s</c> is shifted right by
    /// <c>delays[s]</c>, the leading <c>delays[s]</c> slots filled with <paramref name="fill"/>. The grid grows
    /// by <c>max(delays)</c> columns.</summary>
    public static int[,] Apply(int[,] streams, int[] delays, int fill)
    {
        int n = streams.GetLength(0);
        int t = streams.GetLength(1);
        int maxDelay = Max(delays);
        int[,] outG = new int[n, t + maxDelay];
        for (int s = 0; s < n; s++)
        {
            int d = delays[s];
            for (int j = 0; j < t + maxDelay; j++)
                outG[s, j] = j < d || j - d >= t ? fill : streams[s, j - d];
        }
        return outG;
    }

    /// <summary>Inverts <see cref="Apply"/> — recovers the first <paramref name="tReal"/> real columns of each
    /// stream from a delayed grid.</summary>
    public static int[,] Revert(int[,] delayed, int[] delays, int tReal)
    {
        int n = delayed.GetLength(0);
        int[,] outG = new int[n, tReal];
        for (int s = 0; s < n; s++)
        {
            int d = delays[s];
            for (int j = 0; j < tReal; j++)
                outG[s, j] = delayed[s, j + d];
        }
        return outG;
    }

    public static int Max(int[] delays)
    {
        int m = 0;
        for (int i = 0; i < delays.Length; i++) if (delays[i] > m) m = delays[i];
        return m;
    }
}

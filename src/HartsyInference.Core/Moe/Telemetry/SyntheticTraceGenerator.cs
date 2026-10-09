namespace HartsyInference.Core.Moe.Telemetry;

/// <summary>
/// Seeded synthetic routing traces. Each layer gets its own seeded ranking of experts, and experts are drawn from a Zipf
/// distribution over that ranking, so a few experts per layer are hot. The same seed always produces the same trace.
/// </summary>
public static class SyntheticTraceGenerator
{
    /// <summary>
    /// Generates <paramref name="steps"/> steps; each step routes <paramref name="topK"/> distinct experts in every layer.
    /// </summary>
    public static ExpertTraceRecord[] Zipf(
        ulong seed,
        int layers,
        int expertsPerLayer,
        int steps,
        int topK,
        double exponent,
        uint bytesPerExpert)
    {
        if (layers <= 0 || layers > ushort.MaxValue) throw new ArgumentOutOfRangeException(nameof(layers));
        if (expertsPerLayer <= 0 || expertsPerLayer > ushort.MaxValue) throw new ArgumentOutOfRangeException(nameof(expertsPerLayer));
        if (steps < 0) throw new ArgumentOutOfRangeException(nameof(steps));
        if (topK <= 0 || topK > expertsPerLayer) throw new ArgumentOutOfRangeException(nameof(topK));
        if (!(exponent >= 0.0)) throw new ArgumentOutOfRangeException(nameof(exponent));

        SplitMix64 rng = new(seed);
        double[] cdf = BuildCdf(expertsPerLayer, exponent);
        int[][] rankToExpert = new int[layers][];
        for (int layer = 0; layer < layers; layer++)
        {
            rankToExpert[layer] = Shuffle(expertsPerLayer, ref rng);
        }

        ExpertTraceRecord[] trace = new ExpertTraceRecord[checked(steps * layers * topK)];
        int[] chosen = new int[topK];
        int position = 0;
        for (int step = 0; step < steps; step++)
        {
            for (int layer = 0; layer < layers; layer++)
            {
                int picked = 0;
                while (picked < topK)
                {
                    int rank = SampleRank(cdf, rng.NextUnit());
                    if (Array.IndexOf(chosen, rank, 0, picked) >= 0) continue;
                    chosen[picked++] = rank;
                }

                for (int k = 0; k < topK; k++)
                {
                    int expert = rankToExpert[layer][chosen[k]];
                    trace[position++] = new ExpertTraceRecord((ushort)layer, (ushort)expert, bytesPerExpert);
                }
            }
        }

        return trace;
    }

    private static double[] BuildCdf(int n, double exponent)
    {
        double[] cdf = new double[n];
        double total = 0.0;
        for (int rank = 0; rank < n; rank++)
        {
            total += 1.0 / Math.Pow(rank + 1, exponent);
            cdf[rank] = total;
        }

        for (int rank = 0; rank < n; rank++)
        {
            cdf[rank] /= total;
        }

        cdf[n - 1] = 1.0;
        return cdf;
    }

    private static int SampleRank(double[] cdf, double u)
    {
        int lo = 0;
        int hi = cdf.Length - 1;
        while (lo < hi)
        {
            int mid = lo + ((hi - lo) / 2);
            if (cdf[mid] > u) hi = mid;
            else lo = mid + 1;
        }

        return lo;
    }

    private static int[] Shuffle(int n, ref SplitMix64 rng)
    {
        int[] values = new int[n];
        for (int i = 0; i < n; i++) values[i] = i;
        for (int i = n - 1; i > 0; i--)
        {
            int j = (int)(rng.Next() % (ulong)(i + 1));
            (values[i], values[j]) = (values[j], values[i]);
        }

        return values;
    }

    private struct SplitMix64
    {
        private ulong state;

        public SplitMix64(ulong seed) => state = seed;

        public ulong Next()
        {
            state += 0x9E3779B97F4A7C15UL;
            ulong z = state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }

        public double NextUnit() => (Next() >> 11) * (1.0 / (1UL << 53));
    }
}

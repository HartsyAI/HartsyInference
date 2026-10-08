using Xunit;
using HartsyInference.LLM.Generation.Speculative;

namespace HartsyInference.LLM.Tests.Speculative;

/// <summary>Deterministic draws for the statistical tests: splitmix64 on the 53 high bits, uniform on [0, 1).</summary>
internal static class SpeculativeTestSupport
{
    public static Func<double> Uniform(ulong seed)
    {
        ulong state = seed;
        return () =>
        {
            state += 0x9E3779B97F4A7C15UL;
            ulong z = state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            z ^= z >> 31;
            return (z >> 11) * (1.0 / (1UL << 53));
        };
    }

    /// <summary>Pearson chi-square of observed counts against expected probabilities; categories with zero expected mass must have zero counts.</summary>
    public static double ChiSquare(long[] observed, double[] expectedProbs, out int df)
    {
        long n = observed.Sum();
        double stat = 0;
        int cells = 0;
        for (int v = 0; v < observed.Length; v++)
        {
            if (expectedProbs[v] <= 0)
            {
                Assert.Equal(0, observed[v]);
                continue;
            }
            double expected = expectedProbs[v] * n;
            stat += Math.Pow(observed[v] - expected, 2) / expected;
            cells++;
        }
        df = cells - 1;
        return stat;
    }

    /// <summary>Two-sample chi-square over the categories that either histogram hits.</summary>
    public static double TwoSampleChiSquare(long[] a, long[] b, out int df)
    {
        double stat = 0;
        int cells = 0;
        long na = a.Sum(), nb = b.Sum();
        for (int v = 0; v < a.Length; v++)
        {
            if (a[v] + b[v] == 0) continue;
            double pooled = (a[v] + b[v]) / (double)(na + nb);
            double ea = pooled * na, eb = pooled * nb;
            stat += Math.Pow(a[v] - ea, 2) / ea + Math.Pow(b[v] - eb, 2) / eb;
            cells++;
        }
        df = cells - 1;
        return stat;
    }

    // chi-square critical values at alpha = 0.001, for the degrees of freedom these tests use
    public static double Critical(int df) => df switch
    {
        1 => 10.83, 2 => 13.82, 3 => 16.27, 4 => 18.47, 5 => 20.52, 6 => 22.46, 7 => 24.32, 8 => 26.12, 9 => 27.88, 10 => 29.59,
        _ => throw new ArgumentOutOfRangeException(nameof(df)),
    };
}

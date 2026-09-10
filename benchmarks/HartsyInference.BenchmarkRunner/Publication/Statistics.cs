namespace HartsyInference.BenchmarkRunner.Publication;
/// <summary>Equal-weight machine summaries and deterministic machine-level bootstrap intervals.</summary>
public static class Statistics
{
    public static double Median(IEnumerable<double> source)
    {
        double[] values = source.Order().ToArray();
        if (values.Length == 0 || values.Any(v => !double.IsFinite(v)))
            throw new ArgumentException("Empty or nonfinite sample.");
        return values.Length % 2 == 0 ? (values[values.Length / 2 - 1] + values[values.Length / 2]) / 2 : values[values.Length / 2];
    }

    public static (double Low, double High) Interval(double[] machines)
    {
        if (machines.Length < 2)
            return (Median(machines), Median(machines));
        Random random = new(42);
        double[] bootstraps = new double[2000], sample = new double[machines.Length];
        for (int i = 0; i < bootstraps.Length; i++)
        {
            for (int j = 0; j < sample.Length; j++)
                sample[j] = machines[random.Next(machines.Length)];
            bootstraps[i] = Median(sample);
        }

        Array.Sort(bootstraps);
        return (bootstraps[49], bootstraps[1949]);
    }
}

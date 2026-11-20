using System.Diagnostics;

namespace HartsyInference.ThreeD.Models.Idol.Budget;

/// <summary>Stage timers for profiling the Budget3D forward (off by default; a test turns it on). Not thread-safe: the pipeline runs one request at a time.</summary>
internal static class BudgetTimer
{
    private static readonly Dictionary<string, double> Totals = [];

    internal static bool Enabled { get; set; }

    internal static void Reset() => Totals.Clear();

    internal static IReadOnlyDictionary<string, double> Milliseconds => Totals;

    /// <summary>Times <paramref name="work"/> under <paramref name="name"/> when profiling is on.</summary>
    internal static T Measure<T>(string name, Func<T> work)
    {
        if (!Enabled)
            return work();
        long start = Stopwatch.GetTimestamp();
        try
        {
            return work();
        }
        finally
        {
            Totals[name] = Totals.GetValueOrDefault(name) + Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }
    }
}

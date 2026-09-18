using System.Diagnostics;
using HartsyInference.Core.Logging;

namespace HartsyInference.Gpu;

/// <summary>Per-op host-side timing for a GPU backend: how long each <see cref="IBackend"/> op took from entry to
/// exit, how many dispatches it issued, and where the time went.
///
/// <para>Wall-clock deliberately, not GPU timestamps. It includes shader time AND any host wait, which is the point
/// — the recurring failure on these backends is an op that looks cheap because its kernel is cheap, while every call
/// pays a device sync and a re-upload around it. A GPU timer cannot see that; this can. When the question is
/// specifically how long a kernel ran, use the backend's own GPU timer instead.</para>
///
/// <para>Both GPU backends want this and only one had it, so the pipelines that call
/// <c>ResetOpProfile</c>/<c>DumpOpProfile</c> produced nothing at all on the other. Costs a branch when disabled.</para></summary>
public sealed class OpProfile
{
    private readonly Dictionary<string, OpStats> _stats = new(StringComparer.Ordinal);
    private long _totalTicks;
    private long _totalOps;

    /// <summary>Whether anything is being recorded. Set once by the backend from its own diagnostics knob.</summary>
    public bool IsEnabled { get; init; }

    /// <summary>Where a dump goes when it is not stderr; null for stderr.</summary>
    public string? OutputPath { get; init; }

    /// <summary>Records one completed op. <paramref name="opName"/> is expected to be a compile-time constant — the
    /// caller's member name — so it interns and this allocates nothing per call.</summary>
    public void Record(string opName, long elapsedTicks, int dispatches)
    {
        if (!IsEnabled)
        {
            return;
        }
        if (!_stats.TryGetValue(opName, out OpStats? stats))
        {
            stats = new OpStats();
            _stats[opName] = stats;
        }
        stats.Count++;
        stats.TotalTicks += elapsedTicks;
        stats.TotalDispatches += dispatches;
        if (elapsedTicks > stats.MaxTicks)
        {
            stats.MaxTicks = elapsedTicks;
        }
        _totalTicks += elapsedTicks;
        _totalOps++;
    }

    /// <summary>Discards everything recorded so far, so a caller can measure one phase rather than a whole run.</summary>
    public void Reset()
    {
        _stats.Clear();
        _totalTicks = 0;
        _totalOps = 0;
    }

    /// <summary>Writes the ops that accounted for the most time, ordered by total. <paramref name="label"/> names the
    /// phase being reported, so several dumps in one run can be told apart.</summary>
    public void Dump(string label, int topN = 20)
    {
        if (!IsEnabled || _totalOps == 0)
        {
            return;
        }

        TextWriter writer = Console.Error;
        StreamWriter? file = null;
        if (!string.IsNullOrEmpty(OutputPath))
        {
            try
            {
                file = new StreamWriter(OutputPath, append: true);
                writer = file;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
            {
                Logs.Warning($"OpProfile: cannot write {OutputPath}, using stderr instead: {ex.Message}");
            }
        }

        try
        {
            double totalMs = TicksToMs(_totalTicks);
            writer.WriteLine();
            writer.WriteLine($"=== op profile: {label} ===");
            writer.WriteLine($"  {_totalOps:N0} ops, {totalMs:F1}ms host-wall");
            writer.WriteLine($"{"Op",-32} {"Count",8} {"Total(ms)",12} {"Avg(ms)",10} {"Max(ms)",10} {"Dispatches",12} {"%",6}");
            writer.WriteLine(new string('-', 96));

            foreach ((string op, OpStats stats) in _stats.OrderByDescending(entry => entry.Value.TotalTicks).Take(topN))
            {
                double total = TicksToMs(stats.TotalTicks);
                double percent = totalMs > 0 ? 100.0 * total / totalMs : 0;
                writer.WriteLine(
                    $"{op,-32} {stats.Count,8:N0} {total,12:F1} {total / stats.Count,10:F2} "
                    + $"{TicksToMs(stats.MaxTicks),10:F2} {stats.TotalDispatches,12:N0} {percent,5:F1}%");
            }
            writer.WriteLine();
        }
        finally
        {
            file?.Dispose();
        }
    }

    private static double TicksToMs(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

    private sealed class OpStats
    {
        public long Count;
        public long TotalTicks;
        public long MaxTicks;
        public long TotalDispatches;
    }
}

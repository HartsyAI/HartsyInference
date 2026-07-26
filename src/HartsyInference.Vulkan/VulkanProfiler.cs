using System.Diagnostics;
using HartsyInference.Core.Logging;

namespace HartsyInference.Vulkan;

/// <summary>Per-op host-side timing for the Vulkan backend, enabled via <c>HARTSYINFERENCE_VK_PROFILE=1</c>.</summary>
/// <remarks>Captures wall-clock from <c>EnterOp</c> to <c>OpScope.Dispose</c> for each public IBackend op plus
/// the dispatch count. Wall-clock includes GPU shader time and any host wait (the per-op
/// <c>WaitIdleHost</c>) — Phase C1's question is whether host-wait dominates. On backend disposal dumps a
/// top-N table to stderr (or to <c>HARTSYINFERENCE_VK_PROFILE_FILE</c>). Cheap when disabled.</remarks>
internal sealed class VulkanProfiler
{
    private readonly bool _enabled;
    private readonly Dictionary<string, OpStats> _stats = new();
    private long _totalElapsedTicks;
    private int _totalOps;

    public bool IsEnabled => _enabled;

    public VulkanProfiler()
    {
        _enabled = Environment.GetEnvironmentVariable("HARTSYINFERENCE_VK_PROFILE") == "1";
    }

    public void Record(string opName, long elapsedTicks, int dispatches)
    {
        if (!_enabled) return;
        if (!_stats.TryGetValue(opName, out OpStats? s))
        {
            s = new OpStats();
            _stats[opName] = s;
        }
        s.Count++;
        s.TotalTicks += elapsedTicks;
        s.TotalDispatches += dispatches;
        if (elapsedTicks > s.MaxTicks) s.MaxTicks = elapsedTicks;
        _totalElapsedTicks += elapsedTicks;
        _totalOps++;
    }

    /// <summary>Dumps the top 20 ops by total time. Output goes to stderr unless
    /// <c>HARTSYINFERENCE_VK_PROFILE_FILE</c> points to a writable path.</summary>
    public void Dump()
    {
        if (!_enabled || _totalOps == 0) return;

        TextWriter writer = Console.Error;
        string? path = Environment.GetEnvironmentVariable("HARTSYINFERENCE_VK_PROFILE_FILE");
        StreamWriter? fileWriter = null;
        if (!string.IsNullOrEmpty(path))
        {
            try { fileWriter = new StreamWriter(path); writer = fileWriter; }
            catch (Exception ex) { Logs.Warning($"VulkanProfiler: failed to open profile output file {path}, falling back to stderr: {ex.Message}"); }
        }

        try
        {
            double totalMs = TicksToMs(_totalElapsedTicks);
            writer.WriteLine();
            writer.WriteLine("=== VulkanBackend op profile ===");
            writer.WriteLine($"  Total ops: {_totalOps:N0}, total host-wall: {totalMs:F1}ms");
            writer.WriteLine($"{"Op",-32} {"Count",8} {"Total(ms)",12} {"Avg(ms)",10} {"Max(ms)",10} {"Dispatches",12} {"%",6}");
            writer.WriteLine(new string('-', 96));

            foreach (KeyValuePair<string, OpStats> kvp in _stats
                .OrderByDescending(p => p.Value.TotalTicks)
                .Take(20))
            {
                OpStats s = kvp.Value;
                double total = TicksToMs(s.TotalTicks);
                double avg = total / s.Count;
                double max = TicksToMs(s.MaxTicks);
                double pct = totalMs > 0 ? 100.0 * total / totalMs : 0;
                writer.WriteLine($"{kvp.Key,-32} {s.Count,8:N0} {total,12:F1} {avg,10:F2} {max,10:F2} {s.TotalDispatches,12:N0} {pct,5:F1}%");
            }
            writer.WriteLine();
        }
        finally
        {
            fileWriter?.Dispose();
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

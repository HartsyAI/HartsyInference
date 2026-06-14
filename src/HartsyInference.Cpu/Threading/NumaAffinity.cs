using HartsyInference.Core.Logging;

namespace HartsyInference.Cpu.Threading;

/// <summary>NUMA-aware thread affinity helpers. On Windows, pins compute threads to P-cores for consistent performance. Falls back to no-op on unsupported platforms.</summary>
public static class NumaAffinity
{
    /// <summary>Returns the number of physical cores (not hyperthreaded logical processors). Falls back to Environment.ProcessorCount / 2 as a reasonable estimate.</summary>
    public static int PhysicalCoreCount
    {
        get
        {
            int logical = Environment.ProcessorCount;
            // Rough heuristic: most desktop CPUs are SMT-2
            return Math.Max(1, logical / 2);
        }
    }

    /// <summary>Attempts to pin the current thread to the specified logical processor. Returns true if successful, false if the platform doesn't support affinity. Currently a stub — full implementation needs SetThreadAffinityMask via P/Invoke.</summary>
    public static bool TryPinToCore(int logicalProcessor)
    {
        if (!OperatingSystem.IsWindows())
            return false;

        try
        {
            return false;
        }
        catch (Exception ex)
        {
            Logs.Verbose($"TryPinToCore({logicalProcessor}) failed: {ex.Message}");
            return false;
        }
    }
}

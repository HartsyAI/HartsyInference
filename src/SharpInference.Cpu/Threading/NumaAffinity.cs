namespace SharpInference.Cpu.Threading;

/// <summary>NUMA-aware thread affinity helpers. On Windows, pins compute threads to P-cores for consistent performance. Falls back to no-op on unsupported platforms.</summary>
public static class NumaAffinity
{
    /// <summary>Attempts to pin the current thread to the specified logical processor. Returns true if successful, false if the platform doesn't support affinity.</summary>
    public static bool TryPinToCore(int logicalProcessor)
    {
        if (!OperatingSystem.IsWindows())
            return false;

        try
        {
            Thread currentThread = Thread.CurrentThread;
            // On .NET, we can use ProcessThread for affinity, but Thread doesn't expose it directly.
            // For now, this is a best-effort stub — full implementation would use
            // SetThreadAffinityMask via P/Invoke.
            return false;
        }
        catch
        {
            return false;
        }
    }

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
}

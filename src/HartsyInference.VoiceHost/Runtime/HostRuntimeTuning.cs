using System.Runtime;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Numerics;

namespace HartsyInference.VoiceHost.Runtime;

/// <summary>Process-wide settings that keep the GC and the thread pool out of a live call's way. The csproj selects
/// workstation concurrent GC; this adds <see cref="GCLatencyMode.SustainedLowLatency"/> (background collections only, no
/// blocking gen2) while at least one call is up, and a floor of pool threads so CPU kernels cannot starve the async turn
/// loop.</summary>
/// <remarks>The floor is <c>numerics.cpuThreads + </c><see cref="PoolHeadroom"/>. <see cref="CpuParallel"/> runs its
/// fan-out on the shared pool, capped across all callers at <c>numerics.cpuThreads</c> workers, so a synthesis or a
/// decode can hold that many pool threads at once. The host's own asynchronous work (each call's turn loop and event
/// pump, tool calls waiting on the gateway, the GPU thread's job continuations, socket completions) needs a few more
/// runnable at the same moment. Above the floor the pool adds threads by hill climbing, at most one every few hundred
/// milliseconds, which a live turn would hear as a stall; eight covers two calls' worth of that work with room to
/// spare.</remarks>
internal static class HostRuntimeTuning
{
    /// <summary>Pool threads kept beyond what CPU kernels can occupy.</summary>
    public const int PoolHeadroom = 8;

    private static readonly object _lock = new();
    private static int _activeCalls;
    private static GCLatencyMode _idleMode;

    /// <summary>Calls currently holding the low-latency GC mode.</summary>
    public static int ActiveCalls
    {
        get
        {
            lock (_lock)
            {
                return _activeCalls;
            }
        }
    }

    /// <summary>Raises the pool's minimum worker threads to the kernel threads plus <see cref="PoolHeadroom"/> and logs
    /// the GC configuration. Returns the minimum now in force.</summary>
    /// <param name="cpuThreadCap">The host's <c>engine.cpuThreadCap</c>; 0 means whatever <c>numerics.cpuThreads</c>
    /// resolves to.</param>
    public static int ApplyThreadPool(int cpuThreadCap)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(cpuThreadCap);
        int kernelThreads = cpuThreadCap > 0 ? Math.Min(cpuThreadCap, Environment.ProcessorCount) : CpuParallel.MaxThreads;
        ThreadPool.GetMinThreads(out int workers, out int completionPorts);
        int wanted = kernelThreads + PoolHeadroom;
        if (wanted > workers)
        {
            ThreadPool.SetMinThreads(wanted, completionPorts);
            workers = wanted;
        }
        string gen0 = GC.GetConfigurationVariables().TryGetValue("GCGen0MaxBudget", out object? budget) && budget is long bytes
            ? $"{bytes / (1024 * 1024)} MB"
            : "default";
        Logs.Info($"[VoiceHost] GC: server={GCSettings.IsServerGC} concurrent={GCSettings.LatencyMode != GCLatencyMode.Batch} gen0 budget={gen0}; "
            + $"pool min worker threads {workers} (CPU kernel threads {kernelThreads} + {PoolHeadroom}); {Environment.ProcessorCount} CPUs allowed.");
        if (GCSettings.IsServerGC)
        {
            Logs.Warning("[VoiceHost] Server GC is on; its blocking collections stall the 20 ms sender. Build the host with ServerGarbageCollection=false.");
        }
        return workers;
    }

    /// <summary>A call started: the first one switches the GC to sustained low latency.</summary>
    public static void CallStarted()
    {
        lock (_lock)
        {
            if (_activeCalls++ == 0)
            {
                _idleMode = GCSettings.LatencyMode;
                GCSettings.LatencyMode = GCLatencyMode.SustainedLowLatency;
            }
        }
    }

    /// <summary>A call ended: the last one restores the mode the process had before.</summary>
    public static void CallEnded()
    {
        lock (_lock)
        {
            if (_activeCalls == 0)
            {
                return;
            }
            if (--_activeCalls == 0)
            {
                GCSettings.LatencyMode = _idleMode;
            }
        }
    }
}

using System.Runtime;
using HartsyInference.Core.Logging;

namespace HartsyInference.PhoneGateway.Runtime;

/// <summary>Process-wide settings that keep the GC and the pool out of the 20 ms cadence's way. The csproj already
/// selects workstation concurrent GC; this adds <see cref="GCLatencyMode.SustainedLowLatency"/> (background
/// collections only, no blocking gen2) and a floor of pool threads so sipsorcery's per-packet callbacks never wait
/// for pool growth. The gen0 budget is a runtime setting (<c>DOTNET_GCgen0size</c>) the systemd unit sets; the
/// budget the GC actually uses is logged.</summary>
public static class RuntimeTuning
{
    private const int MinPoolThreads = 8;

    public static void Apply()
    {
        GCSettings.LatencyMode = GCLatencyMode.SustainedLowLatency;
        ThreadPool.SetMinThreads(MinPoolThreads, MinPoolThreads);
        string gen0 = GC.GetConfigurationVariables().TryGetValue("GCGen0MaxBudget", out object? budget) && budget is long bytes
            ? $"{bytes / (1024 * 1024)} MB"
            : "default";
        Logs.Info($"[PhoneGateway] GC: server={GCSettings.IsServerGC} latency={GCSettings.LatencyMode} gen0 budget={gen0}; pool min threads {MinPoolThreads}.");
        if (GCSettings.IsServerGC)
        {
            Logs.Warning("[PhoneGateway] Server GC is on; its blocking collections will stall the RTP clock. Build the gateway with ServerGarbageCollection=false.");
        }
    }
}

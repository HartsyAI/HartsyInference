namespace HartsyInference.Core.Moe.Telemetry;

/// <summary>Running totals for one layer. Routed always equals Hits plus Misses.</summary>
public readonly record struct LayerRoutingCounters(long Routed, long Hits, long Misses, long BytesMoved, long CpuRuns, long GpuRuns)
{
    /// <summary>Hits divided by routed accesses; zero when nothing was routed.</summary>
    public double HitRate => Routed == 0 ? 0.0 : (double)Hits / Routed;
}

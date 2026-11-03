namespace HartsyInference.Engine;

/// <summary>Decides which cached pipelines a model switch may evict. Pure, so the rules run without a device.</summary>
internal static class PipelineEvictionPolicy
{
    /// <summary>A resident variant needs more device memory than its checkpoint file holds (activations and sidecars).</summary>
    private const double ResidentFactor = 1.5;

    /// <summary>True when the free device memory leaves room for another variant of a checkpoint of this size.</summary>
    public static bool FitsBesideCached(long freeBytes, long checkpointBytes) =>
        checkpointBytes <= 0 || freeBytes >= (long)(checkpointBytes * ResidentFactor);

    /// <summary>Keys a switch may evict. Same-checkpoint variants stay cached unless memory is tight. The
    /// <paramref name="alsoKeepPrefix"/> survivor is never evicted.</summary>
    public static List<string> SelectVictims(IEnumerable<string> cachedKeys, string keepPrefix, string? alsoKeepPrefix, bool evictSameCheckpoint)
    {
        List<string> victims = new();
        foreach (string key in cachedKeys)
        {
            if (alsoKeepPrefix is not null && key.StartsWith(alsoKeepPrefix, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            if (!evictSameCheckpoint && key.StartsWith(keepPrefix, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            victims.Add(key);
        }
        return victims;
    }
}

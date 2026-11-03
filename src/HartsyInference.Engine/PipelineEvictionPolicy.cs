namespace HartsyInference.Engine;

/// <summary>Decides which cached pipelines a model switch may evict. Pure, so the rules run without a device.</summary>
internal static class PipelineEvictionPolicy
{
    /// <summary>Headroom on a measured resident size: a cached variant still needs its activations, which a measurement does not include.</summary>
    private const double ActivationHeadroom = 1.25;

    /// <summary>The estimate before any variant of the checkpoint has been measured: the file, plus activations and sidecars. It is only
    /// a fallback. A recipe that promotes its weights to a larger resident form than the file is measured instead.</summary>
    private const double FileSizeFactor = 1.5;

    /// <summary>The device bytes another variant of a checkpoint needs beside a cached one. A measurement is taken around the
    /// construction and can miss the weights the first preload places later, so the checkpoint file is its floor.</summary>
    public static long EstimateVariantBytes(long measuredResidentBytes, long checkpointBytes) =>
        measuredResidentBytes > 0
            ? (long)(Math.Max(measuredResidentBytes, checkpointBytes) * ActivationHeadroom)
            : (long)(checkpointBytes * FileSizeFactor);

    /// <summary>True when the free device memory leaves room for another variant at <paramref name="estimatedBytes"/>. An unknown
    /// estimate always fits.</summary>
    public static bool FitsBesideCached(long freeBytes, long estimatedBytes) =>
        estimatedBytes <= 0 || freeBytes >= estimatedBytes;

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

using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Requests;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Checks a reference key/value cache against the device's free memory. It only warns: switching to the joint
/// layout on its own would change what a cache-trained adapter sees.</summary>
public static class ReferenceCachePolicy
{
    /// <summary>True when <paramref name="cacheCount"/> caches of <paramref name="bytesPerCache"/> fit the budget (always
    /// for a backend that reports no device memory, such as the CPU); otherwise logs a warning naming the remedies.</summary>
    public static bool Check(IBackend backend, ReferenceConditioningOptions options, long bytesPerCache, int cacheCount, string family)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(options);
        if (options.Mode != ReferenceAttentionMode.CachedKv)
            return true;
        long free = backend.FreeMemoryBytes();
        long total = bytesPerCache * Math.Max(1, cacheCount);
        if (free <= 0 || total <= free * (double)options.CacheBudgetFraction)
            return true;
        Logs.Warning($"{family}: {cacheCount} reference key/value cache(s) need about {total / (1024 * 1024)} MiB against "
            + $"{free / (1024 * 1024)} MiB free. Use smaller or fewer reference images, or CFG 1 to halve the caches.");
        return false;
    }
}

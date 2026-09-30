namespace HartsyInference.Core.Backends;

/// <summary>A snapshot of expert cache activity; counters are lifetime totals, the rest is current.</summary>
/// <param name="Hits">Requested experts that were already resident.</param>
/// <param name="InFlightHits">Hits that shared an upload a prefetch had started (also counted in <paramref name="Hits"/>).</param>
/// <param name="Misses">Requested experts that had to be uploaded.</param>
/// <param name="Prefetches">Uploads started by <see cref="IExpertCache.Prefetch"/>.</param>
/// <param name="Evictions">Experts removed from the device.</param>
/// <param name="BytesUploaded">Bytes enqueued for upload, misses and prefetches.</param>
/// <param name="ResidentBytes">Bytes currently resident or uploading.</param>
/// <param name="PinnedBytes">Bytes held by unreleased leases.</param>
/// <param name="ResidentExperts">Experts currently resident or uploading.</param>
/// <param name="PinnedExperts">Experts held by unreleased leases.</param>
/// <param name="BudgetBytes">The byte budget the cache stays within.</param>
public readonly record struct ExpertCacheStats(
    long Hits, long InFlightHits, long Misses, long Prefetches, long Evictions, long BytesUploaded,
    long ResidentBytes, long PinnedBytes, int ResidentExperts, int PinnedExperts, long BudgetBytes);

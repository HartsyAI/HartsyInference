namespace HartsyInference.LLM.Generation;

/// <summary>Device memory headroom at one instant; a minimal shape that later slices extend with state-pool figures.</summary>
/// <param name="FreeBytes">Bytes the output device reports free, 0 when the backend cannot report.</param>
/// <param name="TotalBytes">Bytes the output device reports in total, 0 when the backend cannot report.</param>
public readonly record struct CapacitySnapshot(long FreeBytes, long TotalBytes);

namespace HartsyInference.Core.Backends;

/// <summary>
/// Optional residency capability of an expert cache. Kept separate from <see cref="IExpertCache"/> so implementers of the
/// published interface are not broken; a scheduler checks for it before splitting a batch.
/// </summary>
public interface IResidencyAwareExpertCache : IExpertCache
{
    /// <summary>
    /// Reports which keys are resident or already uploading, without changing any state. Returns how many are.
    /// <paramref name="resident"/> must hold one entry per key.
    /// </summary>
    int LookupResident(ReadOnlySpan<ExpertKey> keys, Span<bool> resident);

    /// <summary>
    /// Pins only the experts already resident or uploading, and never uploads or resolves anything. Keys that are not resident
    /// are appended to <paramref name="misses"/> (only when the call succeeds), so the caller can run them elsewhere.
    /// </summary>
    ExpertLease AcquireResident(ReadOnlySpan<ExpertKey> keys, List<ExpertKey> misses);
}

using HartsyInference.Core.Backends;

namespace HartsyInference.Cuda;

/// <summary>Expert cache on the backend's <see cref="CudaStreamingWeightCache"/>: one expert is one <c>BeginUploadAsync</c> of its weights and scales through the pinned staging ring, on the upload stream.</summary>
/// <remarks>Expert tensors are excluded from the ≥1 MB auto-promotion, so an evicted expert is never resurrected as a permanent weight.
/// Disposing drains both streams, evicts every expert and returns the staging ring.</remarks>
public sealed class CudaExpertCache : ExpertCacheBase
{
    private readonly CudaStreamingWeightCache _streaming;
    private readonly bool _previousPinUploadSource;

    /// <summary>Creates a cache of at most <paramref name="budgetBytes"/> resident expert bytes on <paramref name="backend"/>'s device.</summary>
    /// <param name="stagingSlots">Pinned staging slots, one per upload that can be in flight before the host waits; at least the prefetch depth.</param>
    public CudaExpertCache(CudaBackend backend, long budgetBytes, IEnumerable<ExpertBank>? banks = null, int stagingSlots = 4)
        : base(budgetBytes, banks)
    {
        ArgumentNullException.ThrowIfNull(backend);
        _streaming = backend.StreamingCache as CudaStreamingWeightCache
            ?? throw new InvalidOperationException("The backend has no CUDA streaming weight cache.");
        _previousPinUploadSource = _streaming.PinUploadSource;
        _streaming.PinUploadSource = true;
        _streaming.StagingSlotCount = Math.Max(_streaming.StagingSlotCount, stagingSlots);
    }

    /// <inheritdoc/>
    protected override object? BeginUpload(ExpertWeights weights)
    {
        _streaming.ExcludeFromAutoPromotion(weights.Tensors);
        StreamingUploadToken token = _streaming.BeginUploadAsync(weights.Tensors);
        return token.IsEmpty ? null : token;
    }

    /// <inheritdoc/>
    protected override void AwaitUpload(object pending) => _streaming.AwaitWeights((StreamingUploadToken)pending);

    /// <inheritdoc/>
    protected override void Evict(ExpertWeights weights) => _streaming.EvictAsync(weights.Tensors);

    /// <inheritdoc/>
    protected override object RecordFence() => _streaming.RecordComputeFence();

    /// <inheritdoc/>
    protected override bool IsFenceDone(object fence) => _streaming.IsFenceDone((nint)fence);

    /// <inheritdoc/>
    protected override void WaitFence(object fence) => _streaming.WaitFence((nint)fence);

    /// <inheritdoc/>
    protected override void DestroyFence(object fence) => _streaming.DestroyFence((nint)fence);

    /// <inheritdoc/>
    protected override void Drain()
    {
        _streaming.DrainAndReleasePool();
        _streaming.UnregisterPinnedSources();
        _streaming.PinUploadSource = _previousPinUploadSource;
    }
}

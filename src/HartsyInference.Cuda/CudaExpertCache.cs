using HartsyInference.Core.Backends;

namespace HartsyInference.Cuda;

/// <summary>Expert cache on the backend's <see cref="CudaStreamingWeightCache"/>: one expert is one <c>BeginUploadAsync</c> of its weights and scales through the pinned staging ring, on the upload stream.</summary>
/// <remarks>Source pinning stays off: experts are mmap-backed and must reach the device through the pinned staging ring, never host-registration.
/// Expert tensors are excluded from the ≥1 MB auto-promotion, so an evicted expert is never resurrected as a permanent weight.
/// One cache per backend: it changes streaming-cache settings that it restores on dispose. Disposing drains both streams, evicts every expert and returns the staging ring.</remarks>
public sealed class CudaExpertCache : ExpertCacheBase
{
    private readonly CudaStreamingWeightCache _streaming;
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<CudaStreamingWeightCache, object> Active = [];

    private readonly bool _previousPinUploadSource;
    private readonly int _previousStagingSlots;

    /// <summary>Creates a cache of at most <paramref name="budgetBytes"/> resident expert bytes on <paramref name="backend"/>'s device.</summary>
    /// <param name="stagingSlots">Pinned staging slots, one per upload that can be in flight before the host waits; at least the prefetch depth.</param>
    public CudaExpertCache(CudaBackend backend, long budgetBytes, IEnumerable<ExpertBank>? banks = null, int stagingSlots = 4)
        : base(budgetBytes, banks)
    {
        ArgumentNullException.ThrowIfNull(backend);
        _streaming = backend.StreamingCache as CudaStreamingWeightCache
            ?? throw new InvalidOperationException("The backend has no CUDA streaming weight cache.");
        if (!Active.TryAdd(_streaming, this))
            throw new InvalidOperationException("A CudaExpertCache is already active on this backend; the streaming cache settings it changes are not shareable.");
        _previousPinUploadSource = _streaming.PinUploadSource;
        _previousStagingSlots = _streaming.StagingSlotCount;
        try
        {
            _streaming.PinUploadSource = false;
            _streaming.StagingSlotCount = Math.Max(_streaming.StagingSlotCount, stagingSlots);
        }
        catch
        {
            RestoreStreamingSettings();
            throw;
        }
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
    protected override void AbandonUpload(object pending) => _streaming.SynchronizeUploads();

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
        try { _streaming.DrainAndReleasePool(); }
        finally { RestoreStreamingSettings(); }
    }

    private void RestoreStreamingSettings()
    {
        try
        {
            _streaming.PinUploadSource = _previousPinUploadSource;
            _streaming.StagingSlotCount = _previousStagingSlots;
        }
        finally { Active.Remove(_streaming); }
    }
}

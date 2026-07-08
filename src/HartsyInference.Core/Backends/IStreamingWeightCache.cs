using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Backend capability for streaming weight tensors to device memory on a side stream while compute runs on the main stream. Enables ComfyUI-style "lowvram" per-block offloading. All methods are single-threaded and must be called from the backend's owning thread.</summary>
public interface IStreamingWeightCache
{
    /// <summary>Queues an async upload of <paramref name="weights"/> to device memory and returns a token that must be passed to <see cref="AwaitWeights"/> before any op reads them. Already-cached tensors are silently skipped.</summary>
    StreamingUploadToken BeginUploadAsync(IEnumerable<Tensor> weights);

    /// <summary>Inserts a wait barrier on the compute stream gating subsequent ops on <paramref name="token"/>'s upload. Host thread does not block; only the GPU compute stream is gated. Disposes the token's underlying event — token must not be reused.</summary>
    void AwaitWeights(StreamingUploadToken token);

    /// <summary>Asynchronously frees device memory for <paramref name="weights"/> on the compute stream so the free is ordered after any prior op that read them. Tensors not currently cached are silently skipped.</summary>
    void EvictAsync(IEnumerable<Tensor> weights);

    /// <summary>Returns bytes available for the weight cache after reserving <paramref name="activationReserve"/> for in-flight activations and workspace.</summary>
    long QueryAvailableWeightCacheBytes(long activationReserve);

    /// <summary>Drains in-flight uploads and releases backend-internal pool reservations to the driver. Required between streaming and eager-allocation phases on CUDA — the stream-ordered allocator pool is invisible to subsequent sync allocations until trimmed. No-op on backends without a stream-ordered allocator.</summary>
    void DrainAndReleasePool();

    /// <summary>Opt-in: page-lock weight host sources before upload so the async H2D truly overlaps with compute — a pageable source silently degrades to a synchronous staged copy that overlaps with nothing. Worth it only when weights are re-uploaded across steps (block-swap); registration cost is one-time per source. Default no-op for backends without pinning.</summary>
    bool PinUploadSource { get => false; set { } }
}

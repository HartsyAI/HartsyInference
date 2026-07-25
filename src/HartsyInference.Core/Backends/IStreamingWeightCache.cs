using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Backend capability for streaming weight tensors to device memory on a side stream while compute runs (ComfyUI-style "lowvram").</summary>
public interface IStreamingWeightCache
{
    /// <summary>Queues an async weight upload; returns a token to pass to <see cref="AwaitWeights"/> before any op reads them.</summary>
    StreamingUploadToken BeginUploadAsync(IEnumerable<Tensor> weights);

    /// <summary>Inserts a wait barrier gating subsequent compute ops on the token's upload; disposes the token's event (not reusable).</summary>
    void AwaitWeights(StreamingUploadToken token);

    /// <summary>Asynchronously frees device memory for weights on the compute stream, ordered after any prior op that read them.</summary>
    void EvictAsync(IEnumerable<Tensor> weights);

    /// <summary>Returns bytes available for the weight cache after reserving activationReserve for in-flight activations and workspace.</summary>
    long QueryAvailableWeightCacheBytes(long activationReserve);

    /// <summary>Drains in-flight uploads and releases pool reservations; required between streaming and eager-allocation phases on CUDA.</summary>
    void DrainAndReleasePool();

    /// <summary>Opt-in: page-lock weight sources before upload so async H2D overlaps compute; worth it for block-swap re-uploads.</summary>
    bool PinUploadSource { get => false; set { } }
}

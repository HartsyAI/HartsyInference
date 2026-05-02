using SharpInference.Core.Tensors;

namespace SharpInference.Core.Backends;

/// <summary>
/// Optional capability surface for backends that can stream weight tensors to
/// device memory on a side stream while compute runs on the main stream. The
/// per-block / per-layer offloading pattern (the same one ComfyUI calls "lowvram")
/// is built on top of this.
///
/// <para><b>Lifecycle of a single weight batch:</b></para>
/// <list type="number">
/// <item><description><see cref="BeginUploadAsync"/> — alloc + memcpy queued on the
/// upload stream. Returns immediately with a token; the upload is in flight.</description></item>
/// <item><description><see cref="AwaitWeights"/> — gates the compute stream on the
/// upload's completion event. Cheap if the upload already finished, otherwise the
/// next compute op queued after this call will wait. The host thread does not block.</description></item>
/// <item><description>Caller runs ops that read the uploaded tensors.</description></item>
/// <item><description><see cref="EvictAsync"/> — frees device memory once the compute
/// stream's prior work (which read these tensors) finishes. Async on the compute
/// stream so it can't free tensors that an in-flight kernel still references.</description></item>
/// </list>
///
/// <para><b>Threading:</b> All methods must be called from the same thread that
/// constructed the backend (Driver-API context affinity). The upload stream itself
/// runs on the GPU asynchronously — that's the point — but the API surface above
/// it is single-threaded.</para>
///
/// <para><b>Backends that can't stream</b> (CPU has no concept of device memory;
/// Vulkan's allocation model is different) return <c>null</c> from
/// <see cref="IBackend.StreamingCache"/>. Consumers should fall back to the eager
/// <see cref="IBackend.PreloadWeights"/> + <see cref="IBackend.FreeWeights"/> path.</para>
/// </summary>
public interface IStreamingWeightCache
{
    /// <summary>
    /// Queues an asynchronous upload of <paramref name="weights"/> to device memory.
    /// Returns immediately. The returned token must be passed to <see cref="AwaitWeights"/>
    /// before any op reads the uploaded tensors. Tensors that are already cached are
    /// silently skipped (free hit).
    /// </summary>
    StreamingUploadToken BeginUploadAsync(IEnumerable<Tensor> weights);

    /// <summary>
    /// Inserts a wait barrier on the compute stream so it doesn't process subsequent
    /// ops until <paramref name="token"/>'s upload completes. The host thread does not
    /// block — only the GPU's compute stream is gated. Disposes the token's underlying
    /// event resource; the token must not be reused.
    /// </summary>
    void AwaitWeights(StreamingUploadToken token);

    /// <summary>
    /// Asynchronously frees the device memory backing <paramref name="weights"/> on the
    /// compute stream, so the free is ordered after any prior op that read these tensors.
    /// Tensors that aren't currently cached are silently skipped.
    /// </summary>
    void EvictAsync(IEnumerable<Tensor> weights);

    /// <summary>
    /// Returns the number of bytes currently available for the weight cache, after
    /// reserving <paramref name="activationReserve"/> for in-flight activation tensors
    /// and workspace allocations. The result is the budget the streaming controller
    /// should target for its resident working set.
    /// </summary>
    long QueryAvailableWeightCacheBytes(long activationReserve);
}

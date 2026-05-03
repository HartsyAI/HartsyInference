using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Cuda;

/// <summary>
/// CUDA implementation of <see cref="IStreamingWeightCache"/>. Uploads weight
/// tensors on a dedicated <em>upload stream</em>, gates the compute stream on
/// completion via CUDA events, and frees on the compute stream so reclamation
/// is naturally ordered after any prior reads of the cached memory.
///
/// <para><b>Why a separate stream:</b> on a single stream every op is serialised,
/// so an H2D copy blocks the kernel that would have run alongside it. Putting the
/// upload on its own stream lets the GPU's copy engine work in parallel with the
/// compute SMs — the same parallelism Comfy gets via PyTorch's <c>non_blocking=True</c>
/// + offload-stream pattern. On a 3060 a Flux block's weights (~150 MB) take
/// ~5 ms over PCIe Gen4 x16; a block's compute takes ~30-80 ms, so prefetching
/// one block ahead fully hides the transfer.</para>
///
/// <para><b>State sharing:</b> uploaded weights are registered in
/// <see cref="GpuTransferHelper"/>'s shared <c>_weightCache</c> so the existing
/// <c>CopyToDevice</c> fast path on every <see cref="CudaBackend"/> op finds them
/// and reuses the cached dptr — the streaming cache is just a different way to
/// populate the same cache. <see cref="EvictAsync"/> removes them.</para>
/// </summary>
public sealed class CudaStreamingWeightCache : IStreamingWeightCache
{
    private readonly CudaContext _context;
    private readonly nint _computeStream;
    private readonly nint _uploadStream;

    /// <summary>Constructs a streaming cache bound to a compute stream and upload
    /// stream. Both streams should be created with <c>CU_STREAM_NON_BLOCKING</c>
    /// so they can run independently of the legacy NULL stream and of each other.</summary>
    public CudaStreamingWeightCache(CudaContext context, nint computeStream, nint uploadStream)
    {
        if (context is null) throw new ArgumentNullException(nameof(context));
        if (computeStream == 0) throw new ArgumentException("Compute stream handle must be non-zero.", nameof(computeStream));
        if (uploadStream == 0) throw new ArgumentException("Upload stream handle must be non-zero.", nameof(uploadStream));
        if (computeStream == uploadStream)
        {
            throw new ArgumentException(
                "Compute and upload streams must be distinct — sharing them serializes uploads " +
                "with compute and defeats the entire purpose of streaming.", nameof(uploadStream));
        }
        _context = context;
        _computeStream = computeStream;
        _uploadStream = uploadStream;

        // Configure the device's default mempool to be aggressive about returning
        // freed memory to the driver. Without this, the pool can hold many GB of
        // reserved memory across cuMemFreeAsync calls — invisible to subsequent
        // sync cuMemAlloc requests, manifesting as OOM mid-streaming. The default
        // is supposed to be 0 but at least one Linux driver leaves it at "infinite"
        // unless we set it explicitly. Stream syncs (or explicit trim) then return
        // the bytes to the driver immediately.
        context.EnsureCurrent();
        CudaDriverApi.cuDeviceGetDefaultMemPool(out nint pool, context.DeviceOrdinal).ThrowOnError();
        unsafe
        {
            ulong releaseThreshold = 0;
            CudaDriverApi.cuMemPoolSetAttribute(pool, CudaDriverApi.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &releaseThreshold).ThrowOnError();
        }
    }

    /// <inheritdoc/>
    public StreamingUploadToken BeginUploadAsync(IEnumerable<Tensor> weights)
    {
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        _context.EnsureCurrent();

        bool anyUploaded = false;
        foreach (Tensor weight in weights)
        {
            if (GpuTransferHelper.IsWeightCached(weight))
            {
                continue; // Free hit — the dptr is already valid in _weightCache.
            }

            nuint byteSize = GpuTransferHelper.ByteSize(weight);
            // Use sync cuMemAlloc (microseconds, host-side) instead of cuMemAllocAsync
            // because the stream-ordered allocator requires mempool initialization
            // that isn't reliably present on the primary context. cuMemcpyHtoDAsync
            // on the upload stream still gives us the overlap-with-compute behavior
            // we need; the alloc itself was never the bottleneck. Eviction uses
            // cuMemFreeAsync on the compute stream — works on dptrs from either alloc API.
            ulong dptr = CudaMemory.Allocate(byteSize);
            unsafe
            {
                CudaDriverApi.cuMemcpyHtoDAsync(dptr, (nint)weight.DataPointer, byteSize, _uploadStream).ThrowOnError();
            }
            // Register immediately even though the upload is in flight: ops won't
            // read these tensors until AwaitWeights gates the compute stream on
            // the completion event below, at which point the data is guaranteed
            // visible. Registering up front means a parallel BeginUploadAsync
            // call for the same tensor sees it as cached and skips re-upload.
            GpuTransferHelper.RegisterCachedWeight(weight, dptr, byteSize);
            anyUploaded = true;
        }

        if (!anyUploaded)
        {
            return StreamingUploadToken.Empty;
        }

        // Record a completion event after all the queued copies. Disabling timing
        // saves a tiny bit of driver overhead; we never read elapsed time on these.
        CudaDriverApi.cuEventCreate(out nint evt, CudaDriverApi.CU_EVENT_DISABLE_TIMING).ThrowOnError();
        CudaDriverApi.cuEventRecord(evt, _uploadStream).ThrowOnError();
        return new StreamingUploadToken(evt, this);
    }

    /// <inheritdoc/>
    public void AwaitWeights(StreamingUploadToken token)
    {
        if (token.IsEmpty)
        {
            return; // Nothing was uploaded — nothing to wait on.
        }
        if (!ReferenceEquals(token.BackendTag, this))
        {
            throw new InvalidOperationException(
                "StreamingUploadToken was issued by a different cache. Tokens are not " +
                "transferable between backend instances.");
        }
        _context.EnsureCurrent();
        // Compute stream waits until the upload event is recorded — i.e. the H2D
        // copies are visible to any kernel queued after this point. Host thread
        // does not block; only the GPU compute stream is gated.
        CudaDriverApi.cuStreamWaitEvent(_computeStream, token.Handle, CudaDriverApi.CU_EVENT_WAIT_DEFAULT).ThrowOnError();
        // Event is single-use. Destroying it now is safe even though the wait it
        // triggered may not have fired yet — the driver retains the reference
        // internally until the wait is satisfied.
        CudaDriverApi.cuEventDestroy(token.Handle).ThrowOnError();
    }

    /// <inheritdoc/>
    public void EvictAsync(IEnumerable<Tensor> weights)
    {
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        _context.EnsureCurrent();
        foreach (Tensor weight in weights)
        {
            if (GpuTransferHelper.TryUnregisterCachedWeight(weight, out ulong dptr))
            {
                // FreeAsync on the compute stream orders the free after any prior
                // op on that stream which read this tensor. cuMemFreeAsync returns
                // the memory to the stream-ordered allocator pool; the dptr is
                // not safe to reuse until the stream reaches that point.
                CudaDriverApi.cuMemFreeAsync(dptr, _computeStream).ThrowOnError();
            }
        }
    }

    /// <inheritdoc/>
    public long QueryAvailableWeightCacheBytes(long activationReserve)
    {
        if (activationReserve < 0) throw new ArgumentOutOfRangeException(nameof(activationReserve));
        _context.EnsureCurrent();
        CudaDriverApi.cuMemGetInfo(out nuint freeBytes, out _).ThrowOnError();
        long avail = (long)freeBytes - activationReserve;
        return avail < 0 ? 0 : avail;
    }

    /// <inheritdoc/>
    public void DrainAndReleasePool()
    {
        _context.EnsureCurrent();
        // Drain both streams so any queued cuMemFreeAsync calls actually return
        // their memory to the stream-ordered allocator pool. Without this the trim
        // below would only release whatever already-completed frees the pool sees.
        CudaDriverApi.cuStreamSynchronize(_uploadStream).ThrowOnError();
        CudaDriverApi.cuStreamSynchronize(_computeStream).ThrowOnError();
        // Trim the device's default mempool back to 0 reserved bytes — releases
        // the just-drained frees back to the regular driver allocator so subsequent
        // sync cuMemAlloc calls (e.g. inside the VAE) can use that memory.
        CudaDriverApi.cuDeviceGetDefaultMemPool(out nint pool, _context.DeviceOrdinal).ThrowOnError();
        CudaDriverApi.cuMemPoolTrimTo(pool, 0).ThrowOnError();
    }
}

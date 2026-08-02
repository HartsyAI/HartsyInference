using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>CUDA implementation of <see cref="IStreamingWeightCache"/>. Uploads weights on a dedicated upload stream
/// so the copy engine runs in parallel with compute SMs, gates the compute stream on completion via CUDA events,
/// and frees on the compute stream so reclamation is naturally ordered after any prior reads.</summary>
/// <remarks>Uploaded weights register in <see cref="GpuTransferHelper"/>'s shared cache so the existing
/// <see cref="CudaBackend"/> fast path reuses the cached dptr; <see cref="EvictAsync"/> removes them.</remarks>
public sealed class CudaStreamingWeightCache : IStreamingWeightCache
{
    private readonly CudaContext _context;
    private readonly nint _computeStream;
    private readonly nint _uploadStream;

    // Pinned staging ring for PinUploadSource: weights are memcpy'd into a page-locked staging buffer and uploaded
    // async from there. Registering the mmap'd weight pages directly (cuMemHostRegister) is a trap: it needs
    // page-aligned ranges, contiguous weights share boundary pages (ALREADY_REGISTERED aborts the whole call), and a
    // copy whose source straddles pinned/unpinned pages fails with INVALID_VALUE. Staging sidesteps all of it and
    // pins only ring-size host memory instead of the whole checkpoint. 3 slots cover prefetchAhead=2 + the in-flight
    // upload; each slot's event gates reuse.
    private const int StagingSlots = 3;
    private readonly nint[] _stagingPtrs = new nint[StagingSlots];
    private readonly nint[] _stagingEvents = new nint[StagingSlots];
    private readonly bool[] _stagingUsed = new bool[StagingSlots];
    private nuint _stagingBytes;
    private int _stagingIdx;
    private bool _stagingDead;   // allocation failed once — fall back to pageable direct uploads for the session

    /// <summary>Opt-in: page-lock each weight's host source before uploading so <c>cuMemcpyHtoDAsync</c> is genuinely
    /// asynchronous and overlaps with compute (pageable sources silently force a synchronous staging copy that
    /// overlaps with nothing). Defaults to <c>false</c>.</summary>
    /// <remarks>Only beneficial when weights are re-uploaded across steps (block-swap); for a one-shot preload the
    /// registration cost is not amortized. Requires the CPU weight tensors to stay resident (do not dispose them)
    /// for the lifetime of the stream.</remarks>
    public bool PinUploadSource { get; set; }

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
        // Mempool release-threshold policy is DEVICE state owned by DeviceMempoolPolicy (set once per device by
        // the first backend, refcounted) — this ctor used to force it to 0 here, which flipped a same-device
        // sibling backend's live pool into release-everything mode mid-generation.
        context.EnsureCurrent();
    }

    /// <summary>The owning backend's transfer state; bound after registration so this cache's entry points can set
    /// the ambient (two same-device backends share a context, so context identity cannot route these calls).</summary>
    private GpuTransferHelper.State? _state;

    /// <summary>Binds the owning backend's transfer state. Called once by <see cref="CudaBackend"/> right after it
    /// registers with <see cref="GpuTransferHelper"/>.</summary>
    internal void BindState(GpuTransferHelper.State state) => _state = state;

    /// <summary>Entry-point guard: binds this cache's owning backend as the thread's ambient state, then the context.</summary>
    private void Enter()
    {
        if (_state is not null)
        {
            GpuTransferHelper.SetAmbient(_state);
        }
        _context.EnsureCurrent();
    }

    /// <inheritdoc/>
    public StreamingUploadToken BeginUploadAsync(IEnumerable<Tensor> weights)
    {
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        Enter();

        // Materialize the to-upload set first: the staging path needs the total byte count up front.
        List<Tensor> pending = new();
        nuint totalBytes = 0;
        foreach (Tensor weight in weights)
        {
            if (GpuTransferHelper.IsWeightCached(weight))
            {
                continue; // Free hit — the dptr is already valid in _weightCache.
            }
            pending.Add(weight);
            totalBytes += GpuTransferHelper.ByteSize(weight);
        }
        if (pending.Count == 0)
        {
            return StreamingUploadToken.Empty;
        }

        nint staging = PinUploadSource && !_stagingDead ? AcquireStagingSlot(totalBytes) : 0;
        nuint stagingOffset = 0;
        foreach (Tensor weight in pending)
        {
            nuint byteSize = GpuTransferHelper.ByteSize(weight);
            // Use cuMemAllocAsync on the upload stream so the alloc and the eventual
            // cuMemFreeAsync (on the compute stream during eviction) round-trip through
            // the same stream-ordered mempool. The previous mix (sync cuMemAlloc + async
            // cuMemFreeAsync) routed memory in the front door but out the back: freed
            // bytes ended up in the pool while subsequent sync cuMemAlloc calls couldn't
            // see them, manifesting as OOM-despite-free. With consistent async use, the
            // pool's release policy (owned by DeviceMempoolPolicy) governs when bytes
            // return to the driver and DrainAndReleasePool's trim is meaningful.
            ulong dptr = CudaMemory.AllocateAsync(byteSize, _uploadStream);
            unsafe
            {
                nint hostSrc = (nint)weight.DataPointer;
                if (staging != 0)
                {
                    // memcpy into the pinned slot, upload from there: the H2D is then genuinely async and
                    // overlaps compute; the memcpy itself overlaps the GPU's work on earlier blocks (prefetch).
                    Buffer.MemoryCopy((void*)hostSrc, (void*)(staging + (nint)stagingOffset), byteSize, byteSize);
                    CudaDriverApi.cuMemcpyHtoDAsync(dptr, staging + (nint)stagingOffset, byteSize, _uploadStream).ThrowOnError();
                    stagingOffset += byteSize;
                }
                else
                {
                    CudaDriverApi.cuMemcpyHtoDAsync(dptr, hostSrc, byteSize, _uploadStream).ThrowOnError();
                }
            }
            // Register immediately even though the upload is in flight: ops won't
            // read these tensors until AwaitWeights gates the compute stream on
            // the completion event below, at which point the data is guaranteed
            // visible. Registering up front means a parallel BeginUploadAsync
            // call for the same tensor sees it as cached and skips re-upload.
            GpuTransferHelper.RegisterCachedWeight(weight, dptr, byteSize);
        }

        if (staging != 0)
        {
            // Slot guard: reuse of this staging buffer must wait for these uploads to drain.
            CudaDriverApi.cuEventRecord(_stagingEvents[_stagingIdx], _uploadStream).ThrowOnError();
            _stagingUsed[_stagingIdx] = true;
            _stagingIdx = (_stagingIdx + 1) % StagingSlots;
        }

        // Record a completion event after all the queued copies. Disabling timing
        // saves a tiny bit of driver overhead; we never read elapsed time on these.
        CudaDriverApi.cuEventCreate(out nint evt, CudaDriverApi.CU_EVENT_DISABLE_TIMING).ThrowOnError();
        CudaDriverApi.cuEventRecord(evt, _uploadStream).ThrowOnError();
        return new StreamingUploadToken(evt, this);
    }

    /// <summary>Returns the current ring slot's pinned pointer (host-waiting on its prior uploads if still in
    /// flight), growing the ring buffers if <paramref name="totalBytes"/> exceeds the slot size. Returns 0 (and
    /// disables staging for the session) if pinned allocation fails — callers fall back to pageable uploads.</summary>
    private nint AcquireStagingSlot(nuint totalBytes)
    {
        if (totalBytes > _stagingBytes)
        {
            ReleaseStaging();
            for (int i = 0; i < StagingSlots; i++)
            {
                int rc = CudaDriverApi.cuMemHostAlloc(out _stagingPtrs[i], totalBytes, CudaDriverApi.CU_MEMHOSTALLOC_PORTABLE);
                if (rc != 0)
                {
                    Logs.Warning($"cuMemHostAlloc({totalBytes >> 20} MB) failed (rc={rc}); streaming uploads stay pageable (no compute overlap).");
                    ReleaseStaging();
                    _stagingDead = true;
                    return 0;
                }
                CudaDriverApi.cuEventCreate(out _stagingEvents[i], CudaDriverApi.CU_EVENT_DISABLE_TIMING).ThrowOnError();
            }
            _stagingBytes = totalBytes;
            for (int i = 0; i < StagingSlots; i++) _stagingUsed[i] = false;
            _stagingIdx = 0;
        }
        if (_stagingUsed[_stagingIdx])
        {
            CudaDriverApi.cuEventSynchronize(_stagingEvents[_stagingIdx]).ThrowOnError();
        }
        return _stagingPtrs[_stagingIdx];
    }

    /// <summary>Frees the staging ring (waiting out in-flight uploads first). Safe to call repeatedly.</summary>
    private void ReleaseStaging()
    {
        for (int i = 0; i < StagingSlots; i++)
        {
            if (_stagingEvents[i] != 0)
            {
                if (_stagingUsed[i]) CudaDriverApi.cuEventSynchronize(_stagingEvents[i]);
                CudaDriverApi.cuEventDestroy(_stagingEvents[i]);
                _stagingEvents[i] = 0;
            }
            if (_stagingPtrs[i] != 0)
            {
                CudaDriverApi.cuMemFreeHost(_stagingPtrs[i]);
                _stagingPtrs[i] = 0;
            }
            _stagingUsed[i] = false;
        }
        _stagingBytes = 0;
        _stagingIdx = 0;
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
        Enter();
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
        Enter();
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

    /// <summary>Releases pinned host resources (the staging ring). Call before tearing down the backend so
    /// page-locked memory is returned to the OS.</summary>
    public void UnregisterPinnedSources()
    {
        Enter();
        ReleaseStaging();
    }

    /// <inheritdoc/>
    public long QueryAvailableWeightCacheBytes(long activationReserve)
    {
        if (activationReserve < 0) throw new ArgumentOutOfRangeException(nameof(activationReserve));
        Enter();
        CudaDriverApi.cuMemGetInfo(out nuint freeBytes, out _).ThrowOnError();
        long avail = (long)freeBytes - activationReserve;
        return avail < 0 ? 0 : avail;
    }

    /// <inheritdoc/>
    public void DrainAndReleasePool()
    {
        Enter();
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

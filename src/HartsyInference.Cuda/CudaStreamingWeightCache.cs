using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>CUDA implementation of <see cref="IStreamingWeightCache"/>. Uploads weights on a dedicated upload stream so the copy engine runs in parallel with compute SMs, gates the compute stream on completion via CUDA events, and frees on the compute stream so reclamation is naturally ordered after any prior reads.</summary>
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
    private const int DefaultStagingSlots = 3;
    private nint[] _stagingPtrs = new nint[DefaultStagingSlots];
    private nint[] _stagingEvents = new nint[DefaultStagingSlots];
    private bool[] _stagingUsed = new bool[DefaultStagingSlots];
    private nuint _stagingBytes;
    private int _stagingIdx;
    private bool _stagingDead;   // allocation failed once — fall back to pageable direct uploads for the session

    /// <summary>Pinned staging slots in the ring (default 3); each holds one <see cref="BeginUploadAsync"/> call's bytes and its reuse waits on that slot's last upload. Deeper prefetch wants more.</summary>
    public int StagingSlotCount
    {
        get => _stagingPtrs.Length;
        set
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(value, 1);
            if (value == _stagingPtrs.Length) return;
            Enter();
            ReleaseStaging();
            _stagingPtrs = new nint[value];
            _stagingEvents = new nint[value];
            _stagingUsed = new bool[value];
        }
    }

    /// <summary>Opt-in: page-lock each weight's host source before uploading so <c>cuMemcpyHtoDAsync</c> is genuinely asynchronous and overlaps with compute (pageable sources silently force a synchronous staging copy that overlaps with nothing). Defaults to <c>false</c>.</summary>
    /// <remarks>Only beneficial when weights are re-uploaded across steps (block-swap); for a one-shot preload the
    /// registration cost is not amortized. Requires the CPU weight tensors to stay resident (do not dispose them)
    /// for the lifetime of the stream.</remarks>
    public bool PinUploadSource { get; set; }

    /// <summary>Constructs a streaming cache bound to a compute stream and upload stream. Both streams should be created with <c>CU_STREAM_NON_BLOCKING</c> so they can run independently of the legacy NULL stream and of each other.</summary>
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

    /// <summary>The owning backend's transfer state; bound after registration so this cache's entry points can set the ambient (two same-device backends share a context, so context identity cannot route these calls).</summary>
    private GpuTransferHelper.State? _state;

    /// <summary>Binds the owning backend's transfer state. Called once by <see cref="CudaBackend"/> right after it registers with <see cref="GpuTransferHelper"/>.</summary>
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
        List<Tensor> registered = new(pending.Count);
        ulong unregistered = 0;
        try
        {
            foreach (Tensor weight in pending)
            {
                nuint byteSize = GpuTransferHelper.ByteSize(weight);
                // Allocate on the upload stream so this and the eventual cuMemFreeAsync (compute stream, on eviction)
                // round-trip through the same stream-ordered pool; DeviceMempoolPolicy owns when it returns to the driver.
                unregistered = CudaMemory.AllocateAsync(byteSize, _uploadStream);
                UploadOne(weight, unregistered, byteSize, staging, ref stagingOffset);
                // Register while the copy is in flight: ops only read it after AwaitWeights gates the compute
                // stream, and a parallel BeginUploadAsync for the same tensor then sees a free hit.
                GpuTransferHelper.RegisterCachedWeight(weight, unregistered, byteSize);
                registered.Add(weight);
                unregistered = 0;
            }
        }
        catch
        {
            // Drain first: queued copies still read the staging slot and write the buffers freed below.
            CudaDriverApi.cuStreamSynchronize(_uploadStream);
            if (unregistered != 0) CudaDriverApi.cuMemFreeAsync(unregistered, _uploadStream);
            foreach (Tensor weight in registered)
            {
                if (GpuTransferHelper.TryUnregisterCachedWeight(weight, out ulong dptr))
                    CudaDriverApi.cuMemFreeAsync(dptr, _uploadStream);
            }
            throw;
        }

        if (staging != 0)
        {
            // Slot guard: reuse of this staging buffer must wait for these uploads to drain.
            CudaDriverApi.cuEventRecord(_stagingEvents[_stagingIdx], _uploadStream).ThrowOnError();
            _stagingUsed[_stagingIdx] = true;
            _stagingIdx = (_stagingIdx + 1) % _stagingPtrs.Length;
        }

        // Record a completion event after all the queued copies. Disabling timing
        // saves a tiny bit of driver overhead; we never read elapsed time on these.
        CudaDriverApi.cuEventCreate(out nint evt, CudaDriverApi.CU_EVENT_DISABLE_TIMING).ThrowOnError();
        CudaDriverApi.cuEventRecord(evt, _uploadStream).ThrowOnError();
        return new StreamingUploadToken(evt, this);
    }

    private unsafe void UploadOne(Tensor weight, ulong dptr, nuint byteSize, nint staging, ref nuint stagingOffset)
    {
        nint hostSrc = (nint)weight.DataPointer;
        if (staging != 0)
        {
            // memcpy into the pinned slot, upload from there: the H2D is then genuinely async and overlaps compute.
            Buffer.MemoryCopy((void*)hostSrc, (void*)(staging + (nint)stagingOffset), byteSize, byteSize);
            CudaDriverApi.cuMemcpyHtoDAsync(dptr, staging + (nint)stagingOffset, byteSize, _uploadStream).ThrowOnError();
            stagingOffset += byteSize;
        }
        else
        {
            CudaDriverApi.cuMemcpyHtoDAsync(dptr, hostSrc, byteSize, _uploadStream).ThrowOnError();
        }
    }

    /// <summary>Returns the current ring slot's pinned pointer (host-waiting on its prior uploads if still in flight), growing the ring buffers if <paramref name="totalBytes"/> exceeds the slot size. Returns 0 (and disables staging for the session) if pinned allocation fails — callers fall back to pageable uploads.</summary>
    private nint AcquireStagingSlot(nuint totalBytes)
    {
        if (totalBytes > _stagingBytes)
        {
            ReleaseStaging();
            for (int i = 0; i < _stagingPtrs.Length; i++)
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
            for (int i = 0; i < _stagingPtrs.Length; i++) _stagingUsed[i] = false;
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
        List<Exception>? failures = null;
        for (int i = 0; i < _stagingPtrs.Length; i++)
        {
            nint stagingEvent = _stagingEvents[i];
            nint stagingPtr = _stagingPtrs[i];
            bool stagingUsed = _stagingUsed[i];
            _stagingEvents[i] = 0;
            _stagingPtrs[i] = 0;
            _stagingUsed[i] = false;

            if (stagingEvent != 0)
            {
                if (stagingUsed)
                {
                    try { CudaDriverApi.cuEventSynchronize(stagingEvent).ThrowOnError(); }
                    catch (Exception error) { (failures ??= []).Add(error); }
                }
                try { CudaDriverApi.cuEventDestroy(stagingEvent).ThrowOnError(); }
                catch (Exception error) { (failures ??= []).Add(error); }
            }
            if (stagingPtr != 0)
            {
                try { CudaDriverApi.cuMemFreeHost(stagingPtr).ThrowOnError(); }
                catch (Exception error) { (failures ??= []).Add(error); }
            }
        }
        _stagingBytes = 0;
        _stagingIdx = 0;
        if (failures is not null)
            throw new AggregateException("One or more CUDA streaming staging resources failed to release.", failures);
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

    /// <summary>The compute stream the uploads are ordered against.</summary>
    internal nint ComputeStreamHandle => _computeStream;

    /// <summary>Keeps these tensors out of the ≥1 MB auto-promotion for the owning backend, so an evicted expert is never silently resurrected as a permanent weight.</summary>
    internal void ExcludeFromAutoPromotion(IEnumerable<Tensor> tensors)
    {
        Enter();
        foreach (Tensor tensor in tensors) GpuTransferHelper.ExcludeFromAutoPromotion(tensor);
    }

    /// <summary>The device pointer of a tensor this cache uploaded, without uploading anything.</summary>
    internal bool TryGetDevicePointer(Tensor tensor, out ulong dptr)
    {
        Enter();
        return GpuTransferHelper.TryGetCachedDevice(tensor, out dptr);
    }

    /// <summary>Records an event on the compute stream after all work queued so far.</summary>
    internal nint RecordComputeFence()
    {
        Enter();
        CudaDriverApi.cuEventCreate(out nint evt, CudaDriverApi.CU_EVENT_DISABLE_TIMING).ThrowOnError();
        try
        {
            CudaDriverApi.cuEventRecord(evt, _computeStream).ThrowOnError();
        }
        catch
        {
            CudaDriverApi.cuEventDestroy(evt);
            throw;
        }
        return evt;
    }

    /// <summary>True once the compute stream has passed a fence from <see cref="RecordComputeFence"/>.</summary>
    internal bool IsFenceDone(nint fence)
    {
        Enter();
        int rc = CudaDriverApi.cuEventQuery(fence);
        if (rc == 0) return true;
        if (rc == CudaDriverApi.CUDA_ERROR_NOT_READY) return false;
        rc.ThrowOnError();
        return false;
    }

    /// <summary>Blocks the host until the compute stream has passed a fence.</summary>
    internal void WaitFence(nint fence)
    {
        Enter();
        CudaDriverApi.cuEventSynchronize(fence).ThrowOnError();
    }

    /// <summary>Destroys a fence.</summary>
    internal void DestroyFence(nint fence)
    {
        Enter();
        CudaDriverApi.cuEventDestroy(fence).ThrowOnError();
    }

    /// <summary>Releases pinned host resources (the staging ring). Call before tearing down the backend so page-locked memory is returned to the OS.</summary>
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

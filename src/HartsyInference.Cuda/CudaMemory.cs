using HartsyInference.Core.Logging;

namespace HartsyInference.Cuda;

/// <summary>GPU memory allocation and transfer helpers wrapping CUDA Driver API memory functions.</summary>
public static class CudaMemory
{
    /// <summary>Per-context compute streams (keyed by raw context handle), so each backend's transient
    /// allocations land in ITS OWN stream-ordered pool. A single static stream here was half of the
    /// multi-backend poison: backend A's transients allocated/freed on backend B's stream → cross-device
    /// stream ops → CUDA 700 on both GPUs.</summary>
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<nint, nint> _computeStreams = new();

    /// <summary>Fast path for the single-backend case (no cuCtxGetCurrent per alloc). 0 when zero or 2+ registered.</summary>
    private static volatile nint _singleStream;

    /// <summary>Binds the compute stream for a backend's context so transient allocations use the stream-ordered
    /// pool (matching their <c>cuMemFreeAsync</c> frees). Called once from <see cref="CudaBackend"/>'s constructor.</summary>
    public static void SetComputeStream(CudaContext context, nint stream)
    {
        _computeStreams[context.Handle] = stream;
        _singleStream = _computeStreams.Count == 1 ? stream : 0;
    }

    /// <summary>Removes a context's stream binding at backend disposal.</summary>
    public static void RemoveComputeStream(CudaContext context)
    {
        _computeStreams.TryRemove(context.Handle, out _);
        _singleStream = _computeStreams.Count == 1 ? System.Linq.Enumerable.First(_computeStreams.Values) : 0;
    }

    /// <summary>Resolves the calling thread's current context to its registered compute stream (0 = none;
    /// falls back to the synchronous allocator). Callers hold their backend's context current — the same
    /// invariant <see cref="GpuTransferHelper"/> relies on.</summary>
    private static nint ResolveStream()
    {
        nint single = _singleStream;
        if (single != 0) return single;
        if (_computeStreams.IsEmpty) return 0;
        if (CudaDriverApi.cuCtxGetCurrent(out nint current) == 0 && _computeStreams.TryGetValue(current, out nint stream))
            return stream;
        return 0;
    }

    /// <summary>Allocates <b>transient</b> device memory (op outputs, dtype casts, scratch) and returns a device
    /// pointer. Routes through the stream-ordered pool (<c>cuMemAllocAsync</c> on the compute stream) so the memory
    /// is reused by the matching <c>cuMemFreeAsync</c> frees in <see cref="GpuTransferHelper.FreeDevice"/> — the
    /// previous mix of synchronous <c>cuMemAlloc</c> here with async frees there sent freed bytes into the pool
    /// where subsequent sync allocs couldn't see them, so the GPU "filled up" and every op OOM-retried with a full
    /// stream-drain + pool-trim (the cause of the Ideogram-4 ~100s/step thrash on a near-full A100). This mirrors the
    /// fix already applied to the streaming weight cache. Persistent buffers (resident weights, cuBLAS workspaces)
    /// freed via synchronous <see cref="Free"/> must use <see cref="AllocatePersistent"/> instead.</summary>
    public static ulong Allocate(nuint byteSize)
    {
        nint stream = ResolveStream();
        if (stream != 0)
            return AllocateAsync(byteSize, stream);
        return AllocatePersistent(byteSize);
    }

    /// <summary>Allocates <b>persistent</b> device memory with the synchronous driver allocator (<c>cuMemAlloc</c>),
    /// to be released with the synchronous <see cref="Free"/>. Use for buffers that live for the whole model/session
    /// (resident weights, cuBLAS workspaces) — keeping them out of the churning stream-ordered pool. On OOM, drains
    /// the active streams and trims the pool back to the driver before retrying once.</summary>
    public static ulong AllocatePersistent(nuint byteSize)
    {
        int result = CudaDriverApi.cuMemAlloc(out ulong dptr, byteSize);
        if (result == 2) // CUDA_ERROR_OUT_OF_MEMORY
        {
            // Log pre-retry state so we can see exactly how much the driver thinks is free
            // vs how much we asked for. This is the only way to distinguish "genuinely OOM"
            // from "memory stuck in stream-ordered pool" without a debugger attached.
            LogOomDiagnostic("OOM on first attempt", byteSize);
            GpuTransferHelper.SyncStreamsAndReleasePool();
            int retryResult = CudaDriverApi.cuMemAlloc(out dptr, byteSize);
            if (retryResult != 0)
            {
                LogOomDiagnostic("OOM after sync+pool-trim retry", byteSize);
                retryResult.ThrowOnError();
            }
        }
        else
        {
            result.ThrowOnError();
        }
        return dptr;
    }

    /// <summary>Emits a one-line diagnostic showing requested bytes alongside the driver's
    /// view of free / total VRAM. Best-effort: a failure here is swallowed so it can never
    /// mask the real allocation failure that triggered the call.</summary>
    private static void LogOomDiagnostic(string stage, nuint requested)
    {
        try
        {
            int infoResult = CudaDriverApi.cuMemGetInfo(out nuint freeBytes, out nuint totalBytes);
            if (infoResult == 0)
            {
                double reqMb = requested / (1024.0 * 1024.0);
                double freeMb = freeBytes / (1024.0 * 1024.0);
                double totalMb = totalBytes / (1024.0 * 1024.0);
                Logs.Warning($"[CudaMemory] {stage}: requested={reqMb:F1} MB, free={freeMb:F1} MB, total={totalMb:F1} MB ({(double)freeBytes / totalBytes * 100:F1}% free)");
            }
            else
            {
                Logs.Warning($"[CudaMemory] {stage}: requested={requested / (1024.0 * 1024.0):F1} MB, cuMemGetInfo failed (err={infoResult})");
            }
        }
        catch
        {
            // Diagnostic must never throw — the caller is already on an error path.
        }
    }

    /// <summary>Returns (free, total) device memory in bytes via <c>cuMemGetInfo</c>. (0,0) on failure.</summary>
    public static (long FreeBytes, long TotalBytes) GetMemInfo()
    {
        if (CudaDriverApi.cuMemGetInfo(out nuint freeBytes, out nuint totalBytes) == 0)
            return ((long)freeBytes, (long)totalBytes);
        return (0, 0);
    }

    /// <summary>Frees device memory.</summary>
    public static void Free(ulong dptr)
    {
        if (dptr != 0)
        {
            CudaDriverApi.cuMemFree(dptr).ThrowOnError();
        }
    }

    /// <summary>Copies bytes from host to device.</summary>
    public static unsafe void CopyHostToDevice(ulong dst, void* src, nuint byteSize)
    {
        CudaDriverApi.cuMemcpyHtoD(dst, (nint)src, byteSize).ThrowOnError();
    }

    /// <summary>Copies bytes from device to host.</summary>
    public static unsafe void CopyDeviceToHost(void* dst, ulong src, nuint byteSize)
    {
        CudaDriverApi.cuMemcpyDtoH((nint)dst, src, byteSize).ThrowOnError();
    }

    /// <summary>Copies bytes between device pointers (synchronous — serializes against the null stream; NOT
    /// legal during stream capture. Hot paths should use <see cref="CopyDeviceToDeviceAsync"/>).</summary>
    public static void CopyDeviceToDevice(ulong dst, ulong src, nuint byteSize)
    {
        CudaDriverApi.cuMemcpyDtoD(dst, src, byteSize).ThrowOnError();
    }

    /// <summary>Stream-ordered device-to-device copy (capture-legal; no host serialization).</summary>
    public static void CopyDeviceToDeviceAsync(ulong dst, ulong src, nuint byteSize, nint stream)
    {
        CudaDriverApi.cuMemcpyDtoDAsync(dst, src, byteSize, stream).ThrowOnError();
    }

    /// <summary>Zeros device memory.</summary>
    public static void Zero(ulong dptr, nuint byteSize)
    {
        CudaDriverApi.cuMemsetD8(dptr, 0, byteSize).ThrowOnError();
    }

    /// <summary>Fills device memory with a 32-bit value (e.g., float pattern).</summary>
    public static void Fill32(ulong dptr, uint value, nuint count)
    {
        CudaDriverApi.cuMemsetD32(dptr, value, count).ThrowOnError();
    }

    /// <summary>Allocates device memory asynchronously on the given stream. Mirrors
    /// <see cref="Allocate"/>'s OOM retry: if the stream-ordered allocator can't satisfy
    /// the request, drain everything and trim the pool, then retry once.</summary>
    // Capture-window alloc/free tracker (diagnostic): during step-graph capture, every cuMemAllocAsync
    // becomes a graph allocation whose lifetime the graph owns — an alloc without a matching captured free
    // leaks one launch's worth of memory permanently when the graph is destroyed. Enabled by
    // CudaBackend.StepGraphBegin, reported at EndAndLaunch.
    internal static bool TrackCaptureWindow;
    internal static readonly System.Collections.Generic.Dictionary<ulong, nuint> CaptureAllocs = new();
    internal static long CaptureAllocBytes, CaptureFreeBytes;
    internal static int CaptureAllocCount, CaptureFreeCount;

    public static ulong AllocateAsync(nuint byteSize, nint stream)
    {
        int result = CudaDriverApi.cuMemAllocAsync(out ulong dptr, byteSize, stream);
        if (TrackCaptureWindow && result == 0)
        {
            lock (CaptureAllocs)
            {
                CaptureAllocs[dptr] = byteSize;
                CaptureAllocBytes += (long)byteSize;
                CaptureAllocCount++;
            }
        }
        if (result == 2) // CUDA_ERROR_OUT_OF_MEMORY
        {
            LogOomDiagnostic("OOM on async first attempt", byteSize);
            GpuTransferHelper.SyncStreamsAndReleasePool();
            int retryResult = CudaDriverApi.cuMemAllocAsync(out dptr, byteSize, stream);
            if (retryResult == 2)
            {
                // The driver holds lazily-reclaimable caches (measured: ~4.5 GB of destroyed CUDA-graph
                // memory after a step-graph session — reported as USED by cuMemGetInfo, untouched by
                // cuMemPoolTrimTo/cuDeviceGraphMemTrim) that it releases ONLY under synchronous allocation
                // pressure, never to grow a stream-ordered pool. Probe-allocate the requested size with
                // cuMemAlloc to force the reclaim, free the probe, then let the pool grow into it.
                if (CudaDriverApi.cuMemAlloc(out ulong probe, byteSize) == 0)
                {
                    CudaDriverApi.cuMemFree(probe);
                    LogOomDiagnostic("async retry after sync-probe reclaim", byteSize);
                    retryResult = CudaDriverApi.cuMemAllocAsync(out dptr, byteSize, stream);
                }
            }
            if (retryResult == 2)
            {
                // Last resort before failing: surrender the weight-cast cache (pure, rebuildable) — its
                // opportunistic use of free VRAM must never make an allocation fail that would otherwise
                // succeed (gemma-2's 2.25 GB scaled-embed alloc vs prefill cast caching, 2026-07-23).
                long released = GpuTransferHelper.EvictAllWeightCasts();
                if (released > 0)
                {
                    GpuTransferHelper.SyncStreamsAndReleasePool();
                    LogOomDiagnostic($"async retry after cast-cache eviction ({released >> 20} MB released)", byteSize);
                    retryResult = CudaDriverApi.cuMemAllocAsync(out dptr, byteSize, stream);
                }
            }
            if (retryResult != 0)
            {
                LogOomDiagnostic("OOM after async sync+pool-trim retry", byteSize);
                retryResult.ThrowOnError();
            }
        }
        else
        {
            result.ThrowOnError();
        }
        return dptr;
    }

    /// <summary>Frees device memory asynchronously on the given stream.</summary>
    public static void FreeAsync(ulong dptr, nint stream)
    {
        if (dptr != 0)
        {
            if (TrackCaptureWindow)
            {
                lock (CaptureAllocs)
                {
                    if (CaptureAllocs.Remove(dptr, out nuint sz))
                    {
                        CaptureFreeBytes += (long)sz;
                        CaptureFreeCount++;
                    }
                    else
                    {
                        // A free node for memory the graph does NOT own replays on EVERY launch — the
                        // buffer's real owner is left pointing at freed, reusable memory. Surface it.
                        Logs.Warning($"[Cuda] step-graph capture recorded a FREE of external device ptr 0x{dptr:X} — the captured graph will re-free it on every replay.");
                    }
                }
            }
            CudaDriverApi.cuMemFreeAsync(dptr, stream).ThrowOnError();
        }
    }

    /// <summary>Copies host to device asynchronously on the given stream. Host memory must be pinned.</summary>
    public static unsafe void CopyHostToDeviceAsync(ulong dst, void* src, nuint byteSize, nint stream)
    {
        CudaDriverApi.cuMemcpyHtoDAsync(dst, (nint)src, byteSize, stream).ThrowOnError();
    }

    /// <summary>Copies device to host asynchronously on the given stream. Host memory must be pinned.</summary>
    public static unsafe void CopyDeviceToHostAsync(void* dst, ulong src, nuint byteSize, nint stream)
    {
        CudaDriverApi.cuMemcpyDtoHAsync((nint)dst, src, byteSize, stream).ThrowOnError();
    }
}

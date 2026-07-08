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

    // ── Activation arena (HARTSY_ACT_ARENA=1) ──────────────────────────────────────────────────────────────────
    // The measured Krea2 bottleneck (~18 s of a 27.7 s gen) is stream-ordered cuMemAllocAsync/cuMemFreeAsync driver
    // churn: a denoise step allocates+frees thousands of 100–300 MB activation buffers, each a driver call ordered
    // on the compute stream. The arena recycles freed blocks IN-PROCESS instead of returning them to the driver's
    // pool: a size-keyed LIFO free-list per stream. On alloc, pop a same-size idle block (no driver call); on free,
    // push it back (no driver call). cuMemAllocAsync fires only the FIRST time a given size is needed.
    //
    // Correctness: within ONE in-order compute stream this preserves the exact ordering guarantee the async pool
    // gives. A block is pushed only after its last consumer kernel was enqueued; a later op that pops it enqueues
    // its kernel AFTER on the same stream, so the reuse can never race the previous read. Blocks are bucketed by the
    // stream they were allocated on and only handed back to allocs requesting that same stream, so a cross-context
    // free never leaks a block into another device's stream. Result is bit-identical to the driver-pool path (same
    // bytes, same stream order) and — because the denoise step issues an identical alloc/free sequence every step —
    // hands out identical device addresses every step, the precondition for CUDA-graph capture (Phase 2).
    private static readonly bool _arenaEnabled = Environment.GetEnvironmentVariable("HARTSY_ACT_ARENA") == "1";

    /// <summary>Optional cap (bytes) on total VRAM the arena retains; 0 = unbounded. Above it, frees genuinely
    /// release to the driver instead of recycling, so VRAM-poor cards can bound the held working set
    /// (<c>HARTSY_ARENA_MAX_MB</c>). Never touches weight VRAM (weights use the synchronous persistent allocator).</summary>
    private static readonly long _arenaMaxBytes =
        long.TryParse(Environment.GetEnvironmentVariable("HARTSY_ARENA_MAX_MB"), out long mb) ? mb << 20 : 0L;

    private static readonly object _arenaLock = new();
    /// <summary>Idle recyclable blocks, keyed by (owning stream, exact byte size), LIFO for address determinism.</summary>
    private static readonly Dictionary<(nint stream, nuint size), Stack<ulong>> _arenaFree = new();
    /// <summary>Every block the arena owns (alloc'd through <see cref="AllocateAsync"/> while enabled) → its stream+size.
    /// Membership is what marks a pointer as "recycle me" in <see cref="FreeAsync"/>; anything else is a genuine
    /// transient and is freed to the driver as before.</summary>
    private static readonly Dictionary<ulong, (nint stream, nuint size)> _arenaOwned = new();
    /// <summary>Total bytes of all arena-owned blocks (live + idle) — the arena's VRAM footprint, checked against the cap.</summary>
    private static long _arenaOwnedBytes;
    /// <summary>Recycle hits (served from the free-list, no driver call) vs misses (a real cuMemAllocAsync). After the
    /// first denoise step warms every distinct size, a fixed-shape loop should be ~all hits — misses ≈ distinct sizes.</summary>
    private static long _arenaHits, _arenaMisses;

    /// <summary>Arena effectiveness snapshot for logging/tests: (recycle hits, driver-alloc misses, retained MB).</summary>
    public static (long hits, long misses, long retainedMb) GetArenaStats()
    {
        lock (_arenaLock) return (_arenaHits, _arenaMisses, _arenaOwnedBytes >> 20);
    }

    /// <summary>Binds the compute stream for a backend's context so transient allocations use the stream-ordered
    /// pool (matching their <c>cuMemFreeAsync</c> frees). Called once from <see cref="CudaBackend"/>'s constructor.</summary>
    public static void SetComputeStream(CudaContext context, nint stream)
    {
        _computeStreams[context.Handle] = stream;
        _singleStream = _computeStreams.Count == 1 ? stream : 0;
    }

    /// <summary>Removes a context's stream binding at backend disposal, releasing any arena blocks held on its stream.</summary>
    public static void RemoveComputeStream(CudaContext context)
    {
        if (_computeStreams.TryRemove(context.Handle, out nint stream))
            ReleaseArenaForStream(stream);
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

    /// <summary>Copies bytes between device pointers.</summary>
    public static void CopyDeviceToDevice(ulong dst, ulong src, nuint byteSize)
    {
        CudaDriverApi.cuMemcpyDtoD(dst, src, byteSize).ThrowOnError();
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
    public static ulong AllocateAsync(nuint byteSize, nint stream)
    {
        // Arena fast path: hand back a recycled same-size idle block on this stream — no driver call at all.
        if (_arenaEnabled)
        {
            lock (_arenaLock)
            {
                if (_arenaFree.TryGetValue((stream, byteSize), out Stack<ulong>? bucket) && bucket.Count > 0)
                {
                    _arenaHits++;
                    return bucket.Pop();
                }
            }
        }

        int result = CudaDriverApi.cuMemAllocAsync(out ulong dptr, byteSize, stream);
        if (result == 2) // CUDA_ERROR_OUT_OF_MEMORY
        {
            LogOomDiagnostic("OOM on async first attempt", byteSize);
            // SyncStreamsAndReleasePool drains the arena's idle blocks back to the driver before trimming the pool,
            // so a genuine new-size request can reclaim VRAM the arena was holding.
            GpuTransferHelper.SyncStreamsAndReleasePool();
            int retryResult = CudaDriverApi.cuMemAllocAsync(out dptr, byteSize, stream);
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

        if (_arenaEnabled)
        {
            lock (_arenaLock)
            {
                _arenaMisses++;
                _arenaOwned[dptr] = (stream, byteSize);
                _arenaOwnedBytes += (long)byteSize;
            }
        }
        return dptr;
    }

    /// <summary>Frees device memory asynchronously on the given stream. With the arena enabled, an arena-owned block
    /// is recycled to its size bucket (no driver call) instead of freed, unless the retained footprint is over the
    /// optional cap — then it is genuinely released to shrink back toward the cap.</summary>
    public static void FreeAsync(ulong dptr, nint stream)
    {
        if (dptr == 0)
            return;

        if (_arenaEnabled)
        {
            lock (_arenaLock)
            {
                if (_arenaOwned.TryGetValue(dptr, out (nint stream, nuint size) info))
                {
                    // Over the cap: actually release this block (drops the owned footprint toward the cap).
                    if (_arenaMaxBytes > 0 && _arenaOwnedBytes > _arenaMaxBytes)
                    {
                        _arenaOwned.Remove(dptr);
                        _arenaOwnedBytes -= (long)info.size;
                        CudaDriverApi.cuMemFreeAsync(dptr, info.stream).ThrowOnError();
                        return;
                    }
                    // Recycle into the block's OWN stream/size bucket (stream-safe reuse; see the arena note above).
                    if (!_arenaFree.TryGetValue((info.stream, info.size), out Stack<ulong>? bucket))
                        _arenaFree[(info.stream, info.size)] = bucket = new Stack<ulong>();
                    bucket.Push(dptr);
                    return;
                }
            }
        }

        CudaDriverApi.cuMemFreeAsync(dptr, stream).ThrowOnError();
    }

    /// <summary>Releases the arena's IDLE (free-list) blocks back to the driver's stream-ordered pool. Called on the
    /// OOM-retry path (via <see cref="GpuTransferHelper.SyncStreamsAndReleasePool"/>) so a new-size request can reclaim
    /// the VRAM the arena was holding; live blocks (owned but still in use) are left untouched. No-op when disabled.</summary>
    public static void DrainArena()
    {
        if (!_arenaEnabled)
            return;
        lock (_arenaLock)
        {
            foreach (KeyValuePair<(nint stream, nuint size), Stack<ulong>> kv in _arenaFree)
            {
                while (kv.Value.Count > 0)
                {
                    ulong p = kv.Value.Pop();
                    if (_arenaOwned.Remove(p, out (nint stream, nuint size) info))
                        _arenaOwnedBytes -= (long)info.size;
                    CudaDriverApi.cuMemFreeAsync(p, kv.Key.stream).ThrowOnError();
                }
            }
            _arenaFree.Clear();
        }
    }

    /// <summary>Releases the arena's IDLE (free-list) blocks for a stream at backend teardown and drops all of the
    /// stream's owned-block bookkeeping. Frees ONLY free-list blocks: those were removed from the caches'
    /// <c>CachedPointers</c> set before being recycled, so <see cref="GpuTransferHelper.FreeAllCached"/> (which runs
    /// first at teardown and frees every cached pointer synchronously) can never have touched them — no double-free.
    /// Any still-"owned" block that is NOT in the free-list is a LIVE cached pointer that FreeAllCached already
    /// released; here we only forget it (dropping the stale <see cref="_arenaOwned"/> entry), we do not free it again.</summary>
    private static void ReleaseArenaForStream(nint stream)
    {
        if (!_arenaEnabled)
            return;
        lock (_arenaLock)
        {
            List<(nint stream, nuint size)> keys = new();
            foreach ((nint stream, nuint size) key in _arenaFree.Keys)
                if (key.stream == stream) keys.Add(key);
            foreach ((nint stream, nuint size) key in keys)
            {
                Stack<ulong> bucket = _arenaFree[key];
                while (bucket.Count > 0)
                {
                    ulong p = bucket.Pop();
                    _arenaOwned.Remove(p);            // free-list blocks are freed here — drop their ownership too
                    CudaDriverApi.cuMemFreeAsync(p, stream).ThrowOnError();
                }
                _arenaFree.Remove(key);
            }
            // Forget the remaining (live) owned entries for this stream — FreeAllCached freed the memory already.
            List<ulong> stale = new();
            foreach (KeyValuePair<ulong, (nint stream, nuint size)> kv in _arenaOwned)
                if (kv.Value.stream == stream) stale.Add(kv.Key);
            foreach (ulong p in stale)
                if (_arenaOwned.Remove(p, out (nint stream, nuint size) info))
                    _arenaOwnedBytes -= (long)info.size;
        }
        long total = _arenaHits + _arenaMisses;
        if (total > 0)
            Logs.Info($"[CudaMemory] activation arena: {_arenaHits} recycle hits / {_arenaMisses} driver allocs "
                + $"({(total == 0 ? 0 : 100.0 * _arenaHits / total):F1}% hit) across the session.");
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

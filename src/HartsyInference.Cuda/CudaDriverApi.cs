using System.Runtime.InteropServices;

namespace HartsyInference.Cuda;

/// <summary>P/Invoke bindings for the CUDA Driver API. Library name "cuda" is resolved at runtime by CudaLibraryResolver to nvcuda.dll (Windows) or libcuda.so.1 (Linux).</summary>
internal static partial class CudaDriverApi
{
    private const string LibName = "cuda";

    // ── Initialization ──────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuInit(uint flags);

    /// <summary>The latest CUDA version the installed DRIVER supports (e.g. 13020 → CUDA 13.2). Used to pick and version-guard the matching cuDNN build.</summary>
    [LibraryImport(LibName)]
    internal static partial int cuDriverGetVersion(out int driverVersion);

    // ── Device Management ───────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGet(out int device, int ordinal);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetCount(out int count);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetName(nint name, int len, int dev);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetAttribute(out int pi, int attrib, int dev);

    [LibraryImport(LibName, EntryPoint = "cuDeviceTotalMem_v2")]
    internal static partial int cuDeviceTotalMem(out nuint bytes, int dev);

    // ── Context Management ──────────────────────────────────────────────
    //
    // No cuCtxCreate/cuCtxDestroy bindings: this codebase deliberately uses
    // cuDevicePrimaryCtxRetain/Release instead — see CudaContext's type doc for why.

    [LibraryImport(LibName)]
    internal static partial int cuCtxSetCurrent(nint ctx);

    [LibraryImport(LibName)]
    internal static partial int cuCtxGetCurrent(out nint pctx);

    [LibraryImport(LibName)]
    internal static partial int cuCtxSynchronize();

    [LibraryImport(LibName)]
    internal static partial int cuDevicePrimaryCtxRetain(out nint pctx, int dev);

    [LibraryImport(LibName)]
    internal static partial int cuDevicePrimaryCtxRelease(int dev);

    // ── Module / PTX Loading ────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuModuleLoadDataEx(
        out nint module, nint ptxImage,
        uint numOptions, nint options, nint optionValues);

    [LibraryImport(LibName)]
    internal static partial int cuModuleUnload(nint module);

    [LibraryImport(LibName)]
    internal static partial int cuModuleGetFunction(
        out nint function, nint module,
        [MarshalAs(UnmanagedType.LPStr)] string name);

    // ── Kernel Launch ───────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuLaunchKernel(
        nint function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, nint stream,
        nint kernelParams, nint extra);

    // Set a function attribute. CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8 — required to opt a kernel into
    // >48 KB dynamic shared memory (up to ~99 KB/block on sm_89), e.g. the fused flash-attention kernel's K/V/S/O tiles.
    [LibraryImport(LibName)]
    internal static partial int cuFuncSetAttribute(nint function, int attrib, int value);

    // Read a function attribute. Registers and shared bytes together decide how many blocks fit per SM, which
    // explains a hand-written GEMM's achieved throughput far more often than its instruction mix does.
    // MIND THE ENUM: MAX_THREADS_PER_BLOCK = 0, SHARED_SIZE_BYTES = 1, CONST_SIZE_BYTES = 2, LOCAL_SIZE_BYTES = 3,
    // NUM_REGS = 4. Reading 0 expecting NUM_REGS returns the BLOCK SIZE, which for a 256-thread kernel reads as a
    // perfectly plausible "256 registers/thread" and supported a completely wrong occupancy diagnosis for several
    // experiments before a forced CU_JIT_MAX_REGISTERS=32 failed to change it.
    [LibraryImport(LibName)]
    internal static partial int cuFuncGetAttribute(out int value, int attrib, nint function);

    // ── Memory Management ───────────────────────────────────────────────

    [LibraryImport(LibName, EntryPoint = "cuMemAlloc_v2")]
    internal static partial int cuMemAlloc(out ulong dptr, nuint bytesize);

    [LibraryImport(LibName, EntryPoint = "cuMemFree_v2")]
    [SuppressGCTransition]
    internal static partial int cuMemFree(ulong dptr);

    [LibraryImport(LibName, EntryPoint = "cuMemcpyHtoD_v2")]
    internal static partial int cuMemcpyHtoD(ulong dst, nint src, nuint bytes);

    [LibraryImport(LibName, EntryPoint = "cuMemcpyDtoH_v2")]
    internal static partial int cuMemcpyDtoH(nint dst, ulong src, nuint bytes);

    [LibraryImport(LibName, EntryPoint = "cuMemcpyDtoD_v2")]
    internal static partial int cuMemcpyDtoD(ulong dst, ulong src, nuint bytes);

    [LibraryImport(LibName, EntryPoint = "cuMemcpyDtoDAsync_v2")]
    internal static partial int cuMemcpyDtoDAsync(ulong dst, ulong src, nuint bytes, nint stream);

    [LibraryImport(LibName, EntryPoint = "cuMemsetD8_v2")]
    internal static partial int cuMemsetD8(ulong dst, byte value, nuint count);

    [LibraryImport(LibName, EntryPoint = "cuMemsetD32_v2")]
    internal static partial int cuMemsetD32(ulong dst, uint value, nuint count);

    // ── Async Memory (CUDA 11.2+) ───────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuMemAllocAsync(out ulong dptr, nuint bytes, nint stream);

    [LibraryImport(LibName)]
    [SuppressGCTransition]
    internal static partial int cuMemFreeAsync(ulong dptr, nint stream);

    // The libcuda.so symbol table exports BOTH `cuMemcpyHtoDAsync` (legacy CUDA 1.x
    // ABI with 32-bit CUdeviceptr) AND `cuMemcpyHtoDAsync_v2` (modern, 64-bit). The
    // unsuffixed name binds to the legacy ABI on 64-bit Linux — the 64-bit ulong
    // dptr we pass gets misinterpreted, manifesting as CUDA_ERROR_INVALID_CONTEXT
    // (201) when the driver tries to validate the pointer. Same gotcha that
    // cuMemcpyHtoD_v2 / cuMemAlloc_v2 already pin in the sync functions above.
    [LibraryImport(LibName, EntryPoint = "cuMemcpyHtoDAsync_v2")]
    internal static partial int cuMemcpyHtoDAsync(ulong dst, nint src, nuint bytes, nint stream);

    [LibraryImport(LibName, EntryPoint = "cuMemcpyDtoHAsync_v2")]
    internal static partial int cuMemcpyDtoHAsync(nint dst, ulong src, nuint bytes, nint stream);

    // ── Peer (device-to-device across GPUs) ─────────────────────────────
    //
    // CUDA 4.0+ entry points (no legacy _v2 split — they postdate the 64-bit ABI
    // change). Peer copies work with or without peer ACCESS: without enabled
    // access the driver stages through host memory internally; with it (PCIe P2P
    // or NVLink) the copy goes device-to-device direct. CudaPeerAccess owns the
    // probe/enable memo; consumer boards commonly report no P2P, which is fine —
    // the engine's own pinned-staging fallback path is the primary-tested route.

    /// <summary>1 when <paramref name="dev"/> can access <paramref name="peerDev"/>'s memory directly (P2P/NVLink).</summary>
    [LibraryImport(LibName)]
    internal static partial int cuDeviceCanAccessPeer(out int canAccessPeer, int dev, int peerDev);

    /// <summary>Queries a CU_DEVICE_P2P_ATTRIBUTE_* value for the directed pair src→dst. Query-only — never changes context state.</summary>
    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetP2PAttribute(out int value, int attrib, int srcDevice, int dstDevice);

    /// <summary>Grants the CURRENT context direct access to <paramref name="peerContext"/>'s memory. Flags must be 0. Returns CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED (=704) on repeat — treat as success.</summary>
    [LibraryImport(LibName)]
    internal static partial int cuCtxEnablePeerAccess(nint peerContext, uint flags);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyPeerAsync(ulong dstDevice, nint dstContext, ulong srcDevice, nint srcContext, nuint byteCount, nint hStream);

    // ── Pinned (page-locked) host memory ────────────────────────────────
    //
    // The GPU copy engine cannot DMA out of pageable host memory; the driver
    // stages it through a temporary pinned buffer and the "async" copy silently
    // becomes synchronous, overlapping with nothing. Pinning the source makes
    // cuMemcpyHtoDAsync truly async (and ~2x faster on PCIe).

    /// <summary>Allocates a fresh page-locked host buffer. Flags: PORTABLE=1, DEVICEMAP=2, WRITECOMBINED=4.</summary>
    [LibraryImport(LibName, EntryPoint = "cuMemHostAlloc")]
    internal static partial int cuMemHostAlloc(out nint pp, nuint bytes, uint flags);

    [LibraryImport(LibName, EntryPoint = "cuMemFreeHost")]
    internal static partial int cuMemFreeHost(nint p);

    internal const uint CU_MEMHOSTALLOC_PORTABLE = 1;

    // ── Stream Management ───────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuStreamCreate(out nint stream, uint flags);

    [LibraryImport(LibName, EntryPoint = "cuStreamDestroy_v2")]
    internal static partial int cuStreamDestroy(nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuStreamSynchronize(nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuStreamQuery(nint stream);

    /// <summary>Makes <paramref name="stream"/> wait until <paramref name="hEvent"/> has been recorded. The host thread does not block — only subsequent work queued on <paramref name="stream"/> is gated. <paramref name="flags"/> = 0 is always the right value (CU_EVENT_WAIT_DEFAULT).</summary>
    [LibraryImport(LibName)]
    internal static partial int cuStreamWaitEvent(nint stream, nint hEvent, uint flags);

    // ── Event Management ────────────────────────────────────────────────

    /// <summary>Creates a CUDA event. <c>CU_EVENT_DISABLE_TIMING</c> (=2) is the right flag for sync-only events; we never need the timing data and disabling it avoids a tiny bit of driver bookkeeping.</summary>
    [LibraryImport(LibName)]
    internal static partial int cuEventCreate(out nint phEvent, uint flags);

    [LibraryImport(LibName)]
    internal static partial int cuEventDestroy(nint hEvent);

    /// <summary>Records the event when <paramref name="hStream"/> reaches this point in its queue. The event is "complete" once all preceding work on that stream finishes; other streams can wait on it via <see cref="cuStreamWaitEvent"/>.</summary>
    [LibraryImport(LibName)]
    internal static partial int cuEventRecord(nint hEvent, nint hStream);

    [LibraryImport(LibName)]
    internal static partial int cuEventSynchronize(nint hEvent);

    // ── Graph Management (capture / replay) ─────────────────────────────
    //
    // A captured graph records a fixed sequence of stream work and replays it with
    // a single cuGraphLaunch, collapsing thousands of per-kernel CPU launch calls
    // into one. Capture forbids synchronous operations on the stream (no CPU
    // readback, no blocking sync, no sync cuMemAlloc) for its whole duration — so
    // only graphable fixed kernel chains qualify, not a step that interleaves
    // CPU-side scheduler/CFG math or lazy device-to-host syncs.

    /// <summary>Begins capturing work issued to <paramref name="stream"/> into a graph. Mode: GLOBAL=0, THREAD_LOCAL=1, RELAXED=2.</summary>
    [LibraryImport(LibName, EntryPoint = "cuStreamBeginCapture_v2")]
    internal static partial int cuStreamBeginCapture(nint stream, int mode);

    [LibraryImport(LibName)]
    internal static partial int cuStreamEndCapture(nint stream, out nint graph);

    /// <summary>Returns the capture status of a stream (0 = none, 1 = active, 2 = invalidated).</summary>
    [LibraryImport(LibName)]
    internal static partial int cuStreamIsCapturing(nint stream, out int captureStatus);

    /// <summary>Instantiates an executable graph. <paramref name="flags"/> = 0 is the default.</summary>
    [LibraryImport(LibName, EntryPoint = "cuGraphInstantiateWithFlags")]
    internal static partial int cuGraphInstantiate(out nint graphExec, nint graph, ulong flags);

    [LibraryImport(LibName)]
    internal static partial int cuGraphLaunch(nint graphExec, nint stream);

    /// <summary>Enumerates a graph's nodes. Call with nodes = null (nint.Zero array semantics) via the count-query overload first: pass numNodes by ref; when <paramref name="nodes"/> is null the count is returned. Diagnostic use only (HARTSY_GRAPH_DUMP).</summary>
    [LibraryImport(LibName)]
    internal static partial int cuGraphGetNodes(nint graph, [In, Out] nint[]? nodes, ref nuint numNodes);

    /// <summary>Returns a node's CUgraphNodeType (0=kernel, 1=memcpy, 2=memset, 3=host, 4=child graph, 5=empty, 6=wait event, 7=event record, 8=ext-sem signal, 9=ext-sem wait, 10=mem alloc, 11=mem free, 12=batch memop, 13=conditional). Diagnostic use only.</summary>
    [LibraryImport(LibName)]
    internal static partial int cuGraphNodeGetType(nint node, out int nodeType);

    /// <summary>Updates an instantiated graph in place from a re-captured graph of identical topology. Returns CUDA_SUCCESS on success; a non-zero result means the topology changed and the caller must re-instantiate.</summary>

    [LibraryImport(LibName)]
    internal static partial int cuGraphExecDestroy(nint graphExec);

    [LibraryImport(LibName)]
    internal static partial int cuGraphDestroy(nint graph);

    /// <summary>Writes a DOT-format dump of the graph's nodes (kernels, memcpys, alloc/free nodes with their parameters when flags=1 VERBOSE). Diagnostic only.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int cuGraphDebugDotPrint(nint graph, string path, uint flags);

    /// <summary>Returns the device's graph-memory-pool reserves to the OS. Memory allocated by captured allocation nodes (stream-ordered allocs recorded during graph capture) lives in a per-device GRAPH pool that <c>cuGraphExecDestroy</c> does NOT release and <c>cuMemPoolTrimTo</c> does not touch — without this call a destroyed multi-GB step graph keeps its whole working set reserved (invisible to the allocator, fatal for the next model's load on a full card).</summary>
    [LibraryImport(LibName)]
    internal static partial int cuDeviceGraphMemTrim(int device);

    internal const int CU_STREAM_CAPTURE_MODE_GLOBAL = 0;
    internal const int CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1;
    internal const int CU_STREAM_CAPTURE_MODE_RELAXED = 2;

    // ── Memory Info ─────────────────────────────────────────────────────

    /// <summary>Returns the free and total amount of memory available for allocation by the CUDA context. Reports the values for the calling context's device.</summary>
    [LibraryImport(LibName, EntryPoint = "cuMemGetInfo_v2")]
    internal static partial int cuMemGetInfo(out nuint free, out nuint total);

    // ── Profiler Control ────────────────────────────────────────────────
    //
    // Used by `nsys profile --capture-range=cudaProfilerApi` to scope the trace to a window
    // bounded by cuProfilerStart() and cuProfilerStop(). Wrapped by `CudaProfilerControl`.

    [LibraryImport(LibName, EntryPoint = "cuProfilerStart")]
    internal static partial int cuProfilerStart();

    [LibraryImport(LibName, EntryPoint = "cuProfilerStop")]
    internal static partial int cuProfilerStop();

    // ── Memory Pool Management ──────────────────────────────────────────
    //
    // The stream-ordered allocator (cuMemAllocAsync / cuMemFreeAsync) uses a
    // per-device "default mempool" that is SEPARATE from the regular driver allocator
    // that sync cuMemAlloc/cuMemFree use. Memory freed via cuMemFreeAsync is returned
    // to the mempool but stays reserved by it — sync cuMemAlloc cannot see those bytes
    // until the mempool releases them back to the driver. cuMemPoolTrimTo forces that
    // release. We need this every time we transition out of a streaming phase into
    // an eager-allocation phase (e.g. transformer-streaming → VAE-decode-eager) so the
    // VAE sees the memory the streamer just freed.

    /// <summary>Gets the default memory pool of the specified device — used by cuMemAllocAsync / cuMemFreeAsync when no explicit pool is given.</summary>
    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetDefaultMemPool(out nint pool, int dev);

    /// <summary>Releases memory back to the OS until the pool's reserved size is at most <paramref name="minBytesToKeep"/>. Pass 0 to release everything not currently in use by an in-flight async allocation.</summary>
    [LibraryImport(LibName)]
    internal static partial int cuMemPoolTrimTo(nint pool, nuint minBytesToKeep);

    /// <summary>Sets a memory pool attribute. Used to configure <c>CU_MEMPOOL_ATTR_RELEASE_THRESHOLD</c> to 0 so the pool releases reserved memory back to the driver on every stream sync — without this, the default behaviour on some drivers is to hold pool memory indefinitely, starving subsequent sync <c>cuMemAlloc</c> calls.</summary>
    [LibraryImport(LibName)]
    internal static unsafe partial int cuMemPoolSetAttribute(nint pool, int attr, void* value);

    /// <summary>Memory pool attribute: amount of reserved memory in bytes to hold onto before trying to release back to the OS on the next sync. Value type is <c>cuuint64_t</c> (a 64-bit unsigned integer). Set to 0 to be aggressive.</summary>
    internal const int CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4;

    // ── Error Handling ──────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuGetErrorName(int error, out nint pStr);

    [LibraryImport(LibName)]
    internal static partial int cuGetErrorString(int error, out nint pStr);

    // ── Device Attribute Constants ──────────────────────────────────────

    internal const int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1;
    internal const int CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2;
    internal const int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5;
    internal const int CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8;
    internal const int CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10;
    internal const int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
    internal const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
    internal const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
    internal const int CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81;

    // ── P2P Attribute Constants (cuDeviceGetP2PAttribute) ───────────────

    internal const int CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 1;
    internal const int CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 2;
    internal const int CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 3;

    // ── Stream Flags ────────────────────────────────────────────────────

    internal const uint CU_STREAM_DEFAULT = 0;
    internal const uint CU_STREAM_NON_BLOCKING = 1;

    // ── Event Flags ─────────────────────────────────────────────────────

    /// <summary>Sync-only event; saves a tiny bit of driver bookkeeping vs default.</summary>
    internal const uint CU_EVENT_DISABLE_TIMING = 2;

    /// <summary>Default flag for <see cref="cuStreamWaitEvent"/>.</summary>
    internal const uint CU_EVENT_WAIT_DEFAULT = 0;
}

using HartsyInference.Core.Logging;

namespace HartsyInference.Cuda;

/// <summary>Captures a fixed sequence of CUDA stream work into a graph and replays it with a single <c>cuGraphLaunch</c>, collapsing thousands of per-kernel CPU launch calls into one.</summary>
/// <remarks>
/// <para>The launch-overhead win is largest on workloads that issue the same kernel topology every iteration
/// (e.g. a diffusion denoise step).</para>
///
/// <para><b>Capture constraints.</b> Stream capture records device work only — it forbids synchronous operations
/// for the capture's whole duration: no CPU readback (no <c>DataPointer</c>/lazy device-to-host sync), no
/// blocking stream sync, and no synchronous <c>cuMemAlloc</c> (use the stream-ordered allocator, which is
/// capturable as graph-memory nodes, or pre-allocate all buffers before capture). Only a graphable fixed kernel
/// chain qualifies; a pipeline step that interleaves CPU-side scheduler/CFG math cannot be captured wholesale
/// without first hoisting that math out of the captured region.</para>
///
/// <para><b>Buffer pointers and shapes are frozen at instantiation.</b> When parameters change between
/// iterations, re-capture into a fresh graph and re-instantiate.</para>
///
/// <para><b>Untested locally.</b> Written from the CUDA Driver API graph docs; not exercised on GPU in this
/// environment. Validate on hardware before relying on it in a pipeline.</para>
/// </remarks>
public sealed class CudaGraph : IDisposable
{
    /// <summary>Base pointer of this graph's capture arena (0 = none) — set by CudaBackend.CaptureGraph, released via GpuTransferHelper.FreeGraphArena in DisposeGraph.</summary>
    internal ulong ArenaBase;

    /// <summary><c>CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH</c>: the executable graph automatically frees any memory its allocation nodes produced in the previous launch before relaunching. Required to replay a graph that captured stream-ordered activation allocations (the backend's per-op <c>cuMemAllocAsync</c>) more than once — without it the second <c>cuGraphLaunch</c> re-allocs still-live graph memory and fails with INVALID_VALUE. Addresses stay stable across launches, so pointers cached at capture time remain valid.</summary>
    private const ulong FlagAutoFreeOnLaunch = 1;

    private readonly nint _stream;
    private readonly ulong _instantiateFlags;
    private nint _graphExec;
    private int _disposed;

    /// <summary>Whether an executable graph has been captured and is ready to <see cref="Launch"/>.</summary>
    public bool IsReady => _graphExec != 0;

    /// <summary>Creates a graph bound to the stream its work will be captured on and replayed to. Set <paramref name="autoFreeAllocationsOnRelaunch"/> when the captured work allocates memory (e.g. the backend's per-op activation allocations) and the graph will be launched more than once — the loop case.</summary>
    public CudaGraph(nint stream, bool autoFreeAllocationsOnRelaunch = false)
    {
        if (stream == 0) throw new ArgumentException("Stream handle must be non-zero.", nameof(stream));
        _stream = stream;
        _instantiateFlags = autoFreeAllocationsOnRelaunch ? FlagAutoFreeOnLaunch : 0;
    }

    /// <summary>Captures the device work issued by <paramref name="recordWork"/> on this graph's stream and instantiates it. The delegate must issue only capturable (asynchronous) work — see the type remarks. Replaces any previously captured graph.</summary>
    public void Capture(Action recordWork)
    {
        if (recordWork is null) throw new ArgumentNullException(nameof(recordWork));
        ThrowIfDisposed();

        CudaDriverApi.cuStreamBeginCapture(_stream, CudaDriverApi.CU_STREAM_CAPTURE_MODE_GLOBAL).ThrowOnError();
        nint graph = 0;
        try
        {
            recordWork();
        }
        catch (Exception ex)
        {
            // End capture to leave the stream in a clean (non-capturing) state before
            // propagating — otherwise the stream stays poisoned for all later work.
            List<Exception> failures = [ex];
            try { CudaDriverApi.cuStreamEndCapture(_stream, out graph).ThrowOnError(); }
            catch (Exception cleanup) { failures.Add(cleanup); }
            if (graph != 0)
            {
                try { DestroyGraph(graph, throwOnError: true); }
                catch (Exception cleanup) { failures.Add(cleanup); }
            }
            Logs.Error("CUDA graph capture delegate threw; capture aborted.", ex);
            if (failures.Count > 1)
                throw new AggregateException("CUDA graph capture and abort both failed.", failures);
            throw;
        }

        CudaDriverApi.cuStreamEndCapture(_stream, out graph).ThrowOnError();
        try
        {
            DumpNodeHistogram(graph);
            DestroyExec();
            CudaDriverApi.cuGraphInstantiate(out _graphExec, graph, _instantiateFlags).ThrowOnError();
        }
        catch (Exception primary)
        {
            try { DestroyGraph(graph, throwOnError: true); }
            catch (Exception cleanup)
            {
                throw new AggregateException("CUDA graph instantiation and source-graph cleanup both failed.", primary, cleanup);
            }
            throw;
        }
        DestroyGraph(graph, throwOnError: true);
    }

    /// <summary>HARTSY_GRAPH_DUMP=1: logs the captured graph's node count and per-type histogram — the direct measurement of how many kernel vs mem-alloc/free vs other nodes a decode step replays (alloc/free nodes come from per-intermediate AllocateDevice/Dispose during capture).</summary>
    private static void DumpNodeHistogram(nint graph)
    {
        if (Environment.GetEnvironmentVariable("HARTSY_GRAPH_DUMP") != "1") return;
        nuint count = 0;
        if (CudaDriverApi.cuGraphGetNodes(graph, null, ref count) != 0 || count == 0) return;
        nint[] nodes = new nint[count];
        if (CudaDriverApi.cuGraphGetNodes(graph, nodes, ref count) != 0) return;
        Dictionary<int, int> hist = new();
        for (int i = 0; i < (int)count; i++)
        {
            if (CudaDriverApi.cuGraphNodeGetType(nodes[i], out int t) != 0) t = -1;
            hist[t] = hist.GetValueOrDefault(t) + 1;
        }
        string[] names = ["kernel", "memcpy", "memset", "host", "child", "empty", "waitEvent", "recordEvent",
            "semSignal", "semWait", "memAlloc", "memFree", "batchMemOp", "conditional"];
        string parts = string.Join(", ", hist.OrderByDescending(kv => kv.Value)
            .Select(kv => $"{(kv.Key >= 0 && kv.Key < names.Length ? names[kv.Key] : $"type{kv.Key}")}={kv.Value}"));
        Logs.Info($"[CudaGraph] captured {count} nodes: {parts}");
    }


    /// <summary>Starts stream capture directly (the open-region twin of <see cref="Capture"/>, for callers whose captured work spans multiple methods). Pair with <see cref="EndCaptureAndInstantiate"/>; on any exception between the two, call <see cref="AbortCapture"/> so the stream doesn't stay in capture mode.</summary>
    public void BeginCapture()
    {
        ThrowIfDisposed();
        // RELAXED (the mode ggml uses for its denoise-step capture): other threads' non-capturable CUDA calls
        // (a second backend, the streaming cache) aren't failed by this thread's capture.
        CudaDriverApi.cuStreamBeginCapture(_stream, CudaDriverApi.CU_STREAM_CAPTURE_MODE_RELAXED).ThrowOnError();
    }

    /// <summary>Ends an open capture region and instantiates the executable graph. NOTE: the work recorded during capture did NOT execute — call <see cref="Launch"/> to run it.</summary>
    public void EndCaptureAndInstantiate()
    {
        ThrowIfDisposed();
        CudaDriverApi.cuStreamEndCapture(_stream, out nint graph).ThrowOnError();
        try
        {
            string? dotPath = Environment.GetEnvironmentVariable("HARTSY_GRAPH_DOT");
            if (!string.IsNullOrEmpty(dotPath))
            {
                int dotRc = CudaDriverApi.cuGraphDebugDotPrint(graph, dotPath, 1);
                Logs.Info($"[CudaGraph] DOT dump rc={dotRc} → {dotPath}");
            }
            DestroyExec();
            CudaDriverApi.cuGraphInstantiate(out _graphExec, graph, _instantiateFlags).ThrowOnError();
        }
        catch (Exception primary)
        {
            try { DestroyGraph(graph, throwOnError: true); }
            catch (Exception cleanup)
            {
                throw new AggregateException("CUDA graph instantiation and source-graph cleanup both failed.", primary, cleanup);
            }
            throw;
        }
        DestroyGraph(graph, throwOnError: true);
    }

    /// <summary>Aborts an open capture region, leaving the stream clean (call from exception handlers between <see cref="BeginCapture"/> and <see cref="EndCaptureAndInstantiate"/>).</summary>
    public void AbortCapture()
    {
        List<Exception>? failures = null;
        nint graph = 0;
        try { CudaDriverApi.cuStreamEndCapture(_stream, out graph).ThrowOnError(); }
        catch (Exception error) { (failures ??= []).Add(error); }
        if (graph != 0)
        {
            try { DestroyGraph(graph, throwOnError: true); }
            catch (Exception error) { (failures ??= []).Add(error); }
        }
        if (failures is not null)
            throw new AggregateException("One or more CUDA graph capture-abort operations failed.", failures);
    }

    /// <summary>Drops the captured executable graph (shape/prompt change → recapture needed).</summary>
    public void Reset() => DestroyExec();

    /// <summary>Replays the captured graph on its stream with a single launch call.</summary>
    public void Launch()
    {
        ThrowIfDisposed();
        if (_graphExec == 0)
            throw new InvalidOperationException("CudaGraph.Launch called before Capture. Capture a graph first.");
        CudaDriverApi.cuGraphLaunch(_graphExec, _stream).ThrowOnError();
    }

    private void DestroyExec(bool throwOnError = true)
    {
        nint graphExec = Interlocked.Exchange(ref _graphExec, 0);
        if (graphExec != 0)
        {
            int result = CudaDriverApi.cuGraphExecDestroy(graphExec);
            if (throwOnError) result.ThrowOnError();
        }
    }

    private static void DestroyGraph(nint graph, bool throwOnError)
    {
        if (graph == 0) return;
        int result = CudaDriverApi.cuGraphDestroy(graph);
        if (throwOnError) result.ThrowOnError();
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
            throw new ObjectDisposedException(nameof(CudaGraph));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        try
        {
            DestroyExec(throwOnError: true);
        }
        finally
        {
            GC.SuppressFinalize(this);
        }
    }

    ~CudaGraph()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        try { DestroyExec(throwOnError: false); }
        catch { /* Finalizers must never surface native-loader or driver failures. */ }
    }
}

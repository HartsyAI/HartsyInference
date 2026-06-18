using HartsyInference.Core.Logging;

namespace HartsyInference.Cuda;

/// <summary>Captures a fixed sequence of CUDA stream work into a graph and replays it with a single <c>cuGraphLaunch</c>, collapsing thousands of per-kernel CPU launch calls into one. The launch-overhead win is largest on workloads that issue the same kernel topology every iteration (e.g. a diffusion denoise step).
///
/// <para><b>Capture constraints.</b> Stream capture records device work only — it forbids synchronous operations for the capture's whole duration: no CPU readback (no <c>DataPointer</c>/lazy device-to-host sync), no blocking stream sync, and no synchronous <c>cuMemAlloc</c> (use the stream-ordered allocator, which is capturable as graph-memory nodes, or pre-allocate all buffers before capture). Only a graphable fixed kernel chain qualifies; a pipeline step that interleaves CPU-side scheduler/CFG math cannot be captured wholesale without first hoisting that math out of the captured region.</para>
///
/// <para><b>Re-capture vs update.</b> Buffer pointers and shapes are frozen at instantiation. When only scalar kernel parameters change between iterations (timestep, sigma, CFG scale), prefer re-capturing into a fresh graph and calling <see cref="TryUpdate"/> rather than re-instantiating, which is cheaper when the topology is identical.</para>
///
/// <para><b>Untested locally.</b> Written from the CUDA Driver API graph docs; not exercised on GPU in this environment. Validate on hardware before relying on it in a pipeline.</para></summary>
public sealed class CudaGraph : IDisposable
{
    private readonly nint _stream;
    private nint _graphExec;
    private int _disposed;

    /// <summary>Whether an executable graph has been captured and is ready to <see cref="Launch"/>.</summary>
    public bool IsReady => _graphExec != 0;

    /// <summary>Creates a graph bound to the stream its work will be captured on and replayed to.</summary>
    public CudaGraph(nint stream)
    {
        if (stream == 0) throw new ArgumentException("Stream handle must be non-zero.", nameof(stream));
        _stream = stream;
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
            CudaDriverApi.cuStreamEndCapture(_stream, out graph);
            if (graph != 0) CudaDriverApi.cuGraphDestroy(graph);
            Logs.Error("CUDA graph capture delegate threw; capture aborted.", ex);
            throw;
        }

        CudaDriverApi.cuStreamEndCapture(_stream, out graph).ThrowOnError();
        try
        {
            DestroyExec();
            CudaDriverApi.cuGraphInstantiate(out _graphExec, graph, 0).ThrowOnError();
        }
        finally
        {
            CudaDriverApi.cuGraphDestroy(graph);
        }
    }

    /// <summary>Re-captures <paramref name="recordWork"/> and updates the existing executable graph in place when the topology is unchanged (cheaper than re-instantiating). Falls back to a full re-instantiate when the topology differs. No-op-safe to call before the first <see cref="Capture"/> (it just captures).</summary>
    public void TryUpdate(Action recordWork)
    {
        if (recordWork is null) throw new ArgumentNullException(nameof(recordWork));
        ThrowIfDisposed();
        if (_graphExec == 0)
        {
            Capture(recordWork);
            return;
        }

        CudaDriverApi.cuStreamBeginCapture(_stream, CudaDriverApi.CU_STREAM_CAPTURE_MODE_GLOBAL).ThrowOnError();
        recordWork();
        CudaDriverApi.cuStreamEndCapture(_stream, out nint graph).ThrowOnError();
        try
        {
            int rc = CudaDriverApi.cuGraphExecUpdate(_graphExec, graph, 0);
            if (rc != 0)
            {
                // Topology changed — re-instantiate from the fresh graph.
                DestroyExec();
                CudaDriverApi.cuGraphInstantiate(out _graphExec, graph, 0).ThrowOnError();
            }
        }
        finally
        {
            CudaDriverApi.cuGraphDestroy(graph);
        }
    }

    /// <summary>Replays the captured graph on its stream with a single launch call.</summary>
    public void Launch()
    {
        ThrowIfDisposed();
        if (_graphExec == 0)
            throw new InvalidOperationException("CudaGraph.Launch called before Capture. Capture a graph first.");
        CudaDriverApi.cuGraphLaunch(_graphExec, _stream).ThrowOnError();
    }

    private void DestroyExec()
    {
        if (_graphExec != 0)
        {
            CudaDriverApi.cuGraphExecDestroy(_graphExec);
            _graphExec = 0;
        }
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
            throw new ObjectDisposedException(nameof(CudaGraph));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        DestroyExec();
        GC.SuppressFinalize(this);
    }
}

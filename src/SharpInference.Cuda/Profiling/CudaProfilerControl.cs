namespace SharpInference.Cuda.Profiling;

/// <summary>Wrappers around <c>cuProfilerStart</c> / <c>cuProfilerStop</c>. Used in conjunction with
/// <c>nsys profile --capture-range=cudaProfilerApi</c> to bound an Nsight Systems trace to the
/// steady-state measurement window only — typically the inner steps of a denoise loop, after
/// warmup has settled.
///
/// <para>Example:</para>
/// <code>
/// // Warmup (not captured)
/// for (int i = 0; i &lt; 3; i++) RunStep();
///
/// using (CudaProfilerControl.Window())
/// {
///     // Captured by nsys when --capture-range=cudaProfilerApi is set
///     for (int i = 0; i &lt; 10; i++) RunStep();
/// }
/// </code></summary>
public static class CudaProfilerControl
{
    /// <summary>Starts the CUDA profiler. The matching <see cref="Stop"/> closes the window.</summary>
    public static void Start()
    {
        int rc = CudaDriverApi.cuProfilerStart();
        // Don't throw: the call returns CUDA_ERROR_PROFILER_DISABLED when nsys isn't actually
        // capturing, which is fine.
        _ = rc;
    }

    /// <summary>Stops the CUDA profiler.</summary>
    public static void Stop()
    {
        int rc = CudaDriverApi.cuProfilerStop();
        _ = rc;
    }

    /// <summary>RAII helper: <c>using (CudaProfilerControl.Window())</c> calls <see cref="Start"/>
    /// on entry and <see cref="Stop"/> on exit.</summary>
    public static ProfilerWindow Window() => new();

    /// <summary>Disposable returned by <see cref="Window"/>.</summary>
    public readonly struct ProfilerWindow : IDisposable
    {
        /// <summary>Creates the window and starts the profiler.</summary>
        public ProfilerWindow()
        {
            Start();
        }

        /// <summary>Stops the profiler.</summary>
        public void Dispose() => Stop();
    }
}

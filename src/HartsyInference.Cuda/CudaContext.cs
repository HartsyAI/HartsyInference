using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace HartsyInference.Cuda;

/// <summary>Manages a CUDA context for a specific device. Handles initialization, context creation, and device info queries. <para>Uses the CUDA Driver API <c>cuDevicePrimaryCtxRetain</c> rather than <c>cuCtxCreate</c>: the primary context is one-per-device-per-process, refcounted, and shared with the CUDA Runtime API. This avoids the thread-stack push side-effect of <c>cuCtxCreate</c> (which silently binds the context only to the constructing thread) and lets any thread in the process bind the same context via <see cref="EnsureCurrent"/>.</para> <para>Thread affinity in the Driver API is explicit: every thread that issues CUDA API calls must have the context current on it (via <c>cuCtxSetCurrent</c>). The <see cref="EnsureCurrent"/> method below uses a <c>[ThreadStatic]</c> cache so the binding is set once per worker thread; subsequent calls on the same thread are a TLS read + branch.</para></summary>
public sealed class CudaContext : IDisposable
{
    private static int _cudaInitialized;

    /// <summary>The primary-context handle currently bound to the calling thread, or 0 if the thread has never been bound. ThreadStatic so each worker maintains its own notion of "what's already current here."</summary>
    [ThreadStatic]
    private static nint _threadCurrentContext;

    /// <summary>The generation of the context bound to the calling thread. Pairs with <see cref="_threadCurrentContext"/>. The CUDA driver reuses primary-context handles after release+retain — so handle equality alone isn't enough to know the binding is still valid. Bumping a process-wide generation on every new <see cref="CudaContext"/> instance and checking the per-thread generation here forces a re-bind whenever the underlying context object has been recreated.</summary>
    [ThreadStatic]
    private static long _threadCurrentGeneration;

    /// <summary>Process-wide monotonic counter; each new <see cref="CudaContext"/> instance gets a fresh generation regardless of whether the driver reused the context handle.</summary>
    private static long _generationCounter;

    private nint _context;
    private readonly long _generation;
    private readonly int _deviceOrdinal;
    private readonly int _deviceHandle;

    /// <summary>The CUDA device ordinal.</summary>
    public int DeviceOrdinal => _deviceOrdinal;

    /// <summary>The CUdevice handle (from <c>cuDeviceGet</c>) — for device-scoped driver calls like <c>cuDeviceGraphMemTrim</c>.</summary>
    internal int DeviceHandle => _deviceHandle;

    /// <summary>The raw driver context handle. Used as the per-backend state key in <see cref="GpuTransferHelper"/>/<see cref="CudaMemory"/> (primary contexts: one handle per device).</summary>
    internal nint Handle => _context;

    /// <summary>The compute capability major version.</summary>
    public int ComputeCapabilityMajor { get; }

    /// <summary>The compute capability minor version.</summary>
    public int ComputeCapabilityMinor { get; }

    /// <summary>Total VRAM in bytes.</summary>
    public nuint TotalMemory { get; }

    /// <summary>Number of streaming multiprocessors.</summary>
    public int MultiprocessorCount { get; }

    /// <summary>Device name.</summary>
    public string DeviceName { get; }

    /// <summary>Creates a CUDA context for the specified device ordinal. Retains the device's primary context (refcounted; safe to call from multiple <see cref="CudaContext"/> instances pointing at the same device) and binds it to the calling thread.</summary>
    public CudaContext(int deviceOrdinal = 0)
    {
        CudaLibraryResolver.Register();
        EnsureCudaInitialized();

        _deviceOrdinal = deviceOrdinal;
        _generation = Interlocked.Increment(ref _generationCounter);

        CudaDriverApi.cuDeviceGet(out _deviceHandle, deviceOrdinal).ThrowOnError();
        CudaDriverApi.cuDevicePrimaryCtxRetain(out _context, _deviceHandle).ThrowOnError();
        // Bind to the constructing thread so device queries below (and any work the
        // caller does immediately after construction) see a current context.
        CudaDriverApi.cuCtxSetCurrent(_context).ThrowOnError();
        _threadCurrentContext = _context;
        _threadCurrentGeneration = _generation;

        // Query device properties
        CudaDriverApi.cuDeviceGetAttribute(
            out int major, CudaDriverApi.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _deviceHandle).ThrowOnError();
        ComputeCapabilityMajor = major;

        CudaDriverApi.cuDeviceGetAttribute(
            out int minor, CudaDriverApi.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _deviceHandle).ThrowOnError();
        ComputeCapabilityMinor = minor;

        CudaDriverApi.cuDeviceTotalMem(out nuint totalMem, _deviceHandle).ThrowOnError();
        TotalMemory = totalMem;

        CudaDriverApi.cuDeviceGetAttribute(
            out int smCount, CudaDriverApi.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, _deviceHandle).ThrowOnError();
        MultiprocessorCount = smCount;

        DeviceName = QueryDeviceName(_deviceHandle);
    }

    /// <summary>Ensures this context is current on the calling thread.</summary>
    /// <remarks>Cheap fast path on already-bound threads (one TLS read + branch). Call from the entry of any code
    /// that issues CUDA Driver API calls — the single guarantee that makes the backend safe to use from
    /// <c>Task.Run</c> worker threads, finalizers, and async continuations.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void EnsureCurrent()
    {
        // Tensor finalizer cleanups are NOT drained here anymore: the cleanup buckets are keyed by backend
        // StateKey (two same-device backends share this context but not their buckets), so only the owning
        // backend can drain its own — CudaBackend.EnterOp does, on every op entry, right after binding here.
        // Both checks matter: handle equality alone is unsafe because the CUDA
        // driver reuses primary-context handles after release+retain — a thread
        // could see a numerically-equal handle that's actually backed by a freshly
        // recreated context. The generation counter forces a re-bind in that case.
        if (_threadCurrentContext == _context && _threadCurrentGeneration == _generation)
        {
            return;
        }
        if (_context == 0)
        {
            throw new ObjectDisposedException(nameof(CudaContext));
        }
        CudaDriverApi.cuCtxSetCurrent(_context).ThrowOnError();
        _threadCurrentContext = _context;
        _threadCurrentGeneration = _generation;
    }

    /// <summary>Forces a re-bind of this context on the calling thread, bypassing the TLS cache. Use when external code may have changed the current context out from under us (e.g. another library that calls <c>cuCtxSetCurrent</c> on the same thread).</summary>
    public void MakeCurrent()
    {
        if (_context == 0)
        {
            throw new ObjectDisposedException(nameof(CudaContext));
        }
        CudaDriverApi.cuCtxSetCurrent(_context).ThrowOnError();
        _threadCurrentContext = _context;
        _threadCurrentGeneration = _generation;
    }

    /// <summary>Binds an independently-retained handle for this same device primary context. Unlike <see cref="EnsureCurrent"/>, this remains valid after this managed owner has been explicitly disposed, so a child native resource can release its own primary-context lease without relying on finalizer order.</summary>
    internal void EnsureRetainedCurrent(nint retainedContext)
    {
        if (retainedContext == 0)
            throw new ObjectDisposedException(nameof(CudaContext), "The retained CUDA context lease was already released.");
        if (_threadCurrentContext == retainedContext && _threadCurrentGeneration == _generation)
            return;
        CudaDriverApi.cuCtxSetCurrent(retainedContext).ThrowOnError();
        _threadCurrentContext = retainedContext;
        _threadCurrentGeneration = _generation;
    }

    /// <summary>Synchronizes the entire context (all streams).</summary>
    public void Synchronize()
    {
        EnsureCurrent();
        CudaDriverApi.cuCtxSynchronize().ThrowOnError();
    }

    /// <summary>Queries free / total device memory via <c>cuMemGetInfo</c>. Free memory is what's actually allocatable right now — it accounts for VRAM held by other processes (e.g. another app's CUDA context, the X server, browser GPU acceleration). Use this for fail-skip decisions in tests on memory-constrained hosts.</summary>
    public (nuint freeBytes, nuint totalBytes) GetMemoryInfo()
    {
        EnsureCurrent();
        CudaDriverApi.cuMemGetInfo(out nuint free, out nuint total).ThrowOnError();
        return (free, total);
    }

    /// <summary>Returns the number of CUDA-capable devices in the system.</summary>
    public static int GetDeviceCount()
    {
        CudaLibraryResolver.Register();
        EnsureCudaInitialized();
        CudaDriverApi.cuDeviceGetCount(out int count).ThrowOnError();
        return count;
    }

    /// <summary>Why the last <see cref="IsAvailable"/> call returned <c>false</c>; <c>null</c> when it returned true. Without this the reason is swallowed and a host silently degrades to CPU with nothing to report.</summary>
    public static string? LastUnavailableReason { get; private set; }

    /// <summary>Probes whether a <see cref="CudaBackend"/> can be constructed in this environment.</summary>
    /// <remarks>Returns <c>false</c> if any of: the driver library is missing (<c>libcuda.so.1</c> / <c>nvcuda.dll</c>),
    /// cuBLAS is missing (<c>libcublas.so.13</c>/<c>.12</c>/<c>.11</c> on Linux or <c>cublas64_13/12/11.dll</c> on
    /// Windows — typically requires the CUDA Toolkit, not just the driver), <c>cuInit</c> fails, or no CUDA-capable
    /// devices are present. Used by tests to skip cleanly when CUDA isn't fully usable, mirroring the
    /// <c>VulkanAvailable</c> pattern.</remarks>
    public static bool IsAvailable()
    {
        try
        {
            CudaLibraryResolver.Register();

            // Probe cuBLAS up front — the driver may be present (e.g., NVIDIA GL extension) without
            // the toolkit, in which case CudaBackend construction would fail at cublasCreate. Uses the
            // resolver's own search (probe dirs like ~/.local/lib/cuda13 + soname fallback) so this
            // check can't disagree with what actual P/Invoke resolution would find.
            if (!CudaLibraryResolver.CublasLoadable())
            {
                LastUnavailableReason = "cuBLAS could not be loaded (libcublas.so.13/.12/.11 — usually means the CUDA Toolkit is missing, not just the driver).";
                return false;
            }

            EnsureCudaInitialized();
            CudaDriverApi.cuDeviceGetCount(out int count).ThrowOnError();
            if (count <= 0)
            {
                LastUnavailableReason = "the CUDA driver loaded but reported 0 CUDA-capable devices.";
                return false;
            }
            LastUnavailableReason = null;
            return true;
        }
        catch (Exception ex)
        {
            LastUnavailableReason = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private static void EnsureCudaInitialized()
    {
        if (Interlocked.CompareExchange(ref _cudaInitialized, 1, 0) == 0)
        {
            CudaDriverApi.cuInit(0).ThrowOnError();
        }
    }

    private static string QueryDeviceName(int deviceHandle)
    {
        nint nameBuffer = Marshal.AllocHGlobal(256);
        try
        {
            CudaDriverApi.cuDeviceGetName(nameBuffer, 256, deviceHandle).ThrowOnError();
            return Marshal.PtrToStringAnsi(nameBuffer) ?? "Unknown";
        }
        finally
        {
            Marshal.FreeHGlobal(nameBuffer);
        }
    }

    public void Dispose()
    {
        try
        {
            ReleasePrimaryContext(throwOnError: true);
        }
        finally
        {
            GC.SuppressFinalize(this);
        }
    }

    ~CudaContext()
    {
        try { ReleasePrimaryContext(throwOnError: false); }
        catch { /* Finalizers must never surface native-loader or driver failures. */ }
    }

    /// <summary>Releases the device's primary context (decrements the refcount). When the count hits zero, the driver tears the context down. Safe to call from the finalizer thread because primary-context release isn't thread-bound the way <c>cuCtxDestroy</c> would be.</summary>
    private void ReleasePrimaryContext(bool throwOnError)
    {
        nint ctx = Interlocked.Exchange(ref _context, 0);
        if (ctx != 0)
        {
            // Clear the TLS marker if the disposing thread still thought it held this
            // context — otherwise a later EnsureCurrent on this thread would skip the
            // re-bind on a different (later-constructed) CudaContext. Generation is
            // cleared too so the per-thread state is unambiguous (matches the "no
            // context bound" sentinel of generation=0 + handle=0).
            if (_threadCurrentContext == ctx && _threadCurrentGeneration == _generation)
            {
                _threadCurrentContext = 0;
                _threadCurrentGeneration = 0;
            }
            int result = CudaDriverApi.cuDevicePrimaryCtxRelease(_deviceHandle);
            if (throwOnError) result.ThrowOnError();
        }
    }
}

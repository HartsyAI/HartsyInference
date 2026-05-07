using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SharpInference.Cuda;

/// <summary>Manages a CUDA context for a specific device. Handles initialization, context creation, and device info queries.
///
/// <para>Uses the CUDA Driver API <c>cuDevicePrimaryCtxRetain</c> rather than <c>cuCtxCreate</c>:
/// the primary context is one-per-device-per-process, refcounted, and shared with the
/// CUDA Runtime API. This avoids the thread-stack push side-effect of <c>cuCtxCreate</c>
/// (which silently binds the context only to the constructing thread) and lets any thread
/// in the process bind the same context via <see cref="EnsureCurrent"/>.</para>
///
/// <para>Thread affinity in the Driver API is explicit: every thread that issues CUDA
/// API calls must have the context current on it (via <c>cuCtxSetCurrent</c>). The
/// <see cref="EnsureCurrent"/> method below uses a <c>[ThreadStatic]</c> cache so the
/// binding is set once per worker thread; subsequent calls on the same thread are a
/// TLS read + branch.</para></summary>
public sealed class CudaContext : IDisposable
{
    private static int _cudaInitialized;

    /// <summary>The primary-context handle currently bound to the calling thread, or 0 if
    /// the thread has never been bound. ThreadStatic so each worker maintains its own
    /// notion of "what's already current here."</summary>
    [ThreadStatic]
    private static nint _threadCurrentContext;

    /// <summary>The generation of the context bound to the calling thread. Pairs with
    /// <see cref="_threadCurrentContext"/>. The CUDA driver reuses primary-context
    /// handles after release+retain — so handle equality alone isn't enough to know
    /// the binding is still valid. Bumping a process-wide generation on every new
    /// <see cref="CudaContext"/> instance and checking the per-thread generation here
    /// forces a re-bind whenever the underlying context object has been recreated.</summary>
    [ThreadStatic]
    private static long _threadCurrentGeneration;

    /// <summary>Process-wide monotonic counter; each new <see cref="CudaContext"/>
    /// instance gets a fresh generation regardless of whether the driver reused
    /// the context handle.</summary>
    private static long _generationCounter;

    private nint _context;
    private readonly long _generation;
    private readonly int _deviceOrdinal;
    private readonly int _deviceHandle;

    /// <summary>The CUDA device ordinal.</summary>
    public int DeviceOrdinal => _deviceOrdinal;

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

    /// <summary>Creates a CUDA context for the specified device ordinal. Retains the
    /// device's primary context (refcounted; safe to call from multiple <see cref="CudaContext"/>
    /// instances pointing at the same device) and binds it to the calling thread.</summary>
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

    /// <summary>
    /// Ensures this context is current on the calling thread. Cheap fast path on already-bound
    /// threads (one TLS read + branch). Call from the entry of any code that issues CUDA
    /// Driver API calls — this is the single guarantee that makes the backend safe to use
    /// from <c>Task.Run</c> worker threads, finalizers, and async continuations.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void EnsureCurrent()
    {
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

    /// <summary>Forces a re-bind of this context on the calling thread, bypassing the TLS
    /// cache. Use when external code may have changed the current context out from under us
    /// (e.g. another library that calls <c>cuCtxSetCurrent</c> on the same thread).</summary>
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

    /// <summary>Synchronizes the entire context (all streams).</summary>
    public void Synchronize()
    {
        EnsureCurrent();
        CudaDriverApi.cuCtxSynchronize().ThrowOnError();
    }

    /// <summary>Queries free / total device memory via <c>cuMemGetInfo</c>. Free memory is what's
    /// actually allocatable right now — it accounts for VRAM held by other processes (e.g. another
    /// app's CUDA context, the X server, browser GPU acceleration). Use this for fail-skip
    /// decisions in tests on memory-constrained hosts.</summary>
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

    /// <summary>Probes whether a <see cref="CudaBackend"/> can be constructed in this environment. Returns <c>false</c> if any of: the driver library is missing (<c>libcuda.so.1</c> / <c>nvcuda.dll</c>), cuBLAS is missing (<c>libcublas.so.13</c>/<c>.12</c>/<c>.11</c> on Linux or <c>cublas64_13/12/11.dll</c> on Windows — typically requires the CUDA Toolkit, not just the driver), <c>cuInit</c> fails, or no CUDA-capable devices are present. Used by tests to skip cleanly when CUDA isn't fully usable, mirroring the <c>VulkanAvailable</c> pattern.</summary>
    public static bool IsAvailable()
    {
        try
        {
            CudaLibraryResolver.Register();

            // Probe cuBLAS up front — the driver may be present (e.g., NVIDIA GL extension) without
            // the toolkit, in which case CudaBackend construction would fail at cublasCreate. Match
            // the version set probed by CudaLibraryResolver.ResolveLibrary("cublas").
            bool cublasOk = false;
            if (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows))
            {
                cublasOk = System.Runtime.InteropServices.NativeLibrary.TryLoad("cublas64_13.dll", out _)
                        || System.Runtime.InteropServices.NativeLibrary.TryLoad("cublas64_12.dll", out _)
                        || System.Runtime.InteropServices.NativeLibrary.TryLoad("cublas64_11.dll", out _);
            }
            else if (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux))
            {
                cublasOk = System.Runtime.InteropServices.NativeLibrary.TryLoad("libcublas.so.13", out _)
                        || System.Runtime.InteropServices.NativeLibrary.TryLoad("libcublas.so.12", out _)
                        || System.Runtime.InteropServices.NativeLibrary.TryLoad("libcublas.so.11", out _);
            }
            if (!cublasOk) return false;

            EnsureCudaInitialized();
            CudaDriverApi.cuDeviceGetCount(out int count).ThrowOnError();
            return count > 0;
        }
        catch
        {
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
        ReleasePrimaryContext();
        GC.SuppressFinalize(this);
    }

    ~CudaContext()
    {
        ReleasePrimaryContext();
    }

    /// <summary>Releases the device's primary context (decrements the refcount). When
    /// the count hits zero, the driver tears the context down. Safe to call from the
    /// finalizer thread because primary-context release isn't thread-bound the way
    /// <c>cuCtxDestroy</c> would be.</summary>
    private void ReleasePrimaryContext()
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
            CudaDriverApi.cuDevicePrimaryCtxRelease(_deviceHandle);
        }
    }
}

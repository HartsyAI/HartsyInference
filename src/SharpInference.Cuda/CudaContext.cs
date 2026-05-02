using System.Runtime.InteropServices;

namespace SharpInference.Cuda;

/// <summary>Manages a CUDA context for a specific device. Handles initialization, context creation, and device info queries.</summary>
public sealed class CudaContext : IDisposable
{
    private static int _cudaInitialized;

    private nint _context;
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

    /// <summary>Creates a CUDA context for the specified device ordinal.</summary>
    public CudaContext(int deviceOrdinal = 0)
    {
        CudaLibraryResolver.Register();
        EnsureCudaInitialized();

        _deviceOrdinal = deviceOrdinal;

        CudaDriverApi.cuDeviceGet(out _deviceHandle, deviceOrdinal).ThrowOnError();
        CudaDriverApi.cuCtxCreate(out _context, 0, _deviceHandle).ThrowOnError();

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

    /// <summary>Makes this context current on the calling thread.</summary>
    public void MakeCurrent()
    {
        if (_context == 0)
            throw new ObjectDisposedException(nameof(CudaContext));
        CudaDriverApi.cuCtxSetCurrent(_context).ThrowOnError();
    }

    /// <summary>Synchronizes the entire context (all streams).</summary>
    public void Synchronize()
    {
        if (_context == 0)
            throw new ObjectDisposedException(nameof(CudaContext));
        CudaDriverApi.cuCtxSynchronize().ThrowOnError();
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
        nint ctx = Interlocked.Exchange(ref _context, 0);
        if (ctx != 0)
        {
            CudaDriverApi.cuCtxDestroy(ctx);
        }
        GC.SuppressFinalize(this);
    }

    ~CudaContext()
    {
        nint ctx = Interlocked.Exchange(ref _context, 0);
        if (ctx != 0)
        {
            CudaDriverApi.cuCtxDestroy(ctx);
        }
    }
}

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

    /// <summary>Probes for a usable CUDA driver + at least one device, without throwing. Returns <c>false</c> if the driver library can't be loaded (e.g. <c>libcuda.so.1</c> / <c>nvcuda.dll</c> missing, or running inside a sandbox that doesn't expose the host driver), if <c>cuInit</c> fails, or if no CUDA-capable devices are present. Mirrors the <c>VulkanAvailable</c> pattern used by the Vulkan tests so CUDA tests can skip cleanly in environments without a working driver.</summary>
    public static bool IsAvailable()
    {
        try
        {
            CudaLibraryResolver.Register();
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

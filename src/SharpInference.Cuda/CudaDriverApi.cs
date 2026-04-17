using System.Runtime.InteropServices;

namespace SharpInference.Cuda;

/// <summary>P/Invoke bindings for the CUDA Driver API. Library name "cuda" is resolved at runtime by CudaLibraryResolver to nvcuda.dll (Windows) or libcuda.so.1 (Linux).</summary>
internal static partial class CudaDriverApi
{
    private const string LibName = "cuda";

    // ── Initialization ──────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuInit(uint flags);

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

    [LibraryImport(LibName, EntryPoint = "cuCtxCreate_v2")]
    internal static partial int cuCtxCreate(out nint pctx, uint flags, int dev);

    [LibraryImport(LibName, EntryPoint = "cuCtxDestroy_v2")]
    internal static partial int cuCtxDestroy(nint ctx);

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
    internal static partial int cuModuleLoadData(out nint module, nint ptxImage);

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

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyHtoDAsync(ulong dst, nint src, nuint bytes, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoHAsync(nint dst, ulong src, nuint bytes, nint stream);

    // ── Stream Management ───────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuStreamCreate(out nint stream, uint flags);

    [LibraryImport(LibName, EntryPoint = "cuStreamDestroy_v2")]
    internal static partial int cuStreamDestroy(nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuStreamSynchronize(nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuStreamQuery(nint stream);

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

    // ── Stream Flags ────────────────────────────────────────────────────

    internal const uint CU_STREAM_DEFAULT = 0;
    internal const uint CU_STREAM_NON_BLOCKING = 1;
}

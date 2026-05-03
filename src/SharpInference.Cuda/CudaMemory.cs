using SharpInference.Core.Logging;

namespace SharpInference.Cuda;

/// <summary>GPU memory allocation and transfer helpers wrapping CUDA Driver API memory functions.</summary>
public static class CudaMemory
{
    /// <summary>Allocates device memory and returns a device pointer. On OOM, drains the
    /// active streams and trims the device's stream-ordered allocator pool back to the
    /// driver before retrying once. The trim is critical when the streaming weight cache
    /// is in use: <c>cuMemFreeAsync</c> calls return memory to a pool that is invisible
    /// to subsequent sync <c>cuMemAlloc</c> calls until the pool releases it. Without
    /// the trim, an op that should succeed against just-evicted streaming memory will
    /// OOM even though several GB are technically free.</summary>
    public static ulong Allocate(nuint byteSize)
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

    /// <summary>Allocates device memory asynchronously on the given stream.</summary>
    public static ulong AllocateAsync(nuint byteSize, nint stream)
    {
        CudaDriverApi.cuMemAllocAsync(out ulong dptr, byteSize, stream).ThrowOnError();
        return dptr;
    }

    /// <summary>Frees device memory asynchronously on the given stream.</summary>
    public static void FreeAsync(ulong dptr, nint stream)
    {
        if (dptr != 0)
        {
            CudaDriverApi.cuMemFreeAsync(dptr, stream).ThrowOnError();
        }
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

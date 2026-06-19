using HartsyInference.Core.Logging;

namespace HartsyInference.Cuda;

/// <summary>GPU memory allocation and transfer helpers wrapping CUDA Driver API memory functions.</summary>
public static class CudaMemory
{
    /// <summary>The compute stream transient allocations are bound to. Set once by the backend so
    /// <see cref="Allocate"/> can route through the stream-ordered pool. Zero (the default) falls back to the
    /// synchronous allocator — used by tests and by the brief window before the backend wires the stream.</summary>
    private static nint _computeStream;

    /// <summary>Binds the compute stream so transient allocations use the stream-ordered pool (matching their
    /// <c>cuMemFreeAsync</c> frees). Called once from <see cref="CudaBackend"/>'s constructor.</summary>
    public static void SetComputeStream(nint stream) => _computeStream = stream;

    /// <summary>Allocates <b>transient</b> device memory (op outputs, dtype casts, scratch) and returns a device
    /// pointer. Routes through the stream-ordered pool (<c>cuMemAllocAsync</c> on the compute stream) so the memory
    /// is reused by the matching <c>cuMemFreeAsync</c> frees in <see cref="GpuTransferHelper.FreeDevice"/> — the
    /// previous mix of synchronous <c>cuMemAlloc</c> here with async frees there sent freed bytes into the pool
    /// where subsequent sync allocs couldn't see them, so the GPU "filled up" and every op OOM-retried with a full
    /// stream-drain + pool-trim (the cause of the Ideogram-4 ~100s/step thrash on a near-full A100). This mirrors the
    /// fix already applied to the streaming weight cache. Persistent buffers (resident weights, cuBLAS workspaces)
    /// freed via synchronous <see cref="Free"/> must use <see cref="AllocatePersistent"/> instead.</summary>
    public static ulong Allocate(nuint byteSize)
    {
        if (_computeStream != 0)
            return AllocateAsync(byteSize, _computeStream);
        return AllocatePersistent(byteSize);
    }

    /// <summary>Allocates <b>persistent</b> device memory with the synchronous driver allocator (<c>cuMemAlloc</c>),
    /// to be released with the synchronous <see cref="Free"/>. Use for buffers that live for the whole model/session
    /// (resident weights, cuBLAS workspaces) — keeping them out of the churning stream-ordered pool. On OOM, drains
    /// the active streams and trims the pool back to the driver before retrying once.</summary>
    public static ulong AllocatePersistent(nuint byteSize)
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

    /// <summary>Allocates device memory asynchronously on the given stream. Mirrors
    /// <see cref="Allocate"/>'s OOM retry: if the stream-ordered allocator can't satisfy
    /// the request, drain everything and trim the pool, then retry once.</summary>
    public static ulong AllocateAsync(nuint byteSize, nint stream)
    {
        int result = CudaDriverApi.cuMemAllocAsync(out ulong dptr, byteSize, stream);
        if (result == 2) // CUDA_ERROR_OUT_OF_MEMORY
        {
            LogOomDiagnostic("OOM on async first attempt", byteSize);
            GpuTransferHelper.SyncStreamsAndReleasePool();
            int retryResult = CudaDriverApi.cuMemAllocAsync(out dptr, byteSize, stream);
            if (retryResult != 0)
            {
                LogOomDiagnostic("OOM after async sync+pool-trim retry", byteSize);
                retryResult.ThrowOnError();
            }
        }
        else
        {
            result.ThrowOnError();
        }
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

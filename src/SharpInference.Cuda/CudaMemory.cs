namespace SharpInference.Cuda;

/// <summary>GPU memory allocation and transfer helpers wrapping CUDA Driver API memory functions.</summary>
public static class CudaMemory
{
    /// <summary>Allocates device memory and returns a device pointer.</summary>
    public static ulong Allocate(nuint byteSize)
    {
        CudaDriverApi.cuMemAlloc(out ulong dptr, byteSize).ThrowOnError();
        return dptr;
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

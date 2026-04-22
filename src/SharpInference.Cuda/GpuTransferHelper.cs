using SharpInference.Core.Tensors;

namespace SharpInference.Cuda;

/// <summary>Provides temporary GPU memory allocation and CPU-GPU transfer for auto-transfer backend mode. Each transfer allocates a fresh device buffer - no caching, so GPU memory never accumulates.</summary>
internal static unsafe class GpuTransferHelper
{
    /// <summary>Copies a CPU tensor's data to a new GPU buffer. Returns the device pointer. Caller must free with FreeDevice.</summary>
    public static ulong CopyToDevice(Tensor cpuTensor)
    {
        nuint byteSize = (nuint)(cpuTensor.ElementCount * cpuTensor.DType.SizeInBytes);
        ulong dptr = CudaMemory.Allocate(byteSize);
        CudaMemory.CopyHostToDevice(dptr, cpuTensor.DataPointer, byteSize);
        return dptr;
    }

    /// <summary>Copies data from a GPU buffer back into a CPU tensor.</summary>
    public static void CopyToHost(Tensor cpuTensor, ulong gpuPtr, nuint byteSize)
    {
        CudaMemory.CopyDeviceToHost(cpuTensor.DataPointer, gpuPtr, byteSize);
    }

    /// <summary>Allocates a GPU buffer of the specified size.</summary>
    public static ulong AllocateDevice(nuint byteSize)
    {
        return CudaMemory.Allocate(byteSize);
    }

    /// <summary>Frees a GPU buffer. Safe to call with 0 (no-op).</summary>
    public static void FreeDevice(ulong gpuPtr)
    {
        if (gpuPtr != 0)
        {
            CudaMemory.Free(gpuPtr);
        }
    }

    /// <summary>Computes the byte size of a tensor's data.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static nuint ByteSize(Tensor tensor)
    {
        return (nuint)(tensor.ElementCount * tensor.DType.SizeInBytes);
    }
}

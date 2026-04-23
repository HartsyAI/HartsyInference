using System.Runtime.CompilerServices;
using SharpInference.Core.Tensors;

namespace SharpInference.Cuda;

/// <summary>
/// GPU memory transfer helper. No caching — every CopyToDevice allocates a fresh GPU buffer
/// and every FreeDevice frees it. This guarantees correctness by eliminating stale cache entries
/// when CPU memory addresses are reused after tensor disposal.
/// Weight caching optimization can be added later once correctness is verified.
/// </summary>
internal static unsafe class GpuTransferHelper
{
    /// <summary>Allocates a GPU buffer and copies CPU tensor data to it. Always does a fresh upload.</summary>
    public static ulong CopyToDevice(Tensor cpuTensor)
    {
        nuint byteSize = ByteSize(cpuTensor);
        ulong dptr = CudaMemory.Allocate(byteSize);
        CudaMemory.CopyHostToDevice(dptr, cpuTensor.DataPointer, byteSize);
        return dptr;
    }

    /// <summary>Copies data from a GPU buffer back into a CPU tensor.</summary>
    public static void CopyToHost(Tensor cpuTensor, ulong gpuPtr, nuint byteSize)
    {
        CudaMemory.CopyDeviceToHost(cpuTensor.DataPointer, gpuPtr, byteSize);
    }

    /// <summary>Allocates a GPU buffer.</summary>
    public static ulong AllocateDevice(nuint byteSize)
    {
        return CudaMemory.Allocate(byteSize);
    }

    /// <summary>Frees a GPU buffer.</summary>
    public static void FreeDevice(ulong gpuPtr)
    {
        if (gpuPtr != 0)
        {
            CudaMemory.Free(gpuPtr);
        }
    }

    /// <summary>No-op — no cache to evict.</summary>
    public static void EvictAll()
    {
    }

    /// <summary>Computes the byte size of a tensor's data.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static nuint ByteSize(Tensor tensor)
    {
        return (nuint)(tensor.ElementCount * tensor.DType.SizeInBytes);
    }

    /// <summary>Returns cache statistics. Always zero since caching is disabled.</summary>
    public static (long cachedBytes, long hits, long misses) GetStats()
    {
        return (0, 0, 0);
    }
}

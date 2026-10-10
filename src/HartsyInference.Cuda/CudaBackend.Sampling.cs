using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// Device-side sampling for captured decode steps.
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    public bool DeviceSamplingSupported => _kernels is { HasSampleKernel: true };

    /// <inheritdoc />
    public int DeviceSamplingMaxTopK => CudaKernels.SampleMaxK;

    /// <inheritdoc />
    public unsafe ulong AllocDeviceRng(ulong seed, int maxTopK)
    {
        using OpScope _op = EnterOp();
        int k = Math.Clamp(maxTopK, 1, CudaKernels.SampleMaxK);
        ulong handle = CudaMemory.AllocatePersistent((nuint)(16 + 8 * k));
        ulong* init = stackalloc ulong[2] { seed == 0 ? 0x9E3779B97F4A7C15ul : seed, 0ul };
        CudaDriverApi.cuMemcpyHtoDAsync(handle, (nint)init, 16, _stream.Handle).ThrowOnError();
        CudaDriverApi.cuStreamSynchronize(_stream.Handle).ThrowOnError();
        return handle;
    }

    /// <inheritdoc />
    public void FreeDeviceRng(ulong handle)
    {
        if (handle == 0) return;
        using OpScope _op = EnterOp();
        CudaMemory.Free(handle);
    }

    /// <inheritdoc />
    public void SampleTopKInto(ulong outputTokenId, Tensor logits, int topK, float temperature, float topP, float minP, ulong rngState)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SampleTopK");
        if (outputTokenId == 0 || rngState == 0 || logits.DType != DType.F32 || topK < 1 || topK > CudaKernels.SampleMaxK)
            throw new NotSupportedException("SampleTopKInto requires F32 logits, a token-id buffer, an RNG buffer and 1 <= topK <= 64.");
        using OpScope _op = EnterOp();
        EnsureKernels();
        int n = (int)logits.Shape[logits.Shape.Rank - 1];
        ulong pIn = GpuTransferHelper.CopyToDevice(logits);
        try
        {
            ulong vals = rngState + 16;
            ulong idx = vals + (ulong)(4 * topK);
            _kernels!.LaunchTopKLastDim(vals, idx, pIn, 0, 1, n, topK, false, _stream.Handle);
            _kernels.LaunchSampleFromTopK(outputTokenId, vals, idx, topK, temperature, topP, minP, rngState, _stream.Handle);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIn);   // a no-op for the cached activation a decode step passes
        }
    }
}

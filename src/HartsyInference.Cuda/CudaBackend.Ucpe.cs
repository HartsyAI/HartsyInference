using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// IBackend.UcpeTransform on the device: one kernel, one read and one write of the vectors.
public sealed partial class CudaBackend
{
    /// <inheritdoc cref="IBackend.UcpeTransform"/>
    public void UcpeTransform(Tensor output, Tensor input, Tensor matrices, Tensor cos, Tensor sin, int tokens, int heads, int headDim)
    {
        UcpeTransformReference.Validate(output, input, matrices, cos, sin, tokens, heads, headDim);
        using NvtxRange _nvtx = NvtxRange.Push("UcpeTransform");
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasUcpeKernels)
            throw new NotSupportedException("UcpeTransform needs ucpe.ptx in the Ptx folder.");

        ulong pIn = 0, pM = 0, pC = 0, pS = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pM = GpuTransferHelper.CopyToDevice(matrices);
            pC = GpuTransferHelper.CopyToDevice(cos);
            pS = GpuTransferHelper.CopyToDevice(sin);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels.LaunchUcpeTransform(pOut, pIn, pM, pC, pS, tokens, heads, headDim, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pM);
            GpuTransferHelper.FreeDevice(pC);
            GpuTransferHelper.FreeDevice(pS);
            if (!cachedOutput && pOut != 0) GpuTransferHelper.FreeDevice(pOut);
        }
    }
}

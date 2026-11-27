using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Reduction;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// Reductions over strided views on the device.
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    /// <remarks>One launch, one block per group; a group past <see cref="CudaKernels.SumProductsBlockElements"/> elements takes two, through a scratch
    /// buffer of partial sums. The result stays resident.</remarks>
    public void SumProducts(Tensor output, Tensor a, Tensor b, in SumProductsSpec spec)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SumProducts");
        SumProductsGeometry g = SumProductsContract.Validate(output, a, b, spec);
        using OpScope _op = EnterOp();
        EnsureKernels();
        ulong pA = 0, pB = 0, pOut = 0, pScratch = 0;
        bool cached = false;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = ReferenceEquals(a, b) ? pA : GpuTransferHelper.CopyToDevice(b);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            int partials = CudaKernels.SumProductsPartials(g.ElementsPerGroup);
            if (partials > 1) pScratch = GpuTransferHelper.AllocateDevice(CudaKernels.SumProductsScratchBytes(g.Groups, partials));
            _kernels!.LaunchSumProducts(pOut, pA, pB, pScratch, partials, g, spec.Terms, output.DType == DType.F64, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            if (pB != pA) GpuTransferHelper.FreeDevice(pB);
            GpuTransferHelper.FreeDevice(pA);
            if (pScratch != 0) GpuTransferHelper.FreeDevice(pScratch);
        }
    }
}

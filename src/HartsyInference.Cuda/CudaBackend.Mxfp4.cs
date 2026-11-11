using HartsyInference.Core.Backends.Gpu;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// The MXFP4-packed linear (IBackend.Mxfp4LinearF32) on the device. Validates through its Core reference and falls back to it when the module is absent.
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    public void Mxfp4LinearF32(Tensor output, Tensor input, Tensor packedWeight, QuantRecipe recipe)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Mxfp4LinearF32");
        Mxfp4LinearReference.Validate(output, input, packedWeight, recipe, out int n, out int rows, out int cols);
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasMxfp4LinearKernel)
        {
            Mxfp4LinearReference.Apply(output, input, packedWeight, recipe);
            return;
        }
        ulong pO = 0, pI = 0, pW = 0, pS = 0;
        bool cached = false;
        try
        {
            pI = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(packedWeight);
            pS = GpuTransferHelper.CopyToDevice(recipe.Scale!);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pO = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels.LaunchMxfp4Linear(pO, pI, pW, pS, n, rows, cols, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pO, outBytes);
            cached = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pI);
            GpuTransferHelper.FreeDevice(pW);
            GpuTransferHelper.FreeDevice(pS);
            if (!cached) GpuTransferHelper.FreeDevice(pO);
        }
    }
}

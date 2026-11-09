using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// IBackend.DepthwiseConvGlu on the device: one fused kernel over the channels-last activations.
public sealed partial class CudaBackend
{
    /// <inheritdoc cref="IBackend.DepthwiseConvGlu"/>
    public void DepthwiseConvGlu(Tensor output, Tensor input, Tensor weight, Tensor? bias, int frames, int height, int width, int kernel)
    {
        int hidden = DepthwiseConvGluReference.Validate(output, input, weight, bias, frames, height, width, kernel);
        using NvtxRange _nvtx = NvtxRange.Push("DepthwiseConvGlu");
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasDepthwiseGluKernels)
            throw new NotSupportedException("DepthwiseConvGlu needs depthwise_glu.ptx in the Ptx folder.");

        ulong pIn = 0, pW = 0, pB = 0, pOut = 0, taps = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            pB = bias is null ? 0 : GpuTransferHelper.CopyToDevice(bias);
            taps = CastIfNeeded(pW, weight.DType, DType.F32, (int)weight.ElementCount, out ulong tapsCast);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            try
            {
                _kernels.LaunchDepthwiseGlu(pOut, pIn, taps, pB, input.DType == DType.BF16, frames, height, width, hidden, kernel, _stream.Handle);
            }
            finally
            {
                if (tapsCast != 0) CudaMemory.FreeAsync(tapsCast, _stream.Handle);
            }

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            if (pB != 0) GpuTransferHelper.FreeDevice(pB);
            if (!cachedOutput && pOut != 0) GpuTransferHelper.FreeDevice(pOut);
        }
    }
}
